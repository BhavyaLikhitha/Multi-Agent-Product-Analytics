"""
Helpfulness Predictor: predicts helpful_votes for reviews.

Architecture: 8-feature feedforward NN (64 -> 32 -> 16 -> 1)
with BatchNorm, ReLU, and Dropout.
Target metric: MAE < 2.0
"""

import os
import pickle
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine, text
from torch.utils.data import DataLoader, TensorDataset

# ── paths ────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
SAVE_DIR = ROOT / "models" / "helpfulness_predictor"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ── DB connection ────────────────────────────────────────────
PG_HOST = os.getenv("POSTGRES_HOST", "localhost")
PG_PORT = os.getenv("POSTGRES_PORT", "5432")
PG_DB = os.getenv("POSTGRES_DB", "product_intelligence")
PG_USER = os.getenv("POSTGRES_USER", "postgres")
PG_PASS = os.getenv("POSTGRES_PASSWORD", "postgres")

PG_URL = f"postgresql://{PG_USER}:{PG_PASS}" f"@{PG_HOST}:{PG_PORT}/{PG_DB}"

# ── hyperparams ──────────────────────────────────────────────
EPOCHS = 30
BATCH_SIZE = 512
LR = 1e-3
DROPOUT = 0.3
SAMPLE_SIZE = 100_000
RANDOM_STATE = 42
FEATURE_COLS = [
    "text_length",
    "word_count",
    "rating",
    "title_length",
    "has_exclamation",
    "has_question",
    "avg_word_length",
    "uppercase_ratio",
]


# ── model ────────────────────────────────────────────────────
class HelpfulnessPredictor(nn.Module):
    """Feedforward NN: 8 -> 64 -> 32 -> 16 -> 1."""

    def __init__(
        self,
        input_dim: int = 8,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ── feature engineering ──────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create model features from raw review columns."""
    text = df["text"].fillna("")
    title = df["title"].fillna("")

    feats = pd.DataFrame()
    feats["text_length"] = text.str.len()
    feats["word_count"] = text.str.split().str.len().fillna(0)
    feats["rating"] = df["rating"].fillna(3).astype(float)
    feats["title_length"] = title.str.len()
    feats["has_exclamation"] = text.str.contains("!", regex=False).astype(int)
    feats["has_question"] = text.str.contains("?", regex=False).astype(int)

    words = text.str.split()
    feats["avg_word_length"] = words.apply(
        lambda w: (np.mean([len(x) for x in w]) if w and len(w) > 0 else 0.0)
    )

    alpha_counts = text.str.findall(r"[A-Za-z]").str.len()
    upper_counts = text.str.findall(r"[A-Z]").str.len()
    feats["uppercase_ratio"] = (upper_counts / alpha_counts.replace(0, 1)).fillna(0.0)

    return feats


# ── data loading ─────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    """Load up to SAMPLE_SIZE reviews from PostgreSQL."""
    engine = create_engine(PG_URL)
    query = text(
        "SELECT rating, title, text, helpful_votes, timestamp "
        "FROM reviews "
        "ORDER BY RANDOM() "
        f"LIMIT {SAMPLE_SIZE}"
    )
    logger.info(f"Loading {SAMPLE_SIZE} reviews from PostgreSQL...")
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    logger.info(f"Loaded {len(df)} reviews.")
    return df


def prepare_splits(df: pd.DataFrame):
    """Engineer features, scale, and split 70/15/15."""
    X = engineer_features(df)
    y = df["helpful_votes"].values.astype(np.float32)

    # train / temp split (70 / 30)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE
    )
    # val / test split (50/50 of 30% = 15/15)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
    )

    # fit scaler on train only
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)
    X_test_sc = scaler.transform(X_test)

    # save scaler
    scaler_path = SAVE_DIR / "scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    logger.info(f"Scaler saved to {scaler_path}")

    def _to_tensors(Xn, yn):
        return (
            torch.tensor(Xn, dtype=torch.float32),
            torch.tensor(yn, dtype=torch.float32),
        )

    train_t = _to_tensors(X_train_sc, y_train)
    val_t = _to_tensors(X_val_sc, y_val)
    test_t = _to_tensors(X_test_sc, y_test)
    return train_t, val_t, test_t, scaler


def _make_loader(tensors, batch_size, shuffle=False):
    ds = TensorDataset(*tensors)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


# ── training loop ────────────────────────────────────────────
def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
) -> nn.Module:
    """Train model and log metrics to MLflow."""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )

    best_val_mae = float("inf")

    for epoch in range(1, EPOCHS + 1):
        # ---- train ----
        model.train()
        train_loss = 0.0
        train_ae = 0.0
        n_train = 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            preds = model(X_b)
            loss = criterion(preds, y_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y_b)
            train_ae += (preds - y_b).abs().sum().item()
            n_train += len(y_b)

        train_mse = train_loss / n_train
        train_mae = train_ae / n_train

        # ---- validate ----
        model.eval()
        val_loss = 0.0
        val_ae = 0.0
        n_val = 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                preds = model(X_b)
                loss = criterion(preds, y_b)
                val_loss += loss.item() * len(y_b)
                val_ae += (preds - y_b).abs().sum().item()
                n_val += len(y_b)

        val_mse = val_loss / n_val
        val_mae = val_ae / n_val
        scheduler.step(val_mse)

        current_lr = optimizer.param_groups[0]["lr"]
        mlflow.log_metrics(
            {
                "train_mse": train_mse,
                "train_mae": train_mae,
                "val_mse": val_mse,
                "val_mae": val_mae,
                "lr": current_lr,
            },
            step=epoch,
        )

        logger.info(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train MAE={train_mae:.4f} | "
            f"val MAE={val_mae:.4f} | "
            f"lr={current_lr:.1e}"
        )

        # checkpoint best
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            ckpt = SAVE_DIR / "best_model.pt"
            torch.save(model.state_dict(), ckpt)
            logger.info(f"  -> new best val MAE={val_mae:.4f}, " f"saved to {ckpt}")

    return model


# ── evaluation ───────────────────────────────────────────────
def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> dict:
    """Compute test MSE and MAE."""
    model.eval()
    total_se = 0.0
    total_ae = 0.0
    n = 0
    with torch.no_grad():
        for X_b, y_b in test_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            preds = model(X_b)
            total_se += ((preds - y_b) ** 2).sum().item()
            total_ae += (preds - y_b).abs().sum().item()
            n += len(y_b)
    metrics = {
        "test_mse": total_se / n,
        "test_mae": total_ae / n,
    }
    return metrics


# ── main ─────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # load and prepare data
    df = load_data()
    train_t, val_t, test_t, scaler = prepare_splits(df)

    train_loader = _make_loader(train_t, BATCH_SIZE, shuffle=True)
    val_loader = _make_loader(val_t, BATCH_SIZE)
    test_loader = _make_loader(test_t, BATCH_SIZE)

    # init model
    model = HelpfulnessPredictor(
        input_dim=len(FEATURE_COLS),
        dropout=DROPOUT,
    ).to(device)

    logger.info(f"Model architecture:\n{model}")

    # MLflow tracking
    mlflow.set_experiment("helpfulness_predictor")
    with mlflow.start_run(run_name="feedforward_v1"):
        mlflow.log_params(
            {
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "lr": LR,
                "dropout": DROPOUT,
                "sample_size": SAMPLE_SIZE,
                "architecture": "8-64-32-16-1",
                "features": ", ".join(FEATURE_COLS),
            }
        )

        # train
        model = train(model, train_loader, val_loader, device)

        # reload best checkpoint for final eval
        best_ckpt = SAVE_DIR / "best_model.pt"
        if best_ckpt.exists():
            model.load_state_dict(torch.load(best_ckpt, weights_only=True))
            logger.info("Loaded best checkpoint for eval.")

        # evaluate on test set
        metrics = evaluate(model, test_loader, device)
        mlflow.log_metrics(metrics)
        logger.info(
            f"Test MSE={metrics['test_mse']:.4f} | "
            f"Test MAE={metrics['test_mae']:.4f}"
        )

        if metrics["test_mae"] < 2.0:
            logger.info("Target MAE < 2.0 achieved!")
        else:
            logger.warning(
                f"MAE {metrics['test_mae']:.4f} >= 2.0. "
                "Consider tuning hyperparameters."
            )

        # log model artifact
        mlflow.pytorch.log_model(model, "model")
        logger.info(f"Artifacts saved to {SAVE_DIR}")


if __name__ == "__main__":
    main()
