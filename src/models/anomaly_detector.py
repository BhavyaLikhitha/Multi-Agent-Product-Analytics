"""Autoencoder-based anomaly detector for product review patterns.

Trains on product_features table (unsupervised). Anomaly score is
the reconstruction error (MSE). Products above the 95th-percentile
threshold trigger quality alerts written to the alerts table.
"""

import json
import os
from datetime import datetime
from pathlib import Path

import joblib
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from dotenv import load_dotenv
from loguru import logger
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine
from torch.utils.data import DataLoader, TensorDataset

load_dotenv()

# ── constants ────────────────────────────────────────────────────
FEATURE_COLS = [
    "daily_sentiment_avg",
    "review_velocity",
    "negative_ratio",
    "complaint_keywords",
]
MODEL_DIR = Path("models/anomaly_detector")
EPOCHS = 50
BATCH_SIZE = 256
LR = 1e-3
DROPOUT = 0.2
PERCENTILE_THRESHOLD = 95
RANDOM_SEED = 42


# ── database helper ──────────────────────────────────────────────
def get_postgres_engine():
    """Build SQLAlchemy engine from environment variables."""
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db = os.environ.get("POSTGRES_DB", "product_intelligence")
    user = os.environ.get("POSTGRES_USER", "postgres")
    pw = os.environ.get("POSTGRES_PASSWORD", "postgres")
    return create_engine(f"postgresql://{user}:{pw}@{host}:{port}/{db}")


# ── model ────────────────────────────────────────────────────────
class ReviewAutoencoder(nn.Module):
    """Symmetric autoencoder: in→32→16→8→16→32→in."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(16, 8),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(32, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


# ── data loading ─────────────────────────────────────────────────
def load_features(engine) -> pd.DataFrame:
    """Read product_features from PostgreSQL."""
    query = (
        "SELECT asin, date, "
        "daily_sentiment_avg, review_velocity, "
        "negative_ratio, complaint_keywords "
        "FROM product_features"
    )
    df = pd.read_sql(query, engine)
    logger.info(f"Loaded {len(df):,} rows from product_features")
    return df


# ── training ─────────────────────────────────────────────────────
def fit_scaler(
    df: pd.DataFrame,
) -> tuple[StandardScaler, np.ndarray]:
    """Fit StandardScaler on feature columns, return scaled array."""
    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURE_COLS].values)
    return scaler, X


def build_dataloader(X: np.ndarray, batch_size: int = BATCH_SIZE) -> DataLoader:
    """Wrap numpy array into a PyTorch DataLoader."""
    tensor = torch.tensor(X, dtype=torch.float32)
    dataset = TensorDataset(tensor, tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train(
    model: ReviewAutoencoder,
    loader: DataLoader,
    epochs: int = EPOCHS,
    lr: float = LR,
) -> list[float]:
    """Train autoencoder and return per-epoch losses."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    epoch_losses: list[float] = []
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            output = model(batch_x)
            loss = criterion(output, batch_x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1

        avg_loss = running_loss / max(n_batches, 1)
        epoch_losses.append(avg_loss)
        if epoch % 10 == 0 or epoch == 1:
            logger.info(f"Epoch {epoch}/{epochs}  loss={avg_loss:.6f}")
    return epoch_losses


# ── scoring ──────────────────────────────────────────────────────
def compute_anomaly_scores(
    model: ReviewAutoencoder,
    X: np.ndarray,
) -> np.ndarray:
    """Return per-sample MSE reconstruction error."""
    device = next(model.parameters()).device
    model.eval()
    tensor = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        recon = model(tensor)
    errors = (tensor - recon).pow(2).mean(dim=1).cpu().numpy()
    return errors


def detect_anomalies(
    df: pd.DataFrame,
    scores: np.ndarray,
    percentile: int = PERCENTILE_THRESHOLD,
) -> tuple[pd.DataFrame, float]:
    """Flag rows above percentile threshold as anomalies."""
    threshold = float(np.percentile(scores, percentile))
    df = df.copy()
    df["anomaly_score"] = scores
    df["is_anomaly"] = scores > threshold
    anomalies = df[df["is_anomaly"]].copy()
    logger.info(
        f"Threshold ({percentile}th pctl): {threshold:.6f} — "
        f"{len(anomalies):,} anomalies detected"
    )
    return anomalies, threshold


# ── persistence ──────────────────────────────────────────────────
def save_artifacts(
    model: ReviewAutoencoder,
    scaler: StandardScaler,
    threshold: float,
):
    """Save model weights, scaler, and threshold to disk."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "autoencoder.pt"
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")

    scaler_path = MODEL_DIR / "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    logger.info(f"Scaler saved to {scaler_path}")

    meta_path = MODEL_DIR / "metadata.json"
    meta = {
        "threshold": threshold,
        "feature_cols": FEATURE_COLS,
        "input_dim": len(FEATURE_COLS),
        "percentile": PERCENTILE_THRESHOLD,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info(f"Metadata saved to {meta_path}")


def write_alerts(
    engine,
    anomalies: pd.DataFrame,
    threshold: float,
):
    """Write detected anomalies to the alerts table."""
    if anomalies.empty:
        logger.info("No anomalies to write.")
        return

    now = datetime.utcnow()
    records = []
    for _, row in anomalies.iterrows():
        details = json.dumps(
            {
                "anomaly_score": round(float(row["anomaly_score"]), 6),
                "threshold": round(threshold, 6),
                "date": str(row["date"]),
            }
        )
        severity = "critical" if row["anomaly_score"] > threshold * 2 else "warning"
        records.append(
            {
                "product_id": row["asin"],
                "alert_type": "quality_anomaly",
                "severity": severity,
                "detected_at": now,
                "details": details,
            }
        )

    alert_df = pd.DataFrame(records)
    alert_df.to_sql(
        "alerts",
        engine,
        if_exists="append",
        index=False,
        chunksize=1000,
    )
    logger.info(f"Wrote {len(alert_df):,} alerts to PostgreSQL")


# ── mlflow logging ───────────────────────────────────────────────
def log_to_mlflow(
    model: ReviewAutoencoder,
    epoch_losses: list[float],
    threshold: float,
    n_anomalies: int,
    n_total: int,
):
    """Log params, metrics, and model artifact to MLflow."""
    mlflow.set_experiment("anomaly_detector")
    with mlflow.start_run(run_name="autoencoder_train"):
        mlflow.log_params(
            {
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "lr": LR,
                "dropout": DROPOUT,
                "percentile_threshold": PERCENTILE_THRESHOLD,
                "feature_cols": json.dumps(FEATURE_COLS),
                "architecture": "in-32-16-8-16-32-in",
            }
        )
        mlflow.log_metrics(
            {
                "final_loss": epoch_losses[-1],
                "threshold": threshold,
                "n_anomalies": n_anomalies,
                "n_total": n_total,
                "anomaly_rate": n_anomalies / max(n_total, 1),
            }
        )
        for i, loss in enumerate(epoch_losses, 1):
            mlflow.log_metric("train_loss", loss, step=i)

        mlflow.pytorch.log_model(model, "autoencoder")
        mlflow.log_artifacts(str(MODEL_DIR), "artifacts")
    logger.info("MLflow run logged.")


# ── main pipeline ────────────────────────────────────────────────
def run():
    """End-to-end: load, train, score, alert, log."""
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    engine = get_postgres_engine()

    # 1. Load features
    df = load_features(engine)
    if df.empty:
        logger.error("No data in product_features. Exiting.")
        return

    # 2. Scale
    scaler, X = fit_scaler(df)
    logger.info(f"Features scaled: {X.shape[0]:,} rows, " f"{X.shape[1]} cols")

    # 3. Build model and train
    input_dim = X.shape[1]
    model = ReviewAutoencoder(input_dim)
    loader = build_dataloader(X)
    epoch_losses = train(model, loader)

    # 4. Compute anomaly scores
    scores = compute_anomaly_scores(model, X)
    anomalies, threshold = detect_anomalies(df, scores)

    # 5. Save artifacts
    save_artifacts(model, scaler, threshold)

    # 6. Write alerts to PostgreSQL
    write_alerts(engine, anomalies, threshold)

    # 7. Log to MLflow
    log_to_mlflow(
        model,
        epoch_losses,
        threshold,
        n_anomalies=len(anomalies),
        n_total=len(df),
    )

    logger.info("Anomaly detection pipeline complete.")


if __name__ == "__main__":
    run()
