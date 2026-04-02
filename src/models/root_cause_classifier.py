"""
Root Cause Classifier — multi-label DistilBERT model.

Classifies e-commerce review text into five root-cause
categories: defect, shipping, description, size, price.

Usage:
    poetry run python src/models/root_cause_classifier.py
"""

from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertModel, DistilBertTokenizer

# ── Paths ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "labeled_reviews.csv"
SAVE_DIR = PROJECT_ROOT / "models" / "root_cause_classifier"

LABEL_COLS = [
    "defect",
    "shipping",
    "description",
    "size",
    "price",
]

# ── Hyper-parameters ──────────────────────────────────
MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 10
LR = 2e-5
SEED = 42
VAL_SIZE = 0.15
TEST_SIZE = 0.15


# ── Dataset ────────────────────────────────────────────
class ReviewDataset(Dataset):
    """Tokenised review dataset for multi-label task."""

    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": (encoding["attention_mask"].squeeze(0)),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float),
        }


# ── Model ──────────────────────────────────────────────
class RootCauseClassifier(nn.Module):
    """DistilBERT + linear head for multi-label output."""

    def __init__(self, num_labels: int = 5, dropout: float = 0.3):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        hidden = self.distilbert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask):
        output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        cls_output = output.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return logits


# ── Helpers ────────────────────────────────────────────
def compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    """Compute inverse-frequency weights per label."""
    pos_counts = labels.sum(axis=0)
    neg_counts = len(labels) - pos_counts
    # weight = neg / pos (higher weight for rare classes)
    weights = neg_counts / np.clip(pos_counts, 1, None)
    # Cap weights to avoid extreme values
    weights = np.clip(weights, 1.0, 15.0)
    return torch.tensor(weights, dtype=torch.float)


def build_loader(texts, labels, tokenizer, shuffle=False):
    """Create a DataLoader from raw texts and labels."""
    ds = ReviewDataset(texts, labels, tokenizer, MAX_LEN)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Run one training epoch; return average loss."""
    model.train()
    total_loss = 0.0
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(ids, mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold=0.5):
    """Evaluate model; return loss, predictions, targets."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(ids, mask)
        loss = criterion(logits, labels)
        total_loss += loss.item()

        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).int().cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    avg_loss = total_loss / len(loader)
    return avg_loss, all_preds, all_labels


def find_best_threshold(model, loader, criterion, device):
    """Try thresholds from 0.3 to 0.6 and pick best F1."""
    best_f1 = 0.0
    best_t = 0.5
    for t in [0.3, 0.35, 0.4, 0.45, 0.5]:
        _, preds, labels = evaluate(model, loader, criterion, device, threshold=t)
        f1 = f1_score(labels, preds, average="macro")
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t, best_f1


# ── Main ───────────────────────────────────────────────
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Load data ──────────────────────────────────────
    print(f"Loading data from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    texts = df["text"].fillna("").values
    labels = df[LABEL_COLS].values.astype(np.float32)
    print(f"  Samples: {len(df)}")

    # ── Compute class weights ──────────────────────────
    class_weights = compute_class_weights(labels)
    print(f"  Class weights: {dict(zip(LABEL_COLS, class_weights.tolist()))}")

    # ── Train / val / test split ───────────────────────
    (
        X_train_val,
        X_test,
        y_train_val,
        y_test,
    ) = train_test_split(
        texts,
        labels,
        test_size=TEST_SIZE,
        random_state=SEED,
    )
    val_frac = VAL_SIZE / (1 - TEST_SIZE)
    (
        X_train,
        X_val,
        y_train,
        y_val,
    ) = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_frac,
        random_state=SEED,
    )
    print(f"  Train: {len(X_train)}  " f"Val: {len(X_val)}  " f"Test: {len(X_test)}")

    # ── Tokenizer & loaders ────────────────────────────
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    train_loader = build_loader(X_train, y_train, tokenizer, shuffle=True)
    val_loader = build_loader(X_val, y_val, tokenizer)
    test_loader = build_loader(X_test, y_test, tokenizer)

    # ── Model, loss, optimiser ─────────────────────────
    model = RootCauseClassifier(num_labels=len(LABEL_COLS)).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # ── MLflow setup ───────────────────────────────────
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("root_cause_classifier")
    mlflow.start_run()
    mlflow.log_params(
        {
            "max_len": MAX_LEN,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "lr": LR,
            "model": "distilbert-base-uncased",
            "dropout": 0.3,
            "class_weights": "inverse_frequency",
        }
    )

    # ── Training loop ──────────────────────────────────
    best_f1 = 0.0
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    best_path = SAVE_DIR / "best_model.pt"

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )
        val_loss, val_preds, val_labels = evaluate(model, val_loader, criterion, device)
        macro_f1 = f1_score(val_labels, val_preds, average="macro")

        print(
            f"Epoch {epoch}/{EPOCHS}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"macro_f1={macro_f1:.4f}"
        )
        mlflow.log_metrics(
            {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_macro_f1": macro_f1,
            },
            step=epoch,
        )

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            torch.save(model.state_dict(), best_path)
            print(f"  -> Saved best model (F1={best_f1:.4f})")

    # ── Find optimal threshold ─────────────────────────
    print("\nLoading best checkpoint...")
    model.load_state_dict(torch.load(best_path, weights_only=True))

    print("Finding optimal threshold on val set...")
    best_threshold, val_f1 = find_best_threshold(model, val_loader, criterion, device)
    print(f"  Best threshold: {best_threshold} " f"(val F1={val_f1:.4f})")

    # ── Test evaluation ────────────────────────────────
    test_loss, test_preds, test_labels = evaluate(
        model,
        test_loader,
        criterion,
        device,
        threshold=best_threshold,
    )
    test_macro_f1 = f1_score(test_labels, test_preds, average="macro")

    mlflow.log_metrics(
        {
            "test_loss": test_loss,
            "test_macro_f1": test_macro_f1,
            "best_threshold": best_threshold,
        }
    )
    mlflow.log_artifact(str(best_path))
    mlflow.end_run()

    print(f"\nTest macro-F1: {test_macro_f1:.4f}")
    target_met = "PASS" if test_macro_f1 > 0.70 else "FAIL"
    print(f"Target (>0.70): {target_met}")

    print("\n── Classification Report (test set) ──")
    print(
        classification_report(
            test_labels,
            test_preds,
            target_names=LABEL_COLS,
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()
