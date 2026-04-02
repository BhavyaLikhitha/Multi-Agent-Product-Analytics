"""Evaluate saved root cause classifier checkpoint."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer

from src.models.root_cause_classifier import (
    LABEL_COLS,
    ReviewDataset,
    RootCauseClassifier,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "labeled_reviews.csv"
)
MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "root_cause_classifier"
    / "best_model.pt"
)

SEED = 42
MAX_LEN = 256
BATCH_SIZE = 16


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"]
        logits = model(ids, mask)
        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).int().cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cpu")

    # Load data with same split as training
    df = pd.read_csv(DATA_PATH)
    texts = df["text"].fillna("").values
    labels = df[LABEL_COLS].values.astype(np.float32)

    X_tv, X_test, y_tv, y_test = train_test_split(
        texts, labels, test_size=0.15, random_state=SEED
    )
    val_frac = 0.15 / 0.85
    _, X_val, _, y_val = train_test_split(
        X_tv, y_tv, test_size=val_frac, random_state=SEED
    )

    tokenizer = DistilBertTokenizer.from_pretrained(
        "distilbert-base-uncased"
    )
    val_ds = ReviewDataset(X_val, y_val, tokenizer, MAX_LEN)
    test_ds = ReviewDataset(
        X_test, y_test, tokenizer, MAX_LEN
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # Load model
    model = RootCauseClassifier(num_labels=5).to(device)
    model.load_state_dict(
        torch.load(MODEL_PATH, weights_only=True)
    )
    print("Loaded checkpoint from", MODEL_PATH)

    # Find best threshold on val set
    print("\nFinding best threshold on val set...")
    best_f1 = 0.0
    best_t = 0.5
    for t in [0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        preds, lbls = evaluate(
            model, val_loader, device, threshold=t
        )
        f1 = f1_score(lbls, preds, average="macro")
        print(f"  threshold={t:.2f}  val_f1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    print(f"\nBest threshold: {best_t} (val F1={best_f1:.4f})")

    # Test evaluation
    test_preds, test_labels = evaluate(
        model, test_loader, device, threshold=best_t
    )
    test_f1 = f1_score(
        test_labels, test_preds, average="macro"
    )

    print(f"\nTest macro-F1: {test_f1:.4f}")
    target = "PASS" if test_f1 > 0.70 else "FAIL"
    print(f"Target (>0.70): {target}")

    print("\n-- Classification Report (test set) --")
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
