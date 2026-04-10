"""Evidently AI drift monitoring for review features.

Compares recent product features against a reference
window to detect data drift that could degrade model
performance.
"""

import os

import pandas as pd
from dotenv import load_dotenv
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset
from loguru import logger
from sqlalchemy import create_engine, text

load_dotenv()

FEATURE_COLS = [
    "daily_sentiment_avg",
    "review_velocity",
    "negative_ratio",
    "complaint_keywords",
]

REPORT_PATH = "reports/drift_report.html"


def get_engine():
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db = os.environ.get("POSTGRES_DB", "product_intelligence")
    user = os.environ.get("POSTGRES_USER", "postgres")
    pw = os.environ.get("POSTGRES_PASSWORD", "postgres")
    return create_engine(f"postgresql://{user}:{pw}@{host}:{port}/{db}")


def load_features(engine):
    """Load product features from PostgreSQL."""
    query = text(
        "SELECT asin, date, "
        "daily_sentiment_avg, review_velocity, "
        "negative_ratio, complaint_keywords "
        "FROM product_features "
        "ORDER BY date"
    )
    df = pd.read_sql(query, engine)
    df["date"] = pd.to_datetime(df["date"])
    logger.info(f"Loaded {len(df):,} feature rows")
    return df


def split_reference_current(df, split_pct=0.7):
    """Split data into reference and current windows."""
    dates = sorted(df["date"].unique())
    split_idx = int(len(dates) * split_pct)
    split_date = dates[split_idx]

    reference = df[df["date"] < split_date]
    current = df[df["date"] >= split_date]

    logger.info(f"Reference: {len(reference):,} rows " f"(before {split_date.date()})")
    logger.info(f"Current: {len(current):,} rows " f"(after {split_date.date()})")
    return reference, current


def run_drift_report(reference, current):
    """Generate Evidently drift report."""
    data_def = DataDefinition(numerical_columns=FEATURE_COLS)

    ref_dataset = Dataset.from_pandas(
        reference[FEATURE_COLS],
        data_definition=data_def,
    )
    cur_dataset = Dataset.from_pandas(
        current[FEATURE_COLS],
        data_definition=data_def,
    )

    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(
        reference_data=ref_dataset,
        current_data=cur_dataset,
    )
    return snapshot


def main():
    engine = get_engine()
    df = load_features(engine)

    if df.empty:
        logger.warning("No features found")
        return

    reference, current = split_reference_current(df)

    if reference.empty or current.empty:
        logger.warning("Not enough data for drift check")
        return

    report = run_drift_report(reference, current)

    os.makedirs("reports", exist_ok=True)
    report.save_html(REPORT_PATH)
    logger.info(f"Drift report saved to {REPORT_PATH}")

    # Save lightweight JSON summary for cloud dashboard
    import json as _json
    import re as _re

    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        html = f.read()
    match = _re.search(r"var metric_\w+ = (\{.*)", html)
    if match:
        raw = match.group(1)
        # Find valid JSON end
        for end in range(len(raw), 0, -1):
            try:
                parsed = _json.loads(raw[:end])
                break
            except _json.JSONDecodeError:
                continue
        else:
            parsed = None

        if parsed:
            widgets = parsed.get("widgets", [{}])[0].get("widgets", [])
            summary = {"dataset_drift_detected": False, "drift_threshold": 0.5,
                       "total_columns": 0, "drifted_columns": 0,
                       "drift_share": 0.0, "columns": []}
            for w in widgets:
                counters = w.get("params", {}).get("counters", [])
                for c in counters:
                    if "NOT detected" in c.get("label", ""):
                        summary["dataset_drift_detected"] = False
                    elif "IS detected" in c.get("label", ""):
                        summary["dataset_drift_detected"] = True
                data_rows = w.get("params", {}).get("data", [])
                for row in data_rows:
                    summary["columns"].append({
                        "name": row.get("column_name", ""),
                        "drift_detected": row.get("data_drift") == "Detected",
                        "drift_score": row.get("drift_score", 0),
                        "stattest": row.get("stattest_name", ""),
                    })
            summary["total_columns"] = len(summary["columns"])
            summary["drifted_columns"] = sum(
                1 for c in summary["columns"] if c["drift_detected"]
            )
            if summary["total_columns"] > 0:
                summary["drift_share"] = (
                    summary["drifted_columns"] / summary["total_columns"]
                )

            summary_path = REPORT_PATH.replace(".html", "_summary.json")
            with open(summary_path, "w") as f:
                _json.dump(summary, f, indent=2)
            logger.info(f"Drift summary saved to {summary_path}")

    logger.info("Open in browser to view results")


if __name__ == "__main__":
    main()
