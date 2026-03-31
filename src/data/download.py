"""Stream Amazon Reviews 2023 from HuggingFace directly into PostgreSQL."""

import os

import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from loguru import logger
from sqlalchemy import create_engine, text

load_dotenv()

BATCH_SIZE = 10_000


def get_postgres_engine():
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db = os.environ.get("POSTGRES_DB", "product_intelligence")
    user = os.environ.get("POSTGRES_USER", "postgres")
    password = os.environ.get("POSTGRES_PASSWORD", "postgres")
    return create_engine(f"postgresql://{user}:{password}@{host}:{port}/{db}")


def stream_reviews(
    engine,
    category: str = "Electronics",
    sample_size: int = 500_000,
):
    """Stream reviews from HuggingFace directly into PostgreSQL."""
    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM reviews")).scalar()
        if count and count > 0:
            logger.info(f"Reviews table already has {count:,} rows, skipping")
            return

    logger.info(f"Streaming {sample_size:,} reviews into PostgreSQL...")
    ds = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        f"raw_review_{category}",
        split="full",
        streaming=True,
        trust_remote_code=True,
    )

    batch = []
    total = 0

    for row in ds:
        batch.append(row)
        if len(batch) >= BATCH_SIZE:
            _insert_review_batch(engine, batch)
            total += len(batch)
            logger.info(f"Inserted {total:,} / {sample_size:,} reviews")
            batch = []
        if total + len(batch) >= sample_size:
            break

    if batch:
        _insert_review_batch(engine, batch)
        total += len(batch)

    logger.info(f"Done — {total:,} reviews in PostgreSQL")


def _insert_review_batch(engine, batch: list[dict]):
    """Insert a batch of reviews into PostgreSQL."""
    df = pd.DataFrame(batch)

    column_map = {"helpful_vote": "helpful_votes"}
    df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")

    keep_cols = [
        "asin",
        "parent_asin",
        "user_id",
        "rating",
        "title",
        "text",
        "helpful_votes",
        "timestamp",
    ]
    df = df[[c for c in keep_cols if c in df.columns]]
    df.to_sql("reviews", engine, if_exists="append", index=False)


def stream_metadata(engine, category: str = "Electronics"):
    """Stream product metadata from HuggingFace into PostgreSQL."""
    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM products")).scalar()
        if count and count > 0:
            logger.info(f"Products table already has {count:,} rows, skipping")
            return

    logger.info("Streaming product metadata into PostgreSQL...")
    ds = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        f"raw_meta_{category}",
        split="full",
        streaming=True,
        trust_remote_code=True,
    )
    # Drop columns with complex nested types that pyarrow can't cast
    drop_cols = [
        c
        for c in ["images", "videos", "bought_together", "subtitle", "author"]
        if c in ds.column_names
    ]
    if drop_cols:
        ds = ds.remove_columns(drop_cols)

    batch = []
    total = 0

    for row in ds:
        batch.append(row)
        if len(batch) >= BATCH_SIZE:
            _insert_metadata_batch(engine, batch)
            total += len(batch)
            logger.info(f"Inserted {total:,} metadata records")
            batch = []

    if batch:
        _insert_metadata_batch(engine, batch)
        total += len(batch)

    logger.info(f"Done — {total:,} products in PostgreSQL")


def _insert_metadata_batch(engine, batch: list[dict]):
    """Insert a batch of product metadata into PostgreSQL."""
    df = pd.DataFrame(batch)

    column_map = {
        "main_category": "category",
        "rating_number": "rating_number",
    }
    df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})

    for col in ["description", "features"]:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: "; ".join(str(i) for i in x) if isinstance(x, list) else x
            )

    keep_cols = [
        "parent_asin",
        "title",
        "description",
        "price",
        "category",
        "features",
        "average_rating",
        "rating_number",
    ]
    df = df[[c for c in keep_cols if c in df.columns]]
    df = df.drop_duplicates(subset=["parent_asin"])
    df.to_sql("products", engine, if_exists="append", index=False)


if __name__ == "__main__":
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login

        login(token=hf_token)

    engine = get_postgres_engine()

    # Create tables first
    from src.data.load_postgres import create_tables

    create_tables(engine)

    stream_reviews(engine)
    stream_metadata(engine)
    logger.info("Download complete — all data in PostgreSQL!")
