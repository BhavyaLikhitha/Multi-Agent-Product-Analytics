"""Load subset of reviews and metadata into PostgreSQL application database."""

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    create_engine,
    text,
)
from sqlalchemy.orm import DeclarativeBase

load_dotenv()

RAW_DIR = Path("data/raw")
REVIEW_SUBSET_SIZE = 1_000_000


class Base(DeclarativeBase):
    pass


class Review(Base):
    __tablename__ = "reviews"

    id = Column(Integer, primary_key=True, autoincrement=True)
    asin = Column(String(20), index=True)
    parent_asin = Column(String(20), index=True)
    user_id = Column(String(50))
    rating = Column(Float, index=True)
    title = Column(Text)
    text = Column(Text)
    helpful_votes = Column(Integer, default=0)
    timestamp = Column(DateTime, index=True)

    __table_args__ = (
        Index("ix_reviews_asin_timestamp", "asin", "timestamp"),
        Index("ix_reviews_asin_rating", "asin", "rating"),
    )


class Product(Base):
    __tablename__ = "products"

    id = Column(Integer, primary_key=True, autoincrement=True)
    parent_asin = Column(String(20), unique=True, index=True)
    title = Column(Text)
    description = Column(Text)
    price = Column(String(50))
    category = Column(String(100))
    features = Column(Text)
    average_rating = Column(Float)
    rating_number = Column(Integer)


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    product_id = Column(String(20), index=True)
    alert_type = Column(String(50))
    severity = Column(String(20))
    detected_at = Column(DateTime)
    details = Column(Text)


class ApplicationTracker(Base):
    __tablename__ = "application_tracker"

    id = Column(Integer, primary_key=True, autoincrement=True)
    product_id = Column(String(20), unique=True, index=True)
    status = Column(String(30))
    last_analyzed = Column(DateTime)
    summary = Column(Text)


def get_postgres_engine():
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db = os.environ.get("POSTGRES_DB", "product_intelligence")
    user = os.environ.get("POSTGRES_USER", "postgres")
    password = os.environ.get("POSTGRES_PASSWORD", "postgres")

    url = f"postgresql://{user}:{password}@{host}:{port}/{db}"
    return create_engine(url)


def create_tables(engine):
    """Create all tables."""
    Base.metadata.create_all(engine)
    logger.info("PostgreSQL tables created")


def load_reviews(engine):
    """Load latest 1M reviews into PostgreSQL."""
    path = RAW_DIR / "reviews.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Reviews file not found: {path}")

    logger.info("Reading reviews parquet...")
    df = pd.read_parquet(path)

    # Rename columns to match schema
    column_map = {
        "helpful_vote": "helpful_votes",
    }
    df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})

    # Sort by timestamp descending and take latest subset
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
        df = df.sort_values("timestamp", ascending=False)

    df = df.head(REVIEW_SUBSET_SIZE)

    # Keep only columns that match the table
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

    logger.info(f"Loading {len(df):,} reviews into PostgreSQL...")
    df.to_sql("reviews", engine, if_exists="append", index=False, chunksize=5000)

    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM reviews")).scalar()
    logger.info(f"Loaded {count:,} reviews into PostgreSQL")


def load_products(engine):
    """Load product metadata into PostgreSQL."""
    path = RAW_DIR / "metadata.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")

    logger.info("Reading metadata parquet...")
    df = pd.read_parquet(path)

    column_map = {
        "main_category": "category",
        "rating_number": "rating_number",
    }
    df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})

    # Flatten list columns
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

    logger.info(f"Loading {len(df):,} products into PostgreSQL...")
    df.to_sql("products", engine, if_exists="append", index=False, chunksize=5000)

    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM products")).scalar()
    logger.info(f"Loaded {count:,} products into PostgreSQL")


if __name__ == "__main__":
    engine = get_postgres_engine()
    create_tables(engine)
    load_reviews(engine)
    load_products(engine)
    logger.info("PostgreSQL loading complete!")
