"""Load raw review and metadata parquet files into Snowflake."""

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from sqlalchemy import create_engine, text

load_dotenv()

RAW_DIR = Path("data/raw")


def get_snowflake_engine():
    account = os.environ["SNOWFLAKE_ACCOUNT"]
    user = os.environ["SNOWFLAKE_USER"]
    password = os.environ["SNOWFLAKE_PASSWORD"]
    database = os.environ.get("SNOWFLAKE_DATABASE", "PRODUCT_INTELLIGENCE")

    url = (
        f"snowflake://{user}:{password}@{account}/{database}/RAW"
        f"?warehouse=COMPUTE_WH"
    )
    return create_engine(url)


def setup_snowflake(engine):
    """Create database and schema if they don't exist."""
    with engine.connect() as conn:
        conn.execute(text("CREATE DATABASE IF NOT EXISTS PRODUCT_INTELLIGENCE"))
        conn.execute(text("USE DATABASE PRODUCT_INTELLIGENCE"))
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS RAW"))
        conn.execute(text("USE SCHEMA RAW"))
        conn.commit()
    logger.info("Snowflake database and schema ready")


def load_reviews(engine):
    """Load reviews parquet into Snowflake REVIEWS table."""
    path = RAW_DIR / "reviews.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Reviews file not found: {path}")

    logger.info("Reading reviews parquet...")
    df = pd.read_parquet(path)

    # Standardize column names
    column_map = {
        "rating": "rating",
        "title": "title",
        "text": "text",
        "helpful_vote": "helpful_votes",
        "timestamp": "timestamp",
        "asin": "asin",
        "user_id": "user_id",
        "parent_asin": "parent_asin",
    }
    df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})

    logger.info(f"Loading {len(df):,} reviews into Snowflake...")
    df.to_sql("reviews", engine, if_exists="replace", index=False, chunksize=10000)

    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM reviews")).scalar()
    logger.info(f"Loaded {count:,} reviews into Snowflake")


def load_metadata(engine):
    """Load product metadata parquet into Snowflake PRODUCTS table."""
    path = RAW_DIR / "metadata.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")

    logger.info("Reading metadata parquet...")
    df = pd.read_parquet(path)

    column_map = {
        "main_category": "category",
        "average_rating": "average_rating",
        "rating_number": "rating_number",
        "title": "title",
        "description": "description",
        "price": "price",
        "features": "features",
        "parent_asin": "parent_asin",
    }
    df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})

    # Flatten list columns to strings for Snowflake compatibility
    for col in df.columns:
        if df[col].apply(type).eq(list).any():
            df[col] = df[col].apply(
                lambda x: "; ".join(str(i) for i in x) if isinstance(x, list) else x
            )

    logger.info(f"Loading {len(df):,} products into Snowflake...")
    df.to_sql("products", engine, if_exists="replace", index=False, chunksize=10000)

    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM products")).scalar()
    logger.info(f"Loaded {count:,} products into Snowflake")


if __name__ == "__main__":
    engine = get_snowflake_engine()
    setup_snowflake(engine)
    load_reviews(engine)
    load_metadata(engine)
    logger.info("Snowflake loading complete!")
