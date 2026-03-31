"""Load data from PostgreSQL into Snowflake warehouse."""

import os

import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from sqlalchemy import create_engine, text

load_dotenv()

BATCH_SIZE = 10_000


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


def get_postgres_engine():
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db = os.environ.get("POSTGRES_DB", "product_intelligence")
    user = os.environ.get("POSTGRES_USER", "postgres")
    password = os.environ.get("POSTGRES_PASSWORD", "postgres")
    return create_engine(f"postgresql://{user}:{password}@{host}:{port}/{db}")


def setup_snowflake(engine):
    """Create database, schema, and tables if they don't exist."""
    with engine.connect() as conn:
        conn.execute(text("CREATE DATABASE IF NOT EXISTS PRODUCT_INTELLIGENCE"))
        conn.execute(text("USE DATABASE PRODUCT_INTELLIGENCE"))
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS RAW"))
        conn.execute(text("USE SCHEMA RAW"))
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS REVIEWS (
                    asin VARCHAR,
                    parent_asin VARCHAR,
                    user_id VARCHAR,
                    rating FLOAT,
                    title TEXT,
                    text TEXT,
                    helpful_votes INTEGER,
                    timestamp TIMESTAMP
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS PRODUCTS (
                    parent_asin VARCHAR,
                    title TEXT,
                    description TEXT,
                    price VARCHAR,
                    category VARCHAR,
                    features TEXT,
                    average_rating FLOAT,
                    rating_number INTEGER
                )
                """
            )
        )
        conn.commit()
    logger.info("Snowflake database, schema, and tables ready")


def load_reviews(sf_engine, pg_engine):
    """Load reviews from PostgreSQL into Snowflake."""
    try:
        with sf_engine.connect() as conn:
            count = conn.execute(text("SELECT COUNT(*) FROM REVIEWS")).scalar()
            if count and count > 0:
                logger.info(f"Snowflake REVIEWS already has {count:,} rows, skipping")
                return
    except Exception:
        logger.info("REVIEWS table does not exist yet, creating...")

    total = pd.read_sql(text("SELECT COUNT(*) FROM reviews"), pg_engine).iloc[0, 0]
    logger.info(f"Loading {total:,} reviews from PostgreSQL into Snowflake...")

    offset = 0
    loaded = 0

    while offset < total:
        query = text(
            f"SELECT * FROM reviews ORDER BY asin LIMIT {BATCH_SIZE} OFFSET {offset}"
        )
        df = pd.read_sql(query, pg_engine)
        if df.empty:
            break

        df = df.drop(columns=["id"], errors="ignore")
        df.to_sql("reviews", sf_engine, if_exists="append", index=False)
        loaded += len(df)
        offset += BATCH_SIZE
        logger.info(f"Loaded {loaded:,} / {total:,} reviews into Snowflake")

    logger.info(f"Done — {loaded:,} reviews in Snowflake")


def load_metadata(sf_engine, pg_engine):
    """Load product metadata from PostgreSQL into Snowflake."""
    try:
        with sf_engine.connect() as conn:
            count = conn.execute(text("SELECT COUNT(*) FROM PRODUCTS")).scalar()
            if count and count > 0:
                logger.info(f"Snowflake PRODUCTS already has {count:,} rows, skipping")
                return
    except Exception:
        logger.info("PRODUCTS table does not exist yet, creating...")

    total = pd.read_sql(text("SELECT COUNT(*) FROM products"), pg_engine).iloc[0, 0]
    logger.info(f"Loading {total:,} products from PostgreSQL into Snowflake...")

    offset = 0
    loaded = 0

    while offset < total:
        query = text(
            "SELECT * FROM products ORDER BY parent_asin"
            f" LIMIT {BATCH_SIZE} OFFSET {offset}"
        )
        df = pd.read_sql(query, pg_engine)
        if df.empty:
            break

        df = df.drop(columns=["id"], errors="ignore")
        df.to_sql("products", sf_engine, if_exists="append", index=False)
        loaded += len(df)
        offset += BATCH_SIZE
        logger.info(f"Loaded {loaded:,} / {total:,} products into Snowflake")

    logger.info(f"Done — {loaded:,} products in Snowflake")


if __name__ == "__main__":
    sf_engine = get_snowflake_engine()
    pg_engine = get_postgres_engine()
    setup_snowflake(sf_engine)
    load_reviews(sf_engine, pg_engine)
    load_metadata(sf_engine, pg_engine)
    logger.info("Snowflake loading complete!")
