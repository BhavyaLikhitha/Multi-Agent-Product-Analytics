"""Migrate local PostgreSQL data to Neon cloud database."""

import os

import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from sqlalchemy import create_engine, text

load_dotenv()

BATCH_SIZE = 5000


def get_local_engine():
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db = os.environ.get(
        "POSTGRES_DB", "product_intelligence"
    )
    user = os.environ.get("POSTGRES_USER", "postgres")
    pw = os.environ.get("POSTGRES_PASSWORD", "postgres")
    return create_engine(
        f"postgresql://{user}:{pw}@{host}:{port}/{db}"
    )


def get_neon_engine():
    url = os.environ["NEON_DATABASE_URL"]
    return create_engine(url)


def migrate_table(
    local_engine, neon_engine, table, limit=None
):
    """Copy a table from local to Neon."""
    query = f"SELECT * FROM {table}"
    if limit:
        query += f" LIMIT {limit}"

    logger.info(f"Reading {table} from local...")
    df = pd.read_sql(text(query), local_engine)
    logger.info(f"  {len(df):,} rows")

    if df.empty:
        logger.warning(f"  {table} is empty, skipping")
        return

    # Drop id column if auto-generated
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    logger.info(f"Writing {table} to Neon...")
    total = 0
    for i in range(0, len(df), BATCH_SIZE):
        chunk = df.iloc[i : i + BATCH_SIZE]
        chunk.to_sql(
            table,
            neon_engine,
            if_exists="append" if i > 0 else "replace",
            index=False,
        )
        total += len(chunk)
        logger.info(
            f"  {total:,} / {len(df):,} rows written"
        )

    # Verify
    with neon_engine.connect() as conn:
        count = conn.execute(
            text(f"SELECT COUNT(*) FROM {table}")
        ).scalar()
    logger.info(f"  Verified: {count:,} rows in Neon")


def main():
    local = get_local_engine()
    neon = get_neon_engine()

    # Test Neon connection
    with neon.connect() as conn:
        v = conn.execute(text("SELECT version()")).scalar()
        logger.info(f"Neon connected: {v[:50]}")

    # Migrate tables in order
    # Reviews: limit to 50K for free tier storage
    migrate_table(local, neon, "reviews", limit=50000)
    # Products: limit to top products
    migrate_table(local, neon, "products", limit=50000)
    # Alerts: all
    migrate_table(local, neon, "alerts")
    # Product features: limit for storage
    migrate_table(
        local, neon, "product_features", limit=50000
    )

    logger.info("Migration complete!")


if __name__ == "__main__":
    main()
