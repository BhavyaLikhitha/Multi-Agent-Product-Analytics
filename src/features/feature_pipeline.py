"""Feature engineering: computes rolling product-level features."""

import json
import os

import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from sqlalchemy import (
    Column,
    Date,
    Float,
    Index,
    Integer,
    String,
    Text,
    create_engine,
    text,
)
from sqlalchemy.orm import DeclarativeBase

from src.features.ner_extractor import extract_batch, load_nlp

load_dotenv()

NEGATIVE_KEYWORDS = [
    "broken",
    "defective",
    "terrible",
    "worst",
    "awful",
    "horrible",
    "waste",
    "garbage",
    "trash",
    "scam",
    "fake",
    "cheap",
    "flimsy",
    "useless",
    "disappointing",
    "refund",
    "return",
]


class Base(DeclarativeBase):
    pass


class ProductFeature(Base):
    __tablename__ = "product_features"

    id = Column(Integer, primary_key=True, autoincrement=True)
    asin = Column(String(20), index=True)
    date = Column(Date, index=True)
    daily_sentiment_avg = Column(Float)
    review_velocity = Column(Float)
    negative_ratio = Column(Float)
    complaint_keywords = Column(Integer)
    ner_entities = Column(Text)

    __table_args__ = (Index("ix_features_asin_date", "asin", "date"),)


def get_postgres_engine():
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db = os.environ.get("POSTGRES_DB", "product_intelligence")
    user = os.environ.get("POSTGRES_USER", "postgres")
    password = os.environ.get("POSTGRES_PASSWORD", "postgres")
    return create_engine(f"postgresql://{user}:{password}@{host}:{port}/{db}")


def count_negative_keywords(text: str) -> int:
    """Count occurrences of negative keywords in review text."""
    if not isinstance(text, str):
        return 0
    text_lower = text.lower()
    return sum(1 for kw in NEGATIVE_KEYWORDS if kw in text_lower)


def compute_features(engine) -> pd.DataFrame:
    """Compute daily product-level features from reviews."""
    logger.info("Loading reviews from PostgreSQL...")
    df = pd.read_sql(
        "SELECT asin, rating, title, text, helpful_votes, timestamp FROM reviews",
        engine,
    )

    if df.empty:
        logger.warning("No reviews found in database")
        return pd.DataFrame()

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date

    logger.info("Computing complaint keyword counts...")
    df["neg_kw_count"] = df["text"].apply(count_negative_keywords)

    logger.info("Running NER extraction on review texts...")
    nlp = load_nlp()
    texts = df["text"].fillna("").tolist()
    ner_results = extract_batch(nlp, texts, batch_size=500)
    df["ner"] = ner_results

    logger.info("Aggregating daily features per product...")
    daily = df.groupby(["asin", "date"]).agg(
        daily_sentiment_avg=("rating", "mean"),
        review_count=("rating", "count"),
        negative_count=("rating", lambda x: (x <= 2).sum()),
        complaint_keywords=("neg_kw_count", "sum"),
        ner_entities=("ner", lambda x: json.dumps(_merge_ner(x.tolist()))),
    )
    daily = daily.reset_index()

    # Compute negative ratio
    daily["negative_ratio"] = daily["negative_count"] / daily["review_count"]

    # Rolling 7-day review velocity (reviews per day)
    daily = daily.sort_values(["asin", "date"])
    daily["review_velocity"] = daily.groupby("asin")["review_count"].transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    )

    # Rolling 7-day sentiment average
    daily["daily_sentiment_avg"] = daily.groupby("asin")[
        "daily_sentiment_avg"
    ].transform(lambda x: x.rolling(7, min_periods=1).mean())

    keep_cols = [
        "asin",
        "date",
        "daily_sentiment_avg",
        "review_velocity",
        "negative_ratio",
        "complaint_keywords",
        "ner_entities",
    ]
    return daily[keep_cols]


def _merge_ner(ner_list: list[dict]) -> dict:
    """Merge NER results from multiple reviews into one summary."""
    components = {}
    issues = {}
    for ner in ner_list:
        for c in ner.get("components", []):
            components[c] = components.get(c, 0) + 1
        for i in ner.get("issues", []):
            issues[i] = issues.get(i, 0) + 1
    return {"components": components, "issues": issues}


def store_features(engine, df: pd.DataFrame):
    """Write computed features to PostgreSQL."""
    Base.metadata.create_all(engine)

    logger.info(f"Storing {len(df):,} feature rows...")
    df.to_sql(
        "product_features", engine, if_exists="replace", index=False, chunksize=5000
    )

    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM product_features")).scalar()
    logger.info(f"Stored {count:,} feature rows in PostgreSQL")


if __name__ == "__main__":
    engine = get_postgres_engine()
    features_df = compute_features(engine)
    if not features_df.empty:
        store_features(engine, features_df)
    logger.info("Feature pipeline complete!")
