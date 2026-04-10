"""Generate review embeddings and store in Pinecone.

Uses sentence-transformers (all-MiniLM-L6-v2, 384-dim)
to embed negative reviews and store them in Pinecone
for semantic search.
"""

import os

import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text

load_dotenv()

MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 100
MAX_REVIEWS = 50_000


def get_postgres_engine():
    neon_url = os.environ.get("NEON_DATABASE_URL")
    if neon_url:
        return create_engine(neon_url)
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db = os.environ.get("POSTGRES_DB", "product_intelligence")
    user = os.environ.get("POSTGRES_USER", "postgres")
    pw = os.environ.get("POSTGRES_PASSWORD", "postgres")
    return create_engine(f"postgresql://{user}:{pw}@{host}:{port}/{db}")


def get_pinecone_index():
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index_name = os.environ.get("PINECONE_INDEX", "review-embeddings")
    return pc.Index(index_name)


def load_reviews(engine, limit=MAX_REVIEWS):
    query = text(
        f"""
        SELECT asin, rating, title, text
        FROM reviews
        WHERE rating <= 3
        AND text IS NOT NULL
        AND LENGTH(text) > 30
        ORDER BY rating ASC
        LIMIT {limit}
        """
    )
    df = pd.read_sql(query, engine)
    logger.info(f"Loaded {len(df):,} reviews for embedding")
    return df


def generate_and_store(df, model, index):
    total = len(df)

    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        batch = df.iloc[start:end]

        texts = batch["text"].fillna("").tolist()
        ids = [f"rev_{start + i}" for i in range(len(batch))]

        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        vectors = []
        for i, (emb, row_idx) in enumerate(zip(embeddings, batch.index)):
            row = batch.loc[row_idx]
            vectors.append(
                {
                    "id": ids[i],
                    "values": emb,
                    "metadata": {
                        "asin": str(row["asin"]),
                        "rating": float(row["rating"]),
                        "title": str(row["title"] or "")[:200],
                        "text": str(row["text"] or "")[:500],
                    },
                }
            )

        index.upsert(vectors=vectors)

        logger.info(f"Embedded {end:,} / {total:,}")


if __name__ == "__main__":
    engine = get_postgres_engine()
    index = get_pinecone_index()

    stats = index.describe_index_stats()
    existing = stats.get("total_vector_count", 0)
    if existing > 0:
        logger.info(f"Index already has {existing:,} vectors")

    logger.info(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    df = load_reviews(engine)
    generate_and_store(df, model, index)

    stats = index.describe_index_stats()
    final = stats.get("total_vector_count", 0)
    logger.info(f"Done! {final:,} vectors in Pinecone")
