"""Generate review embeddings and store in ChromaDB.

Uses sentence-transformers (all-MiniLM-L6-v2, 384-dim)
to embed negative reviews and store them in ChromaDB
for semantic search.
"""

import os

import chromadb
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text

load_dotenv()

MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "review_embeddings"
BATCH_SIZE = 500
MAX_REVIEWS = 50_000  # embed negative + mixed reviews


def get_postgres_engine():
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db = os.environ.get("POSTGRES_DB", "product_intelligence")
    user = os.environ.get("POSTGRES_USER", "postgres")
    pw = os.environ.get("POSTGRES_PASSWORD", "postgres")
    return create_engine(f"postgresql://{user}:{pw}@{host}:{port}/{db}")


def get_chroma_client():
    host = os.environ.get("CHROMA_HOST", "localhost")
    port = int(os.environ.get("CHROMA_PORT", "8000"))
    return chromadb.HttpClient(host=host, port=port)


def load_reviews(engine, limit: int = MAX_REVIEWS):
    """Load reviews for embedding. Prioritize negative."""
    query = text(
        f"""
        SELECT id, asin, rating, title, text
        FROM reviews
        WHERE rating <= 3
        AND text IS NOT NULL
        AND LENGTH(text) > 30
        ORDER BY rating ASC, id ASC
        LIMIT {limit}
        """
    )
    df = pd.read_sql(query, engine)
    logger.info(f"Loaded {len(df):,} reviews for embedding")
    return df


def generate_and_store(
    df: pd.DataFrame,
    model: SentenceTransformer,
    collection,
):
    """Embed reviews in batches and store in ChromaDB."""
    total = len(df)

    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        batch = df.iloc[start:end]

        texts = batch["text"].fillna("").tolist()
        ids = [str(x) for x in batch["id"].tolist()]

        # Check which IDs already exist
        try:
            existing = collection.get(ids=ids)
            existing_ids = set(existing["ids"])
        except Exception:
            existing_ids = set()

        # Filter out already-embedded reviews
        new_mask = [i for i, rid in enumerate(ids) if rid not in existing_ids]
        if not new_mask:
            logger.info(f"Batch {start}-{end}: all already embedded")
            continue

        new_texts = [texts[i] for i in new_mask]
        new_ids = [ids[i] for i in new_mask]
        new_meta = [
            {
                "asin": str(batch.iloc[i]["asin"]),
                "rating": float(batch.iloc[i]["rating"]),
                "title": str(batch.iloc[i]["title"] or "")[:200],
            }
            for i in new_mask
        ]

        embeddings = model.encode(new_texts, show_progress_bar=False).tolist()

        collection.add(
            ids=new_ids,
            embeddings=embeddings,
            documents=new_texts,
            metadatas=new_meta,
        )

        logger.info(f"Embedded {end:,} / {total:,} " f"({len(new_ids)} new)")


if __name__ == "__main__":
    engine = get_postgres_engine()
    chroma = get_chroma_client()

    # Create or get collection
    collection = chroma.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    existing_count = collection.count()
    if existing_count > 0:
        logger.info(f"Collection already has {existing_count:,} " f"embeddings")

    # Load model
    logger.info(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    # Load reviews
    df = load_reviews(engine)

    # Generate and store
    generate_and_store(df, model, collection)

    final_count = collection.count()
    logger.info(f"Done! {final_count:,} embeddings in ChromaDB")
