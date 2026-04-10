"""Semantic search over review embeddings in Pinecone."""

import os

from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()

MODEL_NAME = "all-MiniLM-L6-v2"

_model = None
_index = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _get_index():
    global _index
    if _index is None:
        pc = Pinecone(
            api_key=os.environ["PINECONE_API_KEY"]
        )
        index_name = os.environ.get(
            "PINECONE_INDEX", "review-embeddings"
        )
        _index = pc.Index(index_name)
    return _index


def search_reviews(
    query: str,
    n_results: int = 10,
    min_rating: float = None,
    max_rating: float = None,
) -> list[dict]:
    """Search reviews by semantic similarity."""
    model = _get_model()
    index = _get_index()

    query_embedding = model.encode(query).tolist()

    # Build filter
    pc_filter = {}
    if min_rating is not None and max_rating is not None:
        pc_filter = {
            "$and": [
                {"rating": {"$gte": min_rating}},
                {"rating": {"$lte": max_rating}},
            ]
        }
    elif min_rating is not None:
        pc_filter = {"rating": {"$gte": min_rating}}
    elif max_rating is not None:
        pc_filter = {"rating": {"$lte": max_rating}}

    results = index.query(
        vector=query_embedding,
        top_k=n_results,
        filter=pc_filter if pc_filter else None,
        include_metadata=True,
    )

    reviews = []
    for match in results.get("matches", []):
        meta = match.get("metadata", {})
        reviews.append(
            {
                "id": match["id"],
                "text": meta.get("text", ""),
                "asin": meta.get("asin", ""),
                "rating": meta.get("rating", 0),
                "title": meta.get("title", ""),
                "distance": 1 - match["score"],
            }
        )

    return reviews


if __name__ == "__main__":
    results = search_reviews(
        "bluetooth keeps disconnecting", n_results=5
    )
    for r in results:
        print(
            f"[{r['rating']}] {r['text'][:100]}... "
            f"(sim={1 - r['distance']:.3f})"
        )
