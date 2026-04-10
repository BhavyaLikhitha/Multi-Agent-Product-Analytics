"""Semantic search over review embeddings in Pinecone.

Uses sentence-transformers locally if available,
falls back to Pinecone inference API for cloud.
"""

import os

from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

_pc = None
_index = None
_model = None


def _get_secret(key, default=None):
    try:
        import streamlit as st

        return st.secrets[key]
    except Exception:
        return os.environ.get(key, default)


def _get_pc():
    global _pc
    if _pc is None:
        _pc = Pinecone(
            api_key=_get_secret("PINECONE_API_KEY")
        )
    return _pc


def _get_index():
    global _index
    if _index is None:
        pc = _get_pc()
        index_name = _get_secret(
            "PINECONE_INDEX", "review-embeddings"
        )
        _index = pc.Index(index_name)
    return _index


def _encode_query(query: str) -> list[float]:
    """Encode query: use local model if available, else HuggingFace Inference API."""
    global _model
    try:
        if _model is None:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer("all-MiniLM-L6-v2")
        return _model.encode(query).tolist()
    except ImportError:
        import requests

        resp = requests.post(
            "https://api-inference.huggingface.co/pipeline/feature-extraction/"
            "sentence-transformers/all-MiniLM-L6-v2",
            json={"inputs": query, "options": {"wait_for_model": True}},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()


def search_reviews(
    query: str,
    n_results: int = 10,
    min_rating: float = None,
    max_rating: float = None,
) -> list[dict]:
    """Search reviews by semantic similarity."""
    index = _get_index()

    query_embedding = _encode_query(query)

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
