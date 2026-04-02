"""Semantic search over review embeddings in ChromaDB."""

import os

import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "review_embeddings"

_model = None
_collection = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _get_collection():
    global _collection
    if _collection is None:
        host = os.environ.get("CHROMA_HOST", "localhost")
        port = int(os.environ.get("CHROMA_PORT", "8000"))
        client = chromadb.HttpClient(host=host, port=port)
        _collection = client.get_collection(name=COLLECTION_NAME)
    return _collection


def search_reviews(
    query: str,
    n_results: int = 10,
    min_rating: float = None,
    max_rating: float = None,
) -> list[dict]:
    """Search reviews by semantic similarity.

    Args:
        query: Natural language search query
        n_results: Number of results to return
        min_rating: Filter by minimum rating
        max_rating: Filter by maximum rating

    Returns:
        List of dicts with keys: id, text, asin,
        rating, title, distance
    """
    model = _get_model()
    collection = _get_collection()

    query_embedding = model.encode(query).tolist()

    # Build where filter
    where = None
    if min_rating is not None and max_rating is not None:
        where = {
            "$and": [
                {"rating": {"$gte": min_rating}},
                {"rating": {"$lte": max_rating}},
            ]
        }
    elif min_rating is not None:
        where = {"rating": {"$gte": min_rating}}
    elif max_rating is not None:
        where = {"rating": {"$lte": max_rating}}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    reviews = []
    for i in range(len(results["ids"][0])):
        reviews.append(
            {
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "asin": results["metadatas"][0][i]["asin"],
                "rating": results["metadatas"][0][i]["rating"],
                "title": results["metadatas"][0][i]["title"],
                "distance": results["distances"][0][i],
            }
        )

    return reviews


if __name__ == "__main__":
    # Quick test
    results = search_reviews("bluetooth keeps disconnecting", n_results=5)
    for r in results:
        print(f"[{r['rating']}] {r['text'][:100]}... " f"(dist={r['distance']:.3f})")
