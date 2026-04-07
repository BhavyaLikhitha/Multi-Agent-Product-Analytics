"""FastAPI backend for Product Intelligence platform.

Endpoints:
    GET  /health          - Health check
    GET  /alerts          - List quality alerts
    GET  /alerts/{asin}   - Alerts for a specific product
    POST /classify        - Classify review root cause
    GET  /search          - Semantic search over reviews
    POST /analyze/{asin}  - Run full agent pipeline
    GET  /products/top    - Top products by review count
"""

import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel
from sqlalchemy import create_engine, text

load_dotenv()

app = FastAPI(
    title="Product Intelligence API",
    description="Multi-agent product analytics platform",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_engine():
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db = os.environ.get("POSTGRES_DB", "product_intelligence")
    user = os.environ.get("POSTGRES_USER", "postgres")
    pw = os.environ.get("POSTGRES_PASSWORD", "postgres")
    return create_engine(f"postgresql://{user}:{pw}@{host}:{port}/{db}")


engine = _get_engine()


# --- Request/Response Models ---


class ClassifyRequest(BaseModel):
    text: str


class ClassifyResponse(BaseModel):
    defect: float
    shipping: float
    description: float
    size: float
    price: float


class AlertResponse(BaseModel):
    product_id: str
    alert_type: str
    severity: str
    detected_at: str
    details: str


# --- Endpoints ---


@app.get("/health")
def health():
    return {"status": "ok", "service": "product-intelligence"}


@app.get("/alerts")
def get_alerts(
    limit: int = Query(default=50, le=200),
    severity: str = Query(default=None),
):
    """List quality alerts from anomaly detector."""
    query = "SELECT * FROM alerts"
    params = {}

    if severity:
        query += " WHERE severity = :severity"
        params["severity"] = severity

    query += " ORDER BY detected_at DESC LIMIT :limit"
    params["limit"] = limit

    with engine.connect() as conn:
        rows = conn.execute(text(query), params).mappings().all()

    return [dict(r) for r in rows]


@app.get("/alerts/{asin}")
def get_alerts_for_product(asin: str):
    """Get alerts for a specific product."""
    query = text(
        "SELECT * FROM alerts " "WHERE product_id = :asin " "ORDER BY detected_at DESC"
    )
    with engine.connect() as conn:
        rows = conn.execute(query, {"asin": asin}).mappings().all()

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No alerts for {asin}",
        )
    return [dict(r) for r in rows]


@app.post("/classify", response_model=ClassifyResponse)
def classify_review(req: ClassifyRequest):
    """Classify a review into root cause categories."""
    try:
        # Simple rule-based scoring
        scores = {
            "defect": 0.0,
            "shipping": 0.0,
            "description": 0.0,
            "size": 0.0,
            "price": 0.0,
        }

        defect_words = {
            "broken",
            "defective",
            "malfunction",
            "dead",
            "stopped working",
            "not working",
            "crashed",
            "overheating",
            "freezing",
        }
        shipping_words = {
            "arrived damaged",
            "missing parts",
            "wrong item",
            "late delivery",
            "never arrived",
        }
        desc_words = {
            "misleading",
            "not as described",
            "false advertising",
            "cheaply made",
            "poor quality",
        }
        size_words = {
            "too small",
            "too big",
            "doesn't fit",
            "wrong size",
        }
        price_words = {
            "waste of money",
            "overpriced",
            "rip off",
            "scam",
        }

        text_lower = req.text.lower()

        for w in defect_words:
            if w in text_lower:
                scores["defect"] += 0.3
        for w in shipping_words:
            if w in text_lower:
                scores["shipping"] += 0.3
        for w in desc_words:
            if w in text_lower:
                scores["description"] += 0.3
        for w in size_words:
            if w in text_lower:
                scores["size"] += 0.3
        for w in price_words:
            if w in text_lower:
                scores["price"] += 0.3

        # Cap at 1.0
        for k in scores:
            scores[k] = min(1.0, scores[k])

        return ClassifyResponse(**scores)

    except Exception as e:
        logger.error(f"Classification failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search")
def search_reviews(
    q: str = Query(..., min_length=3),
    n: int = Query(default=10, le=50),
    max_rating: float = Query(default=None),
):
    """Semantic search over review embeddings."""
    try:
        from src.api.semantic_search import search_reviews as _search

        results = _search(
            query=q,
            n_results=n,
            max_rating=max_rating,
        )
        return results
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/{asin}")
def analyze_product(asin: str):
    """Run the full LangGraph agent pipeline."""
    try:
        from src.agents.graph import run_pipeline

        result = run_pipeline(asin)
        return {
            "asin": asin,
            "review_count": result.get("review_count"),
            "avg_rating": result.get("avg_rating"),
            "negative_pct": result.get("negative_pct"),
            "mismatches": result.get("mismatches", []),
            "audit_summary": result.get("audit_summary"),
            "rewritten_title": result.get("rewritten_title"),
            "rewritten_description": result.get("rewritten_description"),
            "changes_made": result.get("changes_made"),
            "final_status": result.get("final_status"),
            "supervisor_notes": result.get("supervisor_notes"),
        }
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/products/top")
def top_products(limit: int = Query(default=20, le=100)):
    """Top products by review count."""
    query = text(
        """
        SELECT r.asin, p.title,
               COUNT(*) as review_count,
               ROUND(AVG(r.rating)::numeric, 2)
                   as avg_rating
        FROM reviews r
        LEFT JOIN products p
            ON r.parent_asin = p.parent_asin
        GROUP BY r.asin, p.title
        ORDER BY review_count DESC
        LIMIT :limit
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(query, {"limit": limit}).mappings().all()

    return [dict(r) for r in rows]
