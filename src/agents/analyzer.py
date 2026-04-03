"""Review Analyzer agent -- first node in the LangGraph pipeline.

Pulls reviews for a product ASIN from PostgreSQL, runs fast
regex NER to extract components and issues, and builds a
structured complaint profile for downstream agents.
"""

import os
from collections import Counter

from dotenv import load_dotenv
from loguru import logger
from sqlalchemy import create_engine, text

from src.features.ner_extractor import extract_batch_fast

load_dotenv()


def _get_engine():
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


def _fetch_reviews(engine, asin: str) -> list[dict]:
    query = text(
        "SELECT rating, title, text "
        "FROM reviews WHERE asin = :asin "
        "ORDER BY timestamp DESC"
    )
    with engine.connect() as conn:
        rows = (
            conn.execute(query, {"asin": asin})
            .mappings()
            .all()
        )
    reviews = [dict(r) for r in rows]
    logger.info(
        "Fetched {} reviews for ASIN {}",
        len(reviews),
        asin,
    )
    return reviews


def analyze_product(state: dict) -> dict:
    asin = state["asin"]
    logger.info("Analyzing product {}", asin)

    engine = _get_engine()
    reviews = _fetch_reviews(engine, asin)

    if not reviews:
        state.update(
            {
                "reviews": [],
                "review_count": 0,
                "avg_rating": 0.0,
                "negative_pct": 0.0,
                "top_components": [],
                "top_issues": [],
                "complaint_profile": {
                    "summary": "No reviews found."
                },
            }
        )
        return state

    ratings = [r["rating"] for r in reviews]
    review_count = len(reviews)
    avg_rating = sum(ratings) / review_count
    neg_count = sum(1 for r in ratings if r <= 2.0)
    negative_pct = (neg_count / review_count) * 100

    texts = [
        f"{r.get('title', '')} {r.get('text', '')}"
        for r in reviews
    ]
    ner_results = extract_batch_fast(texts)

    comp_counter: Counter = Counter()
    issue_counter: Counter = Counter()
    for ner in ner_results:
        for comp in ner["components"]:
            comp_counter[comp] += 1
        for iss in ner["issues"]:
            issue_counter[iss] += 1

    top_components = comp_counter.most_common(10)
    top_issues = issue_counter.most_common(10)

    comp_lines = [
        f"  - {c} ({n} mentions)"
        for c, n in top_components[:5]
    ]
    issue_lines = [
        f"  - {i} ({n} mentions)"
        for i, n in top_issues[:5]
    ]

    summary = (
        f"Product has {review_count} reviews "
        f"(avg {avg_rating:.2f} stars, "
        f"{negative_pct:.1f}% negative).\n"
    )
    if comp_lines:
        summary += "Top components:\n"
        summary += "\n".join(comp_lines) + "\n"
    if issue_lines:
        summary += "Top issues:\n"
        summary += "\n".join(issue_lines) + "\n"

    state.update(
        {
            "reviews": reviews,
            "review_count": review_count,
            "avg_rating": round(avg_rating, 2),
            "negative_pct": round(negative_pct, 1),
            "top_components": top_components,
            "top_issues": top_issues,
            "complaint_profile": {"summary": summary},
        }
    )
    return state


if __name__ == "__main__":
    result = analyze_product({"asin": "B01G8JO5F2"})
    print(result["complaint_profile"])
