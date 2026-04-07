"""Listing Auditor agent -- compares listing vs complaints.

Finds mismatches between product listing claims and
actual customer complaints from the Analyzer output.
"""

import os

from dotenv import load_dotenv
from loguru import logger
from sqlalchemy import create_engine, text

load_dotenv()


def _get_engine():
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db = os.environ.get("POSTGRES_DB", "product_intelligence")
    user = os.environ.get("POSTGRES_USER", "postgres")
    pw = os.environ.get("POSTGRES_PASSWORD", "postgres")
    return create_engine(f"postgresql://{user}:{pw}@{host}:{port}/{db}")


def _fetch_product(engine, asin: str) -> dict:
    query = text(
        "SELECT title, description, features, price "
        "FROM products WHERE parent_asin = :asin "
        "LIMIT 1"
    )
    with engine.connect() as conn:
        row = conn.execute(query, {"asin": asin}).mappings().first()
    if row:
        return dict(row)
    # Try direct asin match
    query2 = text(
        "SELECT title, description, features, price "
        "FROM products WHERE parent_asin IN "
        "(SELECT parent_asin FROM reviews "
        "WHERE asin = :asin LIMIT 1) "
        "LIMIT 1"
    )
    with engine.connect() as conn:
        row = conn.execute(query2, {"asin": asin}).mappings().first()
    return dict(row) if row else {}


def _find_mismatches(
    listing_text: str,
    top_components: list,
    top_issues: list,
) -> list[dict]:
    """Find components/issues in listing text."""
    mismatches = []
    listing_lower = listing_text.lower()

    for comp, count in top_components:
        if comp.lower() in listing_lower and count >= 5:
            # Component mentioned in listing AND
            # frequently complained about
            severity = "high" if count >= 20 else "medium"
            mismatches.append(
                {
                    "claim": (f"Listing mentions '{comp}'"),
                    "complaint": (
                        f"{count} reviews mention " f"'{comp}' in complaints"
                    ),
                    "severity": severity,
                }
            )

    for issue, count in top_issues:
        if count >= 10:
            severity = "high" if count >= 30 else ("medium" if count >= 15 else "low")
            mismatches.append(
                {
                    "claim": "Listing does not address",
                    "complaint": (f"'{issue}' mentioned in " f"{count} reviews"),
                    "severity": severity,
                }
            )

    return mismatches


def audit_listing(state: dict) -> dict:
    asin = state["asin"]
    logger.info("Auditing listing for {}", asin)

    engine = _get_engine()
    product = _fetch_product(engine, asin)

    title = product.get("title", "") or ""
    description = product.get("description", "") or ""
    features = product.get("features", "") or ""
    price = product.get("price", "") or ""

    listing_text = f"{title} {description} {features}"

    top_components = state.get("top_components", [])
    top_issues = state.get("top_issues", [])

    mismatches = _find_mismatches(listing_text, top_components, top_issues)

    # Build audit summary
    if mismatches:
        high = sum(1 for m in mismatches if m["severity"] == "high")
        med = sum(1 for m in mismatches if m["severity"] == "medium")
        low = sum(1 for m in mismatches if m["severity"] == "low")
        audit_summary = (
            f"Found {len(mismatches)} mismatches: "
            f"{high} high, {med} medium, {low} low."
        )
    else:
        audit_summary = "No significant mismatches found."

    state.update(
        {
            "product_title": title,
            "product_description": description,
            "product_features": features,
            "product_price": price,
            "mismatches": mismatches,
            "audit_summary": audit_summary,
        }
    )

    logger.info(
        "Audit complete: {} mismatches",
        len(mismatches),
    )
    return state


if __name__ == "__main__":
    state = {
        "asin": "B01G8JO5F2",
        "top_components": [
            ("battery", 50),
            ("bluetooth", 30),
        ],
        "top_issues": [("disconnecting", 20)],
        "complaint_profile": {},
    }
    result = audit_listing(state)
    print(result["audit_summary"])
    for m in result["mismatches"]:
        print(f"  [{m['severity']}] {m['complaint']}")
