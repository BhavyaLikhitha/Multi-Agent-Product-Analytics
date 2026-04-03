"""Create training pairs for Mistral-7B fine-tuning.

Uses Gemini Flash to generate high-quality structured
summaries for products with 20+ reviews.
Output: JSONL file with (input_reviews, summary) pairs.

Saves incrementally every 10 products.
"""

import json
import os
import time

import anthropic
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from sqlalchemy import create_engine, text

load_dotenv()

OUTPUT_PATH = "data/processed/summary_training_pairs.jsonl"
PROGRESS_PATH = "data/processed/summary_progress.json"
MIN_REVIEWS = 20
TARGET_PRODUCTS = 400

SUMMARY_PROMPT = (
    "You are an expert product analyst. Given a set of "
    "customer reviews for a product, write a structured "
    "executive summary.\n\n"
    "Format your summary EXACTLY like this:\n"
    "## Product Issues Summary\n\n"
    "**Overall Sentiment:** [positive/mixed/negative] "
    "([X]% negative reviews)\n\n"
    "**Top Complaints:**\n"
    "1. [Issue] — [X] mentions — [brief description]\n"
    "2. [Issue] — [X] mentions — [brief description]\n"
    "3. [Issue] — [X] mentions — [brief description]\n\n"
    "**Key Insights:**\n"
    "- [Actionable insight 1]\n"
    "- [Actionable insight 2]\n\n"
    "**Recommendation:** [One sentence action item "
    "for the product team]\n\n"
    "Be specific with numbers. Use the actual review "
    "data, don't make up statistics."
)


def get_engine():
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


def get_products_with_reviews(engine):
    """Find products with enough reviews."""
    query = text(
        f"""
        SELECT r.asin,
               p.title as product_title,
               COUNT(*) as review_count
        FROM reviews r
        LEFT JOIN products p
            ON r.parent_asin = p.parent_asin
        WHERE r.text IS NOT NULL
        AND LENGTH(r.text) > 30
        GROUP BY r.asin, p.title
        HAVING COUNT(*) >= {MIN_REVIEWS}
        ORDER BY COUNT(*) DESC
        LIMIT {TARGET_PRODUCTS * 2}
        """
    )
    df = pd.read_sql(query, engine)
    logger.info(
        f"Found {len(df):,} products with "
        f"{MIN_REVIEWS}+ reviews"
    )
    return df


def get_reviews_for_product(engine, asin, limit=50):
    """Get reviews for a specific product."""
    query = text(
        """
        SELECT rating, title, text
        FROM reviews
        WHERE asin = :asin
        AND text IS NOT NULL
        ORDER BY helpful_votes DESC
        LIMIT :limit
        """
    )
    return pd.read_sql(
        query,
        engine,
        params={"asin": asin, "limit": limit},
    )


def format_reviews_input(reviews_df, product_title):
    """Format reviews into input text for the model."""
    lines = [
        f"Product: {product_title or 'Unknown'}\n"
    ]
    lines.append(
        f"Total reviews shown: {len(reviews_df)}\n"
    )

    neg = reviews_df[reviews_df["rating"] <= 2]
    pos = reviews_df[reviews_df["rating"] >= 4]
    lines.append(
        f"Negative (1-2 stars): {len(neg)} | "
        f"Positive (4-5 stars): {len(pos)}\n"
    )
    lines.append("---\n")

    for _, row in reviews_df.iterrows():
        stars = int(row["rating"])
        title = row["title"] or ""
        review_text = str(row["text"])[:300]
        lines.append(
            f"[{stars} stars] {title}\n{review_text}\n"
        )

    return "\n".join(lines)


def generate_summary(client, reviews_input):
    """Generate summary using Claude Sonnet."""
    try:
        resp = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            system=SUMMARY_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": reviews_input[:4000],
                },
            ],
        )
        return resp.content[0].text.strip()
    except Exception as e:
        logger.warning(
            f"Failed to generate summary: {e}"
        )
        return None


def load_progress():
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH) as f:
            return json.load(f)
    return {"done_asins": [], "count": 0}


def save_progress(progress):
    with open(PROGRESS_PATH, "w") as f:
        json.dump(progress, f)


def main():
    engine = get_engine()

    client = anthropic.Anthropic(
        api_key=os.environ["ANTHROPIC_API_KEY"]
    )
    logger.info("Using Claude Sonnet")

    os.makedirs("data/processed", exist_ok=True)

    products = get_products_with_reviews(engine)
    progress = load_progress()
    done_asins = set(progress["done_asins"])

    products = products[
        ~products["asin"].isin(done_asins)
    ]
    products = products.head(
        TARGET_PRODUCTS - len(done_asins)
    )

    if len(products) == 0:
        logger.info(
            f"Already have {len(done_asins)} pairs!"
        )
        return

    logger.info(
        f"Generating summaries for {len(products)} "
        f"products (resuming from {len(done_asins)})"
    )

    pairs_written = 0

    with open(OUTPUT_PATH, "a") as f:
        for idx, row in products.iterrows():
            asin = row["asin"]
            title = row["product_title"]

            reviews = get_reviews_for_product(
                engine, asin
            )
            if len(reviews) < MIN_REVIEWS:
                continue

            reviews_input = format_reviews_input(
                reviews, title
            )
            summary = generate_summary(
                client, reviews_input
            )

            if summary:
                pair = {
                    "asin": asin,
                    "product_title": title,
                    "input": reviews_input,
                    "summary": summary,
                }
                f.write(json.dumps(pair) + "\n")
                f.flush()
                pairs_written += 1

                done_asins.add(asin)
                progress["done_asins"] = list(done_asins)
                progress["count"] = len(done_asins)

                if pairs_written % 10 == 0:
                    save_progress(progress)
                    logger.info(
                        f"Generated {len(done_asins)}"
                        f" / {TARGET_PRODUCTS} "
                        f"summaries (saved)"
                    )

            # Small delay to avoid rate limits
            time.sleep(2)

    save_progress(progress)
    logger.info(
        f"Done! {len(done_asins)} summary pairs "
        f"saved to {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
