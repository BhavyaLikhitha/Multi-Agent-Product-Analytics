"""Create root cause labels for negative reviews using Groq LLM.

Saves progress incrementally so you can resume after interruptions
or API rate limits. Just rerun the script to continue.
"""

import json
import os
import time

import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from loguru import logger
from sqlalchemy import create_engine, text

load_dotenv()

LABEL_CATEGORIES = [
    "defect",
    "shipping",
    "description",
    "size",
    "price",
]
SAMPLE_SIZE = 3000
OUTPUT_PATH = "data/processed/labeled_reviews.csv"
SAMPLES_PATH = "data/processed/sampled_reviews.csv"

SYSTEM_PROMPT = (
    "You are a product review classifier. "
    "Given a negative product review, classify the root cause "
    "into one or more categories. "
    "Respond ONLY with a JSON object.\n\n"
    "Categories:\n"
    "- defect: Product is broken, malfunctioning, "
    "stops working, hardware/software issues\n"
    "- shipping: Arrived damaged, late delivery, "
    "wrong item sent, missing parts, packaging\n"
    "- description: Product doesn't match listing, "
    "misleading features, false advertising\n"
    "- size: Wrong size, too big, too small, "
    "doesn't fit, size not as expected\n"
    "- price: Overpriced, not worth the money, "
    "cheaper alternatives, price vs quality\n\n"
    "Rules:\n"
    "- A review can have MULTIPLE categories "
    "(e.g., both defect and shipping)\n"
    "- Set each category to 1 (applies) or 0 (does not)\n"
    "- If none clearly apply, set all to 0\n\n"
    "Respond with ONLY this JSON format, no other text:\n"
    '{"defect": 0, "shipping": 0, "description": 0, '
    '"size": 0, "price": 0}'
)


def get_postgres_engine():
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db = os.environ.get("POSTGRES_DB", "product_intelligence")
    user = os.environ.get("POSTGRES_USER", "postgres")
    pw = os.environ.get("POSTGRES_PASSWORD", "postgres")
    return create_engine(f"postgresql://{user}:{pw}@{host}:{port}/{db}")


def sample_negative_reviews(engine, n: int = SAMPLE_SIZE) -> pd.DataFrame:
    """Sample negative reviews (1-2 stars) with text."""
    query = text(
        f"""
        SELECT id, asin, rating, title, text
        FROM reviews
        WHERE rating <= 2
        AND text IS NOT NULL
        AND LENGTH(text) > 50
        ORDER BY RANDOM()
        LIMIT {n}
        """
    )
    df = pd.read_sql(query, engine)
    logger.info(f"Sampled {len(df):,} negative reviews for labeling")
    return df


def label_review(client: Groq, review_text: str, max_retries: int = 3) -> dict:
    """Label a single review using Groq LLM."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": review_text[:1000],
                    },
                ],
                temperature=0.0,
                max_tokens=100,
            )
            content = response.choices[0].message.content.strip()
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                labels = json.loads(content[start:end])
                result = {}
                for cat in LABEL_CATEGORIES:
                    val = labels.get(cat, 0)
                    result[cat] = 1 if val == 1 else 0
                return result
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2**attempt)
            else:
                logger.warning(f"Failed to label review: {e}")

    return {cat: 0 for cat in LABEL_CATEGORIES}


def label_all_reviews(df: pd.DataFrame, start_from: int = 0) -> pd.DataFrame:
    """Label reviews with incremental saving."""
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    total = len(df)

    # Load existing progress if any
    if start_from > 0 and os.path.exists(OUTPUT_PATH):
        existing = pd.read_csv(OUTPUT_PATH)
        logger.info(f"Resuming from {len(existing):,} / {total:,}")
    else:
        existing = pd.DataFrame()

    rows = []
    for idx in range(start_from, total):
        row = df.iloc[idx]
        result = label_review(client, row["text"])

        row_data = row.to_dict()
        row_data.update(result)
        rows.append(row_data)

        done = start_from + len(rows)
        if done % 50 == 0:
            # Save progress every 50 reviews
            new_df = pd.DataFrame(rows)
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined.to_csv(OUTPUT_PATH, index=False)
            logger.info(f"Labeled {done:,} / {total:,} " f"(saved to disk)")
            existing = combined
            rows = []

        if done % 100 == 0:
            logger.info(f"Progress: {done:,} / {total:,} reviews")

        # Rate limiting: Groq free tier = 30 req/min
        if len(rows) > 0 and (start_from + len(rows)) % 28 == 0:
            time.sleep(62)

    # Save any remaining
    if rows:
        new_df = pd.DataFrame(rows)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined.to_csv(OUTPUT_PATH, index=False)

    final = pd.read_csv(OUTPUT_PATH)
    return final


if __name__ == "__main__":
    engine = get_postgres_engine()
    os.makedirs("data/processed", exist_ok=True)

    # Check if we have a completed run
    if os.path.exists(OUTPUT_PATH):
        existing = pd.read_csv(OUTPUT_PATH)
        if len(existing) >= SAMPLE_SIZE:
            logger.info(f"Labels complete: {len(existing):,} rows")
            for cat in LABEL_CATEGORIES:
                count = existing[cat].sum()
                pct = count / len(existing) * 100
                logger.info(f"  {cat}: {count:,} ({pct:.1f}%)")
            exit(0)
        else:
            logger.info(f"Resuming: {len(existing):,} / " f"{SAMPLE_SIZE:,} done")
            # Load the same sample set
            if os.path.exists(SAMPLES_PATH):
                df = pd.read_csv(SAMPLES_PATH)
            else:
                logger.error("Cannot resume — sample file missing")
                exit(1)
            start_from = len(existing)
    else:
        # Fresh start — sample and save the sample set
        df = sample_negative_reviews(engine)
        df.to_csv(SAMPLES_PATH, index=False)
        start_from = 0

    labeled_df = label_all_reviews(df, start_from)

    logger.info(f"Saved {len(labeled_df):,} labeled reviews " f"to {OUTPUT_PATH}")
    for cat in LABEL_CATEGORIES:
        count = labeled_df[cat].sum()
        pct = count / len(labeled_df) * 100
        logger.info(f"  {cat}: {count:,} ({pct:.1f}%)")
