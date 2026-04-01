"""Create root cause labels for negative reviews using Groq LLM."""

import json
import os
import time

import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from loguru import logger
from sqlalchemy import create_engine, text

load_dotenv()

LABEL_CATEGORIES = ["defect", "shipping", "description", "size", "price"]
SAMPLE_SIZE = 3000
OUTPUT_PATH = "data/processed/labeled_reviews.csv"

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
    password = os.environ.get("POSTGRES_PASSWORD", "postgres")
    return create_engine(f"postgresql://{user}:{password}@{host}:{port}/{db}")


def sample_negative_reviews(engine, n: int = SAMPLE_SIZE) -> pd.DataFrame:
    """Sample negative reviews (1-2 stars) with sufficient text."""
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
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": review_text[:1000]},
                ],
                temperature=0.0,
                max_tokens=100,
            )
            content = response.choices[0].message.content.strip()
            # Extract JSON from response
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                labels = json.loads(content[start:end])
                # Validate all keys present and values are 0 or 1
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


def label_all_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Label all reviews using Groq with rate limiting."""
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    labels = []
    total = len(df)

    for i, row in df.iterrows():
        result = label_review(client, row["text"])
        labels.append(result)

        done = len(labels)
        if done % 100 == 0:
            logger.info(f"Labeled {done:,} / {total:,} reviews")

        # Rate limiting: Groq free tier = 30 req/min
        if done % 28 == 0:
            time.sleep(62)

    labels_df = pd.DataFrame(labels)
    return pd.concat([df.reset_index(drop=True), labels_df], axis=1)


if __name__ == "__main__":
    engine = get_postgres_engine()

    # Check if labels already exist
    if os.path.exists(OUTPUT_PATH):
        existing = pd.read_csv(OUTPUT_PATH)
        logger.info(f"Labels already exist: {len(existing):,} rows at {OUTPUT_PATH}")
        logger.info("Delete the file to re-generate labels.")
    else:
        os.makedirs("data/processed", exist_ok=True)

        df = sample_negative_reviews(engine)
        labeled_df = label_all_reviews(df)
        labeled_df.to_csv(OUTPUT_PATH, index=False)

        # Print summary
        logger.info(f"Saved {len(labeled_df):,} labeled reviews to {OUTPUT_PATH}")
        for cat in LABEL_CATEGORIES:
            count = labeled_df[cat].sum()
            pct = count / len(labeled_df) * 100
            logger.info(f"  {cat}: {count:,} ({pct:.1f}%)")
