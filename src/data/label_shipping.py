"""Label additional shipping-focused negative reviews.

Targets the weakest category (shipping=171/3000) by
finding reviews that mention shipping keywords and
labeling them with Groq.
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
MAIN_LABELS_PATH = "data/processed/labeled_reviews.csv"
EXTRA_PATH = "data/processed/extra_shipping_labels.csv"
TARGET_EXTRA = 500

SYSTEM_PROMPT = (
    "You are a product review classifier. "
    "Given a negative product review, classify the root "
    "cause into one or more categories. "
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
    "- A review can have MULTIPLE categories\n"
    "- Set each category to 1 or 0\n"
    "- If none clearly apply, set all to 0\n\n"
    "Respond with ONLY this JSON:\n"
    '{"defect": 0, "shipping": 0, "description": 0, '
    '"size": 0, "price": 0}'
)


def get_engine():
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db = os.environ.get("POSTGRES_DB", "product_intelligence")
    user = os.environ.get("POSTGRES_USER", "postgres")
    pw = os.environ.get("POSTGRES_PASSWORD", "postgres")
    return create_engine(f"postgresql://{user}:{pw}@{host}:{port}/{db}")


def sample_shipping_reviews(engine):
    """Find reviews mentioning shipping keywords."""
    # Get IDs already labeled
    existing = pd.read_csv(MAIN_LABELS_PATH)
    existing_ids = set(existing["id"].tolist())
    if os.path.exists(EXTRA_PATH):
        extra = pd.read_csv(EXTRA_PATH)
        existing_ids.update(extra["id"].tolist())

    query = text(
        f"""
        SELECT id, asin, rating, title, text
        FROM reviews
        WHERE rating <= 2
        AND text IS NOT NULL
        AND LENGTH(text) > 50
        AND (
            LOWER(text) LIKE '%arrived damaged%'
            OR LOWER(text) LIKE '%arrived broken%'
            OR LOWER(text) LIKE '%wrong item%'
            OR LOWER(text) LIKE '%missing parts%'
            OR LOWER(text) LIKE '%late delivery%'
            OR LOWER(text) LIKE '%shipping%'
            OR LOWER(text) LIKE '%package%'
            OR LOWER(text) LIKE '%packaging%'
            OR LOWER(text) LIKE '%box was%'
            OR LOWER(text) LIKE '%never arrived%'
            OR LOWER(text) LIKE '%never received%'
            OR LOWER(text) LIKE '%wrong product%'
            OR LOWER(text) LIKE '%damaged in transit%'
        )
        ORDER BY RANDOM()
        LIMIT {TARGET_EXTRA * 2}
        """
    )
    df = pd.read_sql(query, engine)
    # Remove already labeled
    df = df[~df["id"].isin(existing_ids)]
    df = df.head(TARGET_EXTRA)
    logger.info(f"Found {len(df):,} shipping-related reviews " f"to label")
    return df


def label_review(client, review_text, max_retries=3):
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": review_text[:1000],
                    },
                ],
                temperature=0.0,
                max_tokens=100,
            )
            content = resp.choices[0].message.content.strip()
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                labels = json.loads(content[start:end])
                return {
                    cat: 1 if labels.get(cat, 0) == 1 else 0 for cat in LABEL_CATEGORIES
                }
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2**attempt)
            else:
                logger.warning(f"Failed: {e}")
    return {cat: 0 for cat in LABEL_CATEGORIES}


def label_all(df):
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    total = len(df)

    # Resume support
    if os.path.exists(EXTRA_PATH):
        existing = pd.read_csv(EXTRA_PATH)
        start = len(existing)
        logger.info(f"Resuming from {start}/{total}")
    else:
        existing = pd.DataFrame()
        start = 0

    rows = []
    for idx in range(start, total):
        row = df.iloc[idx]
        result = label_review(client, row["text"])
        row_data = row.to_dict()
        row_data.update(result)
        rows.append(row_data)

        done = start + len(rows)
        if done % 50 == 0:
            new_df = pd.DataFrame(rows)
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined.to_csv(EXTRA_PATH, index=False)
            existing = combined
            rows = []
            logger.info(f"Labeled {done}/{total} (saved)")

        if len(rows) > 0 and (start + len(rows)) % 28 == 0:
            time.sleep(62)

    if rows:
        new_df = pd.DataFrame(rows)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined.to_csv(EXTRA_PATH, index=False)

    return pd.read_csv(EXTRA_PATH)


def merge_labels():
    """Merge extra labels into main labels file."""
    main = pd.read_csv(MAIN_LABELS_PATH)
    extra = pd.read_csv(EXTRA_PATH)
    combined = pd.concat([main, extra], ignore_index=True)
    combined = combined.drop_duplicates(subset=["id"])
    combined.to_csv(MAIN_LABELS_PATH, index=False)
    logger.info(f"Merged: {len(combined):,} total labels")
    for cat in LABEL_CATEGORIES:
        count = combined[cat].sum()
        pct = count / len(combined) * 100
        logger.info(f"  {cat}: {count:,.0f} ({pct:.1f}%)")


if __name__ == "__main__":
    engine = get_engine()

    if os.path.exists(EXTRA_PATH):
        extra = pd.read_csv(EXTRA_PATH)
        if len(extra) >= TARGET_EXTRA:
            logger.info("Extra labels complete, merging...")
            merge_labels()
            exit(0)

    df = sample_shipping_reviews(engine)
    label_all(df)
    merge_labels()
