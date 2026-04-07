"""LLM-as-Judge evaluation for generated summaries.

Uses Groq to score summaries on 4 criteria:
accuracy, completeness, actionability, conciseness.
Each scored 1-5. Stores results in PostgreSQL.
"""

import json
import os
import time

import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from loguru import logger
from sqlalchemy import (
    Column,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase

load_dotenv()

PAIRS_PATH = "data/processed/summary_training_pairs.jsonl"
FINETUNED_PATH = "data/processed/finetuned_summaries.jsonl"
RESULTS_PATH = "data/processed/evaluation_results.csv"

JUDGE_PROMPT = (
    "You are an expert evaluator of product review summaries. "
    "Score the following summary on 4 criteria, each from 1 to 5.\n\n"
    "Criteria:\n"
    "- accuracy: Does the summary correctly reflect the reviews? "
    "(1=completely wrong, 5=perfectly accurate)\n"
    "- completeness: Does it cover all major complaints? "
    "(1=misses everything, 5=covers all key issues)\n"
    "- actionability: Can a product manager act on this? "
    "(1=vague/useless, 5=clear action items)\n"
    "- conciseness: Is it well-structured and not too long? "
    "(1=rambling/messy, 5=crisp and organized)\n\n"
    "Respond with ONLY a JSON object:\n"
    '{"accuracy": 3, "completeness": 3, '
    '"actionability": 3, "conciseness": 3}'
)

CRITERIA = ["accuracy", "completeness", "actionability", "conciseness"]


class Base(DeclarativeBase):
    pass


class Evaluation(Base):
    __tablename__ = "evaluations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    asin = Column(String(20), index=True)
    model_name = Column(String(50))
    accuracy = Column(Float)
    completeness = Column(Float)
    actionability = Column(Float)
    conciseness = Column(Float)
    overall_score = Column(Float)
    summary = Column(Text)


def get_engine():
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db = os.environ.get("POSTGRES_DB", "product_intelligence")
    user = os.environ.get("POSTGRES_USER", "postgres")
    pw = os.environ.get("POSTGRES_PASSWORD", "postgres")
    return create_engine(f"postgresql://{user}:{pw}@{host}:{port}/{db}")


def load_pairs():
    """Load summary pairs from JSONL."""
    pairs = []
    with open(PAIRS_PATH) as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs


def judge_summary(client, reviews_input, summary, max_retries=3):
    """Score a summary using LLM judge."""
    user_msg = (
        f"REVIEWS:\n{reviews_input[:2000]}\n\n" f"SUMMARY TO EVALUATE:\n{summary}"
    )

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": JUDGE_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=100,
            )
            content = resp.choices[0].message.content.strip()
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                scores = json.loads(content[start:end])
                result = {}
                for c in CRITERIA:
                    val = scores.get(c, 3)
                    result[c] = max(1, min(5, float(val)))
                result["overall_score"] = sum(result[c] for c in CRITERIA) / len(
                    CRITERIA
                )
                return result
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2**attempt)
            else:
                logger.warning(f"Judge failed: {e}")

    return {c: 3.0 for c in CRITERIA + ["overall_score"]}


def load_finetuned_pairs():
    """Load fine-tuned model summaries."""
    pairs = []
    with open(FINETUNED_PATH) as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs


def evaluate_model(
    model_name="groq_base",
    use_existing_summaries=True,
):
    """Evaluate summaries and store results."""
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    engine = get_engine()
    Base.metadata.create_all(engine)

    if model_name == "mistral_finetuned":
        pairs = load_finetuned_pairs()
    else:
        pairs = load_pairs()
        # Only evaluate first 200 to match finetuned set
        pairs = pairs[:200]
    logger.info(f"Evaluating {len(pairs)} summaries for model: {model_name}")

    # Resume support
    if os.path.exists(RESULTS_PATH):
        existing = pd.read_csv(RESULTS_PATH)
        existing_model = existing[existing["model_name"] == model_name]
        done_asins = set(existing_model["asin"].tolist())
        logger.info(f"Resuming: {len(done_asins)} already evaluated")
    else:
        existing = pd.DataFrame()
        done_asins = set()

    rows = []
    for i, pair in enumerate(pairs):
        asin = pair["asin"]
        if asin in done_asins:
            continue

        summary = pair["summary"]
        reviews_input = pair["input"]

        scores = judge_summary(client, reviews_input, summary)

        row = {
            "asin": asin,
            "model_name": model_name,
            "summary": summary[:500],
            **scores,
        }
        rows.append(row)

        done = len(done_asins) + len(rows)
        if done % 20 == 0:
            new_df = pd.DataFrame(rows)
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined.to_csv(RESULTS_PATH, index=False)
            existing = combined
            rows = []
            logger.info(f"Evaluated {done} / {len(pairs)}")

        if len(rows) > 0 and (len(done_asins) + len(rows)) % 28 == 0:
            time.sleep(62)

    if rows:
        new_df = pd.DataFrame(rows)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined.to_csv(RESULTS_PATH, index=False)

    # Store in PostgreSQL
    final = pd.read_csv(RESULTS_PATH)
    model_results = final[final["model_name"] == model_name]
    model_results.to_sql("evaluations", engine, if_exists="append", index=False)

    # Print summary
    logger.info(f"\n=== {model_name} Evaluation Results ===")
    for c in CRITERIA:
        avg = model_results[c].mean()
        logger.info(f"  {c}: {avg:.2f}/5")
    overall = model_results["overall_score"].mean()
    logger.info(f"  OVERALL: {overall:.2f}/5")
    target = "PASS" if overall > 4.0 else "FAIL"
    logger.info(f"  Target (>4.0): {target}")

    return model_results


if __name__ == "__main__":
    import sys

    model = sys.argv[1] if len(sys.argv) > 1 else "groq_base"
    evaluate_model(model_name=model)
