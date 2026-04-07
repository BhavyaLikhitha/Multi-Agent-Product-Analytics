"""Listing Rewriter agent -- generates improved listing.

Uses Groq LLM to rewrite the product listing based on
audit mismatches and customer complaint patterns.
"""

import json
import os

from dotenv import load_dotenv
from groq import Groq
from loguru import logger

load_dotenv()

REWRITE_PROMPT = (
    "You are a product listing specialist. "
    "Rewrite the product listing to be more honest "
    "and accurate based on customer feedback.\n\n"
    "Rules:\n"
    "- Keep accurate claims unchanged\n"
    "- Remove or correct misleading claims\n"
    "- Add honest disclaimers where needed\n"
    "- Maintain professional product listing tone\n"
    "- Keep it concise\n\n"
    "Respond with ONLY a JSON object:\n"
    '{"title": "improved title", '
    '"description": "improved description", '
    '"changes": ["change 1", "change 2"]}'
)


def rewrite_listing(state: dict) -> dict:
    asin = state["asin"]
    mismatches = state.get("mismatches", [])

    if not mismatches:
        logger.info(
            "No mismatches for {} — skipping rewrite",
            asin,
        )
        state.update(
            {
                "rewritten_title": state.get("product_title", ""),
                "rewritten_description": state.get("product_description", ""),
                "changes_made": [],
                "status": "no_changes_needed",
            }
        )
        return state

    logger.info(
        "Rewriting listing for {} ({} mismatches)",
        asin,
        len(mismatches),
    )

    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    mismatch_text = "\n".join(
        f"- [{m['severity']}] {m['complaint']}" for m in mismatches
    )

    top_issues = state.get("top_issues", [])
    issues_text = ", ".join(f"{iss} ({cnt}x)" for iss, cnt in top_issues[:5])

    user_msg = (
        f"Current Title: {state.get('product_title', '')}\n"
        f"Current Description: "
        f"{state.get('product_description', '')[:500]}\n"
        f"Current Features: "
        f"{state.get('product_features', '')[:300]}\n\n"
        f"Customer Complaints:\n{mismatch_text}\n\n"
        f"Top Issues: {issues_text}\n\n"
        f"Rewrite the listing to address these issues."
    )

    try:
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": REWRITE_PROMPT,
                },
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
            max_tokens=500,
        )
        content = resp.choices[0].message.content.strip()

        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            result = json.loads(content[start:end])
            rewritten_title = result.get("title", state.get("product_title", ""))
            rewritten_desc = result.get(
                "description",
                state.get("product_description", ""),
            )
            changes = result.get("changes", [])
        else:
            rewritten_title = state.get("product_title", "")
            rewritten_desc = content
            changes = ["Full rewrite generated"]

    except Exception as e:
        logger.warning(f"Rewrite failed: {e}")
        rewritten_title = state.get("product_title", "")
        rewritten_desc = state.get("product_description", "")
        changes = [f"Rewrite failed: {e}"]

    state.update(
        {
            "rewritten_title": rewritten_title,
            "rewritten_description": rewritten_desc,
            "changes_made": changes,
            "status": "pending_approval",
        }
    )

    logger.info("Rewrite complete: {} changes", len(changes))
    return state


if __name__ == "__main__":
    state = {
        "asin": "B01G8JO5F2",
        "product_title": "Bluetooth Headphones - 8hr Battery",
        "product_description": "Long lasting battery...",
        "product_features": "8 hour battery; waterproof",
        "mismatches": [
            {
                "claim": "8hr battery",
                "complaint": "battery dies in 4 hours",
                "severity": "high",
            }
        ],
        "audit_summary": "Battery claims don't match",
        "top_issues": [
            ("battery", 50),
            ("disconnecting", 30),
        ],
    }
    result = rewrite_listing(state)
    print(result["rewritten_description"])
    print(result["changes_made"])
