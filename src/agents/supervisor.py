"""Supervisor agent — reviews rewritten listings and sets
final approval status using rule-based logic (no LLM).
"""

from __future__ import annotations

from loguru import logger


def supervise(state: dict) -> dict:
    """Review the rewritten listing and decide final status.

    Adds to state:
        final_status   – "approved", "needs_review", or "rejected"
        supervisor_notes – reasoning behind the decision

    Rules
    -----
    1. status == "no_changes_needed"  → approved
    2. len(mismatches) > 5            → needs_review
    3. all mismatches low severity    → approved
    4. any mismatch high severity     → needs_review
    5. fallback                       → approved
    """
    asin = state.get("asin", "unknown")
    status = state.get("status", "")
    mismatches: list[dict] = state.get("mismatches", [])

    logger.info(
        "Supervisor reviewing ASIN={} | status={} | "
        "mismatches={}",
        asin,
        status,
        len(mismatches),
    )

    # Rule 1 — nothing to change
    if status == "no_changes_needed":
        state["final_status"] = "approved"
        state["supervisor_notes"] = "No issues found"
        logger.info("ASIN={} → approved (no changes)", asin)
        return state

    # Rule 2 — too many changes for auto-approval
    if len(mismatches) > 5:
        state["final_status"] = "needs_review"
        state["supervisor_notes"] = (
            "Too many changes, needs human review"
        )
        logger.warning(
            "ASIN={} → needs_review ({} mismatches)",
            asin,
            len(mismatches),
        )
        return state

    # Rule 3 & 4 — severity-based decision
    severities = [
        m.get("severity", "low") for m in mismatches
    ]
    has_high = any(s == "high" for s in severities)
    all_low = all(s == "low" for s in severities)

    if all_low:
        state["final_status"] = "approved"
        state["supervisor_notes"] = (
            "All mismatches are low severity — auto-approved"
        )
        logger.info("ASIN={} → approved (all low)", asin)
        return state

    if has_high:
        state["final_status"] = "needs_review"
        state["supervisor_notes"] = (
            "High severity mismatch detected — "
            "requires human review"
        )
        logger.warning(
            "ASIN={} → needs_review (high severity)", asin
        )
        return state

    # Rule 5 — fallback
    state["final_status"] = "approved"
    state["supervisor_notes"] = "Changes look reasonable"
    logger.info("ASIN={} → approved (default)", asin)
    return state
