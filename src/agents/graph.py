"""LangGraph orchestration -- chains the four listing-
optimization agents into a single executable pipeline.

Flow:
    Analyzer -> Auditor -> [Rewriter if mismatches] -> Supervisor
"""

from __future__ import annotations

from typing import Any, List, Optional

from langgraph.graph import END, StateGraph
from loguru import logger
from typing_extensions import TypedDict

from src.agents.analyzer import analyze_product
from src.agents.auditor import audit_listing
from src.agents.rewriter import rewrite_listing
from src.agents.supervisor import supervise


# ------------------------------------------------------------------
# Shared state schema
# ------------------------------------------------------------------

class PipelineState(TypedDict, total=False):
    """Union of all keys produced by every agent."""

    # input
    asin: str

    # analyzer outputs
    complaint_profile: dict
    review_count: int
    avg_rating: float
    anomaly_score: float
    status: str

    # auditor outputs
    mismatches: List[dict]
    audit_summary: str

    # rewriter outputs
    rewritten_listing: str
    original_listing: str
    diff: str

    # supervisor outputs
    final_status: str
    supervisor_notes: str

    # generic
    error: Optional[str]


# ------------------------------------------------------------------
# Node wrappers (thin adapters that catch errors)
# ------------------------------------------------------------------

def _analyzer_node(state: PipelineState) -> dict:
    logger.info(">>> Entering ANALYZER node")
    try:
        return analyze_product(state)
    except Exception as exc:
        logger.error("Analyzer failed: {}", exc)
        return {"error": str(exc)}


def _auditor_node(state: PipelineState) -> dict:
    logger.info(">>> Entering AUDITOR node")
    try:
        return audit_listing(state)
    except Exception as exc:
        logger.error("Auditor failed: {}", exc)
        return {"error": str(exc)}


def _rewriter_node(state: PipelineState) -> dict:
    logger.info(">>> Entering REWRITER node")
    try:
        return rewrite_listing(state)
    except Exception as exc:
        logger.error("Rewriter failed: {}", exc)
        return {"error": str(exc)}


def _supervisor_node(state: PipelineState) -> dict:
    logger.info(">>> Entering SUPERVISOR node")
    return supervise(state)


# ------------------------------------------------------------------
# Conditional edge: skip rewriter when no mismatches
# ------------------------------------------------------------------

def _should_rewrite(state: PipelineState) -> str:
    """Return the next node name based on audit results."""
    mismatches = state.get("mismatches", [])
    status = state.get("status", "")

    if not mismatches or status == "no_changes_needed":
        logger.info(
            "No mismatches -- skipping rewriter"
        )
        return "supervisor"

    logger.info(
        "{} mismatches found -- routing to rewriter",
        len(mismatches),
    )
    return "rewriter"


# ------------------------------------------------------------------
# Graph construction
# ------------------------------------------------------------------

def build_graph() -> StateGraph:
    """Build and return the (uncompiled) StateGraph."""
    graph = StateGraph(PipelineState)

    # Add nodes
    graph.add_node("analyzer", _analyzer_node)
    graph.add_node("auditor", _auditor_node)
    graph.add_node("rewriter", _rewriter_node)
    graph.add_node("supervisor", _supervisor_node)

    # Edges
    graph.set_entry_point("analyzer")
    graph.add_edge("analyzer", "auditor")

    graph.add_conditional_edges(
        "auditor",
        _should_rewrite,
        {
            "rewriter": "rewriter",
            "supervisor": "supervisor",
        },
    )

    graph.add_edge("rewriter", "supervisor")
    graph.add_edge("supervisor", END)

    return graph


def run_pipeline(asin: str) -> dict:
    """Execute the full 4-agent pipeline for *asin*."""
    logger.info(
        "=== Starting pipeline for ASIN={} ===", asin
    )
    graph = build_graph()
    app = graph.compile()
    result = app.invoke({"asin": asin})
    logger.info(
        "=== Pipeline complete -- final_status={} ===",
        result.get("final_status"),
    )
    return result


# ------------------------------------------------------------------
# Standalone smoke test
# ------------------------------------------------------------------

if __name__ == "__main__":
    result = run_pipeline("B01G8JO5F2")
    print(f"Status: {result.get('final_status')}")
    print(f"Notes: {result.get('supervisor_notes')}")
