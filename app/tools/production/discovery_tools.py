from __future__ import annotations

from typing import Any

from app.agent.state import AgentState
from app.workflows.paper_discovery import discover_papers_workflow


def discover_papers(
    state: AgentState,
    *,
    user_query: str,
    max_results: int | None = None,
    max_selected: int | None = None,
    exclude_seen: bool = True,
    use_llm_query_planner: bool = False,
) -> dict[str, Any]:
    """Agent-facing macro tool for safe paper discovery."""

    plan_step = None if use_llm_query_planner else _skip_query_planning
    return discover_papers_workflow(
        state=state,
        user_query=user_query,
        max_results=max_results,
        max_selected=max_selected,
        exclude_seen=exclude_seen,
        **({"plan_step": plan_step} if plan_step is not None else {}),
    )


def _skip_query_planning(state: AgentState) -> dict[str, Any]:
    state.search_plan = None
    return {
        "status": "skipped",
        "planner": "rule_based",
        "summary": "Skipped LLM query planning; arXiv will use the user query.",
    }
