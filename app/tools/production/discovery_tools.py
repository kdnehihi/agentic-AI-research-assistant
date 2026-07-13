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
) -> dict[str, Any]:
    """Agent-facing macro tool for safe paper discovery."""

    return discover_papers_workflow(
        state=state,
        user_query=user_query,
        max_results=max_results,
        max_selected=max_selected,
        exclude_seen=exclude_seen,
    )
