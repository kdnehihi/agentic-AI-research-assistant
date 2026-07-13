from __future__ import annotations

from typing import Any, Literal

from app.agent.state import AgentState
from app.llm.client import LLMClient
from app.workflows.paper_reporting import (
    generate_paper_report_workflow,
    summarize_papers_workflow,
)


def summarize_papers(
    state: AgentState,
    *,
    paper_ids: list[str],
    summary_mode: Literal["abstract", "full_paper", "method", "contributions", "limitations"] = "abstract",
    llm_client: LLMClient | None = None,
) -> dict[str, Any]:
    """Agent-facing summarization tool for explicit paper ids."""

    return summarize_papers_workflow(
        state=state,
        paper_ids=paper_ids,
        summary_mode=summary_mode,
        llm_client=llm_client,
    )


def generate_paper_report(
    state: AgentState,
    *,
    paper_ids: list[str],
    report_type: Literal["digest", "comparison", "literature_review", "brief"] = "digest",
    user_focus: str | None = None,
    llm_client: LLMClient | None = None,
) -> dict[str, Any]:
    """Agent-facing report generator for explicit paper ids."""

    return generate_paper_report_workflow(
        state=state,
        paper_ids=paper_ids,
        report_type=report_type,
        user_focus=user_focus,
        llm_client=llm_client,
    )
