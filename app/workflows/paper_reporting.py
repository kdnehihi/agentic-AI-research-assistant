from __future__ import annotations

from typing import Any, Literal

from app.agent.state import AgentState, PaperSummary
from app.llm.client import LLMClient
from app.tools.llm_summary_tools import summarize_papers_with_llm
from app.tools.report_tools import generate_report_from_abstracts, summarize_papers_from_abstracts
from app.workflows.paper_resolution import papers_by_id, set_selected_for_ids


def summarize_papers_workflow(
    state: AgentState,
    *,
    paper_ids: list[str],
    summary_mode: Literal["abstract", "full_paper", "method", "contributions", "limitations"] = "abstract",
    llm_client: LLMClient | None = None,
) -> dict[str, Any]:
    """Summarize explicit papers while keeping summary implementation reusable."""

    papers, missing = papers_by_id(state, paper_ids)
    previous_selected = list(state.selected_papers)
    set_selected_for_ids(state, papers)
    if summary_mode == "abstract":
        observation = summarize_papers_with_llm(state, llm_client=llm_client)
    else:
        observation = summarize_papers_from_abstracts(state)
        observation["status"] = "partial_success"
        observation["summary_mode_note"] = (
            f"Mode '{summary_mode}' currently falls back to abstract summaries."
        )
    state.set_selected_papers(previous_selected)

    return {
        "status": "partial_success" if missing else observation["status"],
        "paper_ids": [paper.paper_id for paper in papers if paper.paper_id],
        "missing_paper_ids": missing,
        "summary_mode": summary_mode,
        "summaries": [_summary_record(summary) for summary in state.paper_summaries],
        "summary": observation.get("summary", ""),
    }


def generate_paper_report_workflow(
    state: AgentState,
    *,
    paper_ids: list[str],
    report_type: Literal["digest", "comparison", "literature_review", "brief"] = "digest",
    user_focus: str | None = None,
    llm_client: LLMClient | None = None,
) -> dict[str, Any]:
    """Resolve papers, ensure summaries exist, and generate the markdown report."""

    papers, missing = papers_by_id(state, paper_ids)
    previous_selected = list(state.selected_papers)
    previous_topic = state.topic
    if user_focus:
        state.topic = user_focus
    set_selected_for_ids(state, papers)
    if not _has_summaries_for(state, paper_ids):
        summarize_papers_workflow(
            state,
            paper_ids=[paper.paper_id for paper in papers if paper.paper_id],
            summary_mode="abstract",
            llm_client=llm_client,
        )
        set_selected_for_ids(state, papers)
    observation = generate_report_from_abstracts(state)
    state.set_selected_papers(previous_selected)
    state.topic = previous_topic
    return {
        "status": "partial_success" if missing else observation["status"],
        "paper_ids": [paper.paper_id for paper in papers if paper.paper_id],
        "missing_paper_ids": missing,
        "report_type": report_type,
        "report": state.report,
        "summary": observation.get("summary", ""),
    }


def _has_summaries_for(state: AgentState, paper_ids: list[str]) -> bool:
    """Return whether all requested paper ids already have summaries in state."""

    summary_ids = {summary.paper_id for summary in state.paper_summaries}
    return all(paper_id in summary_ids for paper_id in paper_ids)


def _summary_record(summary: PaperSummary) -> dict[str, Any]:
    """Convert a PaperSummary to compact structured output."""

    return summary.model_dump(mode="json")
