from __future__ import annotations

from typing import Any

from app.agent.state import AgentState
from app.storage.paper_store import PaperStore


def build_fallback_paper_id(source: str, title: str, url: str | None = None) -> str:
    """
    Build a stable fallback id for papers that do not come with an external id.
    """
    if url:
        return f"{source}:{url.rstrip('/').split('/')[-1]}"

    normalized_title = "_".join(title.lower().split())
    return f"{source}:{normalized_title}"


def ensure_paper_id(paper) -> None:
    if not paper.paper_id:
        paper.paper_id = build_fallback_paper_id(
            source=paper.source,
            title=paper.title,
            url=paper.url,
        )


def filter_seen_papers(
    state: AgentState,
    store: PaperStore | None = None,
) -> dict[str, Any]:
    """
    Remove papers that already exist in the persistent paper store.

    This prevents the agent from repeatedly surfacing papers it has seen
    in previous runs.
    """
    store = store or PaperStore()

    seen_ids = store.get_seen_paper_ids()

    before = len(state.candidate_papers)

    for paper in state.candidate_papers:
        ensure_paper_id(paper)

    new_papers = [
        paper
        for paper in state.candidate_papers
        if paper.paper_id not in seen_ids
    ]

    removed_seen = before - len(new_papers)

    state.set_candidate_papers(new_papers)

    return {
        "status": "success",
        "before": before,
        "after": len(new_papers),
        "removed_seen": removed_seen,
        "summary": f"Filtered {removed_seen} previously seen papers.",
    }


def save_candidate_papers_to_kb(
    state: AgentState,
    store: PaperStore | None = None,
) -> dict[str, Any]:
    """
    Save all candidate papers to the persistent paper store.

    Use this if you want the system to remember every paper it retrieved,
    even papers that were not selected in the final report.
    """
    store = store or PaperStore()

    for paper in state.candidate_papers:
        ensure_paper_id(paper)

    saved_count = store.save_papers(
        papers=state.candidate_papers,
        topic=state.topic,
        selected=False,
    )

    return {
        "status": "success",
        "saved": saved_count,
        "summary": f"Saved {saved_count} candidate papers to knowledge base.",
    }


def save_selected_papers_to_kb(
    state: AgentState,
    store: PaperStore | None = None,
) -> dict[str, Any]:
    """
    Save selected papers to the persistent paper store.

    This is usually called at the end of the workflow after ranking/filtering.
    """
    store = store or PaperStore()

    for paper in state.selected_papers:
        ensure_paper_id(paper)

    saved_count = store.save_papers(
        papers=state.selected_papers,
        topic=state.topic,
        selected=True,
    )

    return {
        "status": "success",
        "saved": saved_count,
        "summary": f"Saved {saved_count} selected papers to knowledge base.",
    }


def remove_papers_from_kb(
    state: AgentState,
    paper_ids: list[str] | None = None,
    store: PaperStore | None = None,
) -> dict[str, Any]:
    """
    Remove papers from the persistent paper store.

    If paper_ids is not provided, this removes the currently selected papers.
    """
    store = store or PaperStore()

    if paper_ids is None:
        for paper in state.selected_papers:
            ensure_paper_id(paper)

        paper_ids = [
            paper.paper_id
            for paper in state.selected_papers
            if paper.paper_id
        ]

    unique_paper_ids = list(dict.fromkeys(paper_ids))
    removed_count = store.remove_papers(unique_paper_ids)
    missing_count = len(unique_paper_ids) - removed_count

    return {
        "status": "success",
        "requested": len(unique_paper_ids),
        "removed": removed_count,
        "missing": missing_count,
        "summary": (
            f"Removed {removed_count} papers from knowledge base; "
            f"{missing_count} were not found."
        ),
    }
