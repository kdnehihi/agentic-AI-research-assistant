from __future__ import annotations

from collections.abc import Iterable

from app.agent.state import AgentState, Paper
from app.storage.paper_store import PaperStore
from app.tools.knowledge_base_tools import ensure_paper_id


def papers_by_id(
    state: AgentState,
    paper_ids: Iterable[str],
    store: PaperStore | None = None,
) -> tuple[list[Paper], list[str]]:
    """Resolve explicit paper ids from runtime state first, then SQLite."""

    store = store or PaperStore()
    wanted = list(dict.fromkeys(paper_ids))
    state_papers: dict[str, Paper] = {}
    for paper in [*state.candidate_papers, *state.selected_papers]:
        ensure_paper_id(paper)
        if paper.paper_id:
            state_papers[paper.paper_id] = paper

    resolved: list[Paper] = []
    missing: list[str] = []
    for paper_id in wanted:
        paper = state_papers.get(paper_id) or store.get_paper(paper_id)
        if paper is None:
            missing.append(paper_id)
            continue
        resolved.append(paper)

    return resolved, missing


def set_selected_for_ids(state: AgentState, papers: list[Paper]) -> None:
    """Set selected papers to an explicit list while preserving AgentState behavior."""

    state.set_selected_papers(papers)
