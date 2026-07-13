from __future__ import annotations

from typing import Any

from app.agent.state import AgentState
from app.storage.paper_store import PaperStore
from app.vectorstores.base import VectorStore
from app.workflows.paper_ingestion import ensure_papers_retrievable_workflow


def ensure_papers_retrievable(
    state: AgentState,
    *,
    paper_ids: list[str],
    force_reindex: bool = False,
    store: PaperStore | None = None,
    vector_store: VectorStore | None = None,
) -> dict[str, Any]:
    """Agent-facing tool that prepares papers for semantic retrieval."""

    return ensure_papers_retrievable_workflow(
        state=state,
        paper_ids=paper_ids,
        force_reindex=force_reindex,
        store=store,
        vector_store=vector_store,
    )
