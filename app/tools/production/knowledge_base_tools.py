from __future__ import annotations

from typing import Any, Literal

from app.agent.state import AgentState
from app.storage.paper_store import PaperStore
from app.storage.factory import create_paper_store, create_vector_store
from app.tools.knowledge_base_tools import ensure_paper_id
from app.vectorstores.base import VectorStore
from app.workflows.paper_resolution import papers_by_id


def list_papers(
    state: AgentState,
    *,
    knowledge_base_id: str | None = None,
    paper_ids: list[str] | None = None,
    published_after: str | None = None,
    published_before: str | None = None,
    added_after: str | None = None,
    limit: int = 10,
    sort_by: Literal["published_date", "added_date", "relevance"] = "published_date",
    descending: bool = True,
    store: PaperStore | None = None,
) -> dict[str, Any]:
    """Read-only tool for listing compact paper metadata from SQLite."""

    del state
    store = store or create_paper_store()
    records = store.list_paper_records(
        paper_ids=paper_ids,
        published_after=published_after,
        published_before=published_before,
        added_after=added_after,
        limit=limit,
        sort_by=sort_by,
        descending=descending,
    )
    return {
        "status": "success",
        "knowledge_base_id": knowledge_base_id,
        "count": len(records),
        "papers": [_compact_paper_record(record) for record in records],
        "note": (
            "knowledge_base_id is accepted for planner compatibility; current "
            "SQLite schema stores paper/topic history rather than named KB rows."
        ) if knowledge_base_id else None,
        "summary": f"Listed {len(records)} papers from the knowledge base.",
    }


def get_paper_metadata(
    state: AgentState,
    *,
    paper_ids: list[str],
    store: PaperStore | None = None,
    vector_store: VectorStore | None = None,
) -> dict[str, Any]:
    """Read-only tool for compact paper metadata and artifact readiness."""

    store = store or create_paper_store()
    vector_store = vector_store or _optional_vector_store()
    papers: list[dict[str, Any]] = []
    missing: list[str] = []

    state_by_id = {}
    for paper in [*state.candidate_papers, *state.selected_papers]:
        ensure_paper_id(paper)
        if paper.paper_id:
            state_by_id[paper.paper_id] = paper

    for paper_id in dict.fromkeys(paper_ids):
        stored_record = store.get_paper_record(paper_id)
        state_paper = state_by_id.get(paper_id)
        if stored_record is None and state_paper is None:
            missing.append(paper_id)
            continue

        record = stored_record or state_paper.model_dump(mode="json")
        papers.append(
            {
                **_compact_paper_record(record),
                "exists_in_kb": stored_record is not None,
                "pdf_fetched": _pdf_exists(paper_id, store, state_paper),
                "clean_text_exists": store.clean_text_path(paper_id).exists(),
                "chunks_exist": store.chunks_path(paper_id).exists(),
                "embeddings_exist": store.embeddings_path(paper_id).exists(),
                "indexed": _is_indexed(paper_id, vector_store),
            }
        )

    return {
        "status": "partial_success" if missing else "success",
        "papers": papers,
        "missing_paper_ids": missing,
        "summary": f"Resolved metadata for {len(papers)} papers; missing {len(missing)}.",
    }


def save_papers_to_kb(
    state: AgentState,
    *,
    paper_ids: list[str],
    knowledge_base_id: str = "default",
    store: PaperStore | None = None,
) -> dict[str, Any]:
    """Persist explicit paper ids to SQLite without relying on selected/candidate selectors."""

    store = store or create_paper_store()
    papers, missing = papers_by_id(state, paper_ids, store=store)
    inserted: list[str] = []
    already_present: list[str] = []
    failed = [
        {
            "paper_id": paper_id,
            "stage": "metadata",
            "error_type": "missing_paper_metadata",
            "message": "Paper was not found in runtime state or SQLite.",
            "retryable": False,
        }
        for paper_id in missing
    ]

    for paper in papers:
        ensure_paper_id(paper)
        paper_id = paper.paper_id or ""
        if store.paper_exists(paper_id):
            already_present.append(paper_id)
            continue
        try:
            store.save_paper(paper, topic=knowledge_base_id, selected=True)
            inserted.append(paper_id)
        except Exception as exc:
            failed.append(
                {
                    "paper_id": paper_id,
                    "stage": "save",
                    "error_type": "persistent_store_unavailable",
                    "message": str(exc),
                    "retryable": True,
                }
            )

    status = "success" if not failed else "partial_success" if inserted or already_present else "failed"
    return {
        "status": status,
        "knowledge_base_id": knowledge_base_id,
        "inserted_paper_ids": inserted,
        "updated_paper_ids": [],
        "already_present_paper_ids": already_present,
        "failed": failed,
        "summary": (
            f"Saved {len(inserted)} new papers; {len(already_present)} were already present; "
            f"{len(failed)} failed."
        ),
    }


def _compact_paper_record(record: dict[str, Any]) -> dict[str, Any]:
    """Return planner-safe paper metadata without full text or embeddings."""

    return {
        "paper_id": record.get("paper_id"),
        "title": record.get("title"),
        "authors": record.get("authors") or [],
        "source": record.get("source"),
        "source_url": record.get("url"),
        "published_date": record.get("published_date"),
        "added_date": record.get("added_date"),
    }


def _optional_vector_store() -> VectorStore | None:
    """Open Chroma if available; metadata tools should still work without it."""

    try:
        return create_vector_store()
    except Exception:
        return None


def _pdf_exists(paper_id: str, store: PaperStore, state_paper) -> bool:
    """Return whether a fetched PDF exists for metadata reporting."""

    if state_paper and state_paper.full_text_path:
        from pathlib import Path

        return Path(state_paper.full_text_path).exists()
    return store.pdf_path(paper_id).exists()


def _is_indexed(paper_id: str, vector_store: VectorStore | None) -> bool:
    """Return whether vector records exist for this paper."""

    if vector_store is None:
        return False
    try:
        return bool(vector_store.get_by_paper(paper_id))
    except Exception:
        return False
