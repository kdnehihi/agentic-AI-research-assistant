from __future__ import annotations

from pathlib import Path
from typing import Any

from app.agent.state import AgentState, Paper
from app.storage.factory import create_paper_store, create_vector_store
from app.storage.paper_store import PaperStore
from app.tools.chunking_tools import chunk_selected_papers_by_section
from app.tools.embedding_tools import embed_selected_paper_chunks
from app.tools.fetch_selected_papers import fetch_selected_papers
from app.tools.pdf_text_tools import extract_pdf_text_for_selected_papers
from app.tools.vector_store_tools import index_selected_paper_chunks
from app.vectorstores.base import VectorStore
from app.workflows.paper_resolution import papers_by_id, set_selected_for_ids


def ensure_papers_retrievable_workflow(
    state: AgentState,
    *,
    paper_ids: list[str],
    force_reindex: bool = False,
    store: PaperStore | None = None,
    vector_store: VectorStore | None = None,
) -> dict[str, Any]:
    """Fetch, extract, chunk, embed, and index only missing paper artifacts."""

    store = store or create_paper_store()
    if vector_store is None:
        try:
            vector_store = create_vector_store()
        except Exception:
            vector_store = None
    resolved_papers, missing = papers_by_id(state, paper_ids, store=store)
    previous_selected = list(state.selected_papers)

    ready: list[str] = []
    already_ready: list[str] = []
    newly_fetched: list[str] = []
    newly_extracted: list[str] = []
    newly_chunked: list[str] = []
    newly_embedded: list[str] = []
    newly_indexed: list[str] = []
    failures = [
        _failure(paper_id=paper_id, stage="metadata", message="Paper metadata was not found.")
        for paper_id in missing
    ]

    for paper in resolved_papers:
        paper_id = paper.paper_id or ""
        try:
            if _is_retrievable(paper_id, vector_store=vector_store):
                already_ready.append(paper_id)
                ready.append(paper_id)
                continue

            set_selected_for_ids(state, [paper])
            _sync_artifact_paths(state, paper_id, store)
            if force_reindex or not _pdf_exists(paper, store):
                fetch_obs = fetch_selected_papers(state=state)
                if fetch_obs["status"] == "failed":
                    raise StageError("fetch", fetch_obs.get("summary", "PDF fetch failed."))
                newly_fetched.append(paper_id)

            if force_reindex or not store.clean_text_path(paper_id).exists():
                extract_obs = extract_pdf_text_for_selected_papers(state=state, file_store=store)
                if extract_obs["status"] == "failed":
                    raise StageError("extract", extract_obs.get("summary", "Text extraction failed."))
                newly_extracted.append(paper_id)
                _sync_artifact_paths(state, paper_id, store)

            if force_reindex or not store.chunks_path(paper_id).exists():
                chunk_obs = chunk_selected_papers_by_section(state=state, file_store=store)
                if chunk_obs["status"] == "failed":
                    raise StageError("chunk", chunk_obs.get("summary", "Chunking failed."))
                newly_chunked.append(paper_id)
                _sync_artifact_paths(state, paper_id, store)

            if force_reindex or not store.embeddings_path(paper_id).exists():
                embed_obs = embed_selected_paper_chunks(state=state, file_store=store)
                if embed_obs["status"] == "failed":
                    raise StageError("embed", embed_obs.get("summary", "Embedding failed."))
                newly_embedded.append(paper_id)
                _sync_artifact_paths(state, paper_id, store)

            index_obs = index_selected_paper_chunks(state=state, vector_store=vector_store)
            if index_obs["status"] == "failed":
                raise StageError("index", index_obs.get("summary", "Indexing failed."))
            newly_indexed.append(paper_id)
            ready.append(paper_id)
        except StageError as exc:
            failures.append(_failure(paper_id=paper_id, stage=exc.stage, message=str(exc)))
        except Exception as exc:
            failures.append(_failure(paper_id=paper_id, stage="unknown", message=str(exc)))

    state.set_selected_papers(previous_selected)
    status = "success" if not failures else "partial_success" if ready else "failed"
    return {
        "status": status,
        "ready_paper_ids": ready,
        "already_ready_paper_ids": already_ready,
        "newly_fetched_paper_ids": newly_fetched,
        "newly_extracted_paper_ids": newly_extracted,
        "newly_chunked_paper_ids": newly_chunked,
        "newly_embedded_paper_ids": newly_embedded,
        "newly_indexed_paper_ids": newly_indexed,
        "failed": failures,
        "summary": f"Prepared {len(ready)} papers for semantic retrieval; failed {len(failures)}.",
    }


class StageError(Exception):
    """Stage-level error used for structured ingestion failure reporting."""

    def __init__(self, stage: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage


def _pdf_exists(paper: Paper, store: PaperStore) -> bool:
    """Return whether a paper already has an accessible local PDF."""

    if paper.full_text_path and Path(paper.full_text_path).suffix.lower() == ".pdf":
        return Path(paper.full_text_path).exists()
    return store.pdf_path(paper.paper_id or "").exists()


def _sync_artifact_paths(state: AgentState, paper_id: str, store: PaperStore) -> None:
    """Restore state path references from existing files for resumable workflows."""

    text_paths = dict(state.paper_text_paths)
    chunk_paths = dict(state.paper_chunk_paths)
    embedding_paths = dict(state.paper_embedding_paths)

    if store.clean_text_path(paper_id).exists():
        text_paths[paper_id] = str(store.clean_text_path(paper_id))
    if store.chunks_path(paper_id).exists():
        chunk_paths[paper_id] = str(store.chunks_path(paper_id))
    if store.embeddings_path(paper_id).exists():
        embedding_paths[paper_id] = str(store.embeddings_path(paper_id))

    state.set_paper_text_paths(text_paths)
    state.set_paper_chunk_paths(chunk_paths)
    state.set_paper_embedding_paths(embedding_paths)


def _is_retrievable(paper_id: str, vector_store: VectorStore | None) -> bool:
    """Return whether Chroma/vector storage already has chunks for a paper."""

    try:
        if vector_store is None:
            return False
        return bool(vector_store.get_by_paper(paper_id))
    except Exception:
        return False


def _failure(paper_id: str, stage: str, message: str) -> dict[str, Any]:
    """Build a planner-safe structured failure payload."""

    return {
        "paper_id": paper_id,
        "stage": stage,
        "error_type": "missing_prerequisite" if stage == "metadata" else "stage_failure",
        "message": message,
        "retryable": stage not in {"metadata"},
    }
