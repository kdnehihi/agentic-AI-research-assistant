from __future__ import annotations

from dataclasses import asdict
from typing import Any

from app.agent.state import AgentState
from app.config import get_settings
from app.retrieval.embedding_adapter import ExistingEmbeddingAdapter, ExistingEmbedderInterface
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.models import RetrievalFilters, RetrievalRequest
from app.storage.factory import create_vector_store
from app.tools.embedding_tools import DEFAULT_BGE_MODEL_NAME, load_bge_embedder
from app.vectorstores.base import VectorStore
from app.vectorstores.metadata import published_date_to_yyyymmdd


def retrieve_evidence(
    state: AgentState,
    *,
    query: str,
    paper_ids: list[str] | None = None,
    knowledge_base_ids: list[str] | None = None,
    section_groups: list[str] | None = None,
    published_after: str | None = None,
    published_before: str | None = None,
    top_k: int | None = None,
    candidate_k: int | None = None,
    embedder: ExistingEmbedderInterface | None = None,
    vector_store: VectorStore | None = None,
    model_name: str = DEFAULT_BGE_MODEL_NAME,
    embedding_dimension: int = 384,
) -> dict[str, Any]:
    """Retrieve compact evidence chunks without performing ingestion."""

    del state
    settings = get_settings()
    vector_store = vector_store or create_vector_store(
        embedding_model_id=model_name,
        embedding_dimension=embedding_dimension,
    )
    missing_indexed = _missing_indexed_papers(paper_ids or [], vector_store)
    if missing_indexed:
        return {
            "status": "failed",
            "error_type": "paper_not_retrievable",
            "missing_paper_ids": missing_indexed,
            "message": (
                "Some requested papers are not indexed. Call "
                "ensure_papers_retrievable before retrieve_evidence."
            ),
            "evidence": [],
            "summary": "Retrieval prerequisite failed because papers are not indexed.",
        }

    embedder = embedder or ExistingEmbeddingAdapter(
        embedder=load_bge_embedder(model_name=model_name),
        model_name=model_name,
    )
    retriever = HybridRetriever(embedder=embedder, vector_store=vector_store)
    request = RetrievalRequest(
        query=query,
        top_k=top_k or settings.retrieval_default_top_k,
        candidate_k=candidate_k or settings.retrieval_default_candidate_k,
        filters=RetrievalFilters(
            paper_ids=tuple(paper_ids or ()),
            knowledge_base_ids=tuple(knowledge_base_ids or ()),
            section_groups=tuple(section_groups or ()),
            published_from_yyyymmdd=(
                published_date_to_yyyymmdd(published_after)
                if published_after
                else None
            ),
            published_to_yyyymmdd=(
                published_date_to_yyyymmdd(published_before)
                if published_before
                else None
            ),
        ),
    )
    try:
        results = retriever.retrieve(request)
    except Exception as exc:
        return {
            "status": "failed",
            "error_type": "retrieval_failure",
            "message": str(exc),
            "evidence": [],
            "summary": "Evidence retrieval failed.",
        }

    evidence = [_evidence_record(result) for result in results]
    return {
        "status": "success",
        "query": query,
        "retrieved": len(evidence),
        "evidence": evidence,
        "summary": f"Retrieved {len(evidence)} evidence chunks.",
    }


def _missing_indexed_papers(paper_ids: list[str], vector_store: VectorStore) -> list[str]:
    """Return requested paper ids with no vector records."""

    missing: list[str] = []
    for paper_id in paper_ids:
        try:
            if not vector_store.get_by_paper(paper_id):
                missing.append(paper_id)
        except Exception:
            missing.append(paper_id)
    return missing


def _evidence_record(result) -> dict[str, Any]:
    """Convert a RetrievedChunk into planner-safe evidence output."""

    metadata = result.metadata
    return {
        "chunk_id": result.chunk_id,
        "paper_id": result.paper_id,
        "title": metadata.get("title"),
        "section": metadata.get("section"),
        "section_group": metadata.get("section_group"),
        "page": metadata.get("page"),
        "text": result.document,
        "semantic_score": result.semantic_score,
        "lexical_score": metadata.get("bm25_score"),
        "metadata_score": result.metadata_score,
        "final_score": result.final_score,
        "rank": result.rank,
        "raw": asdict(result),
    }
