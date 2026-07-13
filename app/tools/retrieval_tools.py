from __future__ import annotations

from dataclasses import asdict
from typing import Any

from app.agent.state import AgentState
from app.config import get_settings
from app.retrieval.embedding_adapter import ExistingEmbeddingAdapter, ExistingEmbedderInterface
from app.retrieval.models import (
    RetrievalFilters,
    RetrievalRequest,
    SemanticMetadataHints,
)
from app.retrieval.retriever import MetadataAwareRetriever
from app.tools.embedding_tools import DEFAULT_BGE_MODEL_NAME, load_bge_embedder
from app.vectorstores.base import VectorStore
from app.vectorstores.chroma_store import ChromaVectorStore


def retrieve_chunks_from_knowledge_base(
    state: AgentState,
    *,
    query: str | None = None,
    knowledge_base_ids: tuple[str, ...] = (),
    top_k: int | None = None,
    candidate_k: int | None = None,
    metadata_hints: SemanticMetadataHints | None = None,
    embedder: ExistingEmbedderInterface | None = None,
    vector_store: VectorStore | None = None,
    model_name: str = DEFAULT_BGE_MODEL_NAME,
    embedding_dimension: int = 384,
) -> dict[str, Any]:
    settings = get_settings()
    resolved_query = query or state.topic
    embedder = embedder or ExistingEmbeddingAdapter(
        embedder=load_bge_embedder(model_name=model_name),
        model_name=model_name,
    )
    vector_store = vector_store or ChromaVectorStore(
        embedding_model_id=model_name,
        embedding_dimension=embedding_dimension,
    )
    retriever = MetadataAwareRetriever(
        embedder=embedder,
        vector_store=vector_store,
    )
    request = RetrievalRequest(
        query=resolved_query,
        top_k=top_k or settings.retrieval_default_top_k,
        candidate_k=candidate_k or settings.retrieval_default_candidate_k,
        filters=RetrievalFilters(knowledge_base_ids=knowledge_base_ids),
        metadata_hints=metadata_hints or SemanticMetadataHints(),
        metadata_weight=settings.retrieval_metadata_weight,
    )
    results = retriever.retrieve(request)
    return {
        "status": "success",
        "query": resolved_query,
        "retrieved": len(results),
        "results": [asdict(result) for result in results],
        "summary": f"Retrieved {len(results)} chunks from vector knowledge base.",
    }


def retrieve_chunks_from_papers(
    state: AgentState,
    *,
    query: str | None = None,
    paper_ids: tuple[str, ...] = (),
    top_k: int | None = None,
    candidate_k: int | None = None,
    embedder: ExistingEmbedderInterface | None = None,
    vector_store: VectorStore | None = None,
    model_name: str = DEFAULT_BGE_MODEL_NAME,
    embedding_dimension: int = 384,
) -> dict[str, Any]:
    settings = get_settings()
    if not paper_ids:
        paper_ids = tuple(
            paper.paper_id
            for paper in state.selected_papers
            if paper.paper_id
        )
    embedder = embedder or ExistingEmbeddingAdapter(
        embedder=load_bge_embedder(model_name=model_name),
        model_name=model_name,
    )
    vector_store = vector_store or ChromaVectorStore(
        embedding_model_id=model_name,
        embedding_dimension=embedding_dimension,
    )
    retriever = MetadataAwareRetriever(embedder=embedder, vector_store=vector_store)
    request = RetrievalRequest(
        query=query or state.topic,
        top_k=top_k or settings.retrieval_default_top_k,
        candidate_k=candidate_k or settings.retrieval_default_candidate_k,
        filters=RetrievalFilters(paper_ids=paper_ids),
        metadata_weight=settings.retrieval_metadata_weight,
    )
    results = retriever.retrieve(request)
    return {
        "status": "success",
        "query": query or state.topic,
        "retrieved": len(results),
        "results": [asdict(result) for result in results],
        "summary": f"Retrieved {len(results)} chunks from selected papers.",
    }
