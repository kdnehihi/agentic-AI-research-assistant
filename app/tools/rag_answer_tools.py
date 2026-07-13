from __future__ import annotations

from typing import Any

from app.agent.state import AgentState
from app.config import get_settings
from app.llm.client import LLMClient, OpenAILLMClient
from app.retrieval.answering import RetrievalAugmentedAnswerer
from app.retrieval.embedding_adapter import ExistingEmbeddingAdapter, ExistingEmbedderInterface
from app.retrieval.hybrid_retriever import HybridRetriever, HybridScoreWeights
from app.retrieval.models import (
    RetrievalFilters,
    RetrievalRequest,
    SemanticMetadataHints,
)
from app.retrieval.retriever import MetadataAwareRetriever
from app.tools.embedding_tools import DEFAULT_BGE_MODEL_NAME, load_bge_embedder
from app.vectorstores.base import VectorStore
from app.vectorstores.chroma_store import ChromaVectorStore


def answer_question_with_retrieval(
    state: AgentState,
    *,
    query: str | None = None,
    paper_ids: tuple[str, ...] = (),
    top_k: int | None = None,
    candidate_k: int | None = None,
    metadata_hints: SemanticMetadataHints | None = None,
    metadata_weight: float | None = None,
    use_hybrid_retrieval: bool = True,
    hybrid_semantic_weight: float = 0.65,
    hybrid_bm25_weight: float = 0.25,
    hybrid_metadata_weight: float = 0.10,
    llm_client: LLMClient | None = None,
    retriever: Any | None = None,
    embedder: ExistingEmbedderInterface | None = None,
    vector_store: VectorStore | None = None,
    model_name: str = DEFAULT_BGE_MODEL_NAME,
    embedding_dimension: int = 384,
    max_context_chars: int = 12000,
    max_chunk_chars: int = 1800,
) -> dict[str, Any]:
    resolved_query = query or state.topic
    if not paper_ids:
        paper_ids = tuple(
            paper.paper_id
            for paper in state.selected_papers
            if paper.paper_id
        )

    settings = get_settings()
    retriever = retriever or _build_default_retriever(
        embedder=embedder,
        vector_store=vector_store,
        model_name=model_name,
        embedding_dimension=embedding_dimension,
        use_hybrid_retrieval=use_hybrid_retrieval,
        hybrid_weights=HybridScoreWeights(
            semantic=hybrid_semantic_weight,
            bm25=hybrid_bm25_weight,
            metadata=hybrid_metadata_weight,
        ),
    )
    llm_client = llm_client or OpenAILLMClient()
    answerer = RetrievalAugmentedAnswerer(
        retriever=retriever,
        llm_client=llm_client,
    )
    request = RetrievalRequest(
        query=resolved_query,
        top_k=top_k or settings.retrieval_default_top_k,
        candidate_k=candidate_k or settings.retrieval_default_candidate_k,
        filters=RetrievalFilters(paper_ids=paper_ids),
        metadata_hints=metadata_hints or SemanticMetadataHints(),
        metadata_weight=(
            settings.retrieval_metadata_weight
            if metadata_weight is None
            else metadata_weight
        ),
    )

    try:
        answer = answerer.answer(
            request,
            max_context_chars=max_context_chars,
            max_chunk_chars=max_chunk_chars,
        )
    except Exception as exc:
        return {
            "status": "failed",
            "query": resolved_query,
            "answer": "",
            "error": str(exc),
            "summary": "Failed to answer question with retrieved evidence.",
        }

    answer_dict = answer.to_dict()
    return {
        "status": "success",
        **answer_dict,
        "retrieved": len(answer.evidence_chunks),
        "cited": len(answer.cited_chunk_ids),
        "summary": (
            f"Answered query using {len(answer.evidence_chunks)} retrieved "
            f"evidence chunks and {len(answer.cited_chunk_ids)} cited chunks."
        ),
    }


def _build_default_retriever(
    *,
    embedder: ExistingEmbedderInterface | None,
    vector_store: VectorStore | None,
    model_name: str,
    embedding_dimension: int,
    use_hybrid_retrieval: bool,
    hybrid_weights: HybridScoreWeights | None,
) -> Any:
    resolved_embedder = embedder or ExistingEmbeddingAdapter(
        embedder=load_bge_embedder(model_name=model_name),
        model_name=model_name,
    )
    resolved_vector_store = vector_store or ChromaVectorStore(
        embedding_model_id=model_name,
        embedding_dimension=embedding_dimension,
    )
    if use_hybrid_retrieval:
        return HybridRetriever(
            embedder=resolved_embedder,
            vector_store=resolved_vector_store,
            weights=hybrid_weights,
        )

    return MetadataAwareRetriever(
        embedder=resolved_embedder,
        vector_store=resolved_vector_store,
    )
