from __future__ import annotations

from dataclasses import dataclass

from app.retrieval.embedding_adapter import ExistingEmbedderInterface
from app.retrieval.lexical_scoring import bm25_scores_for_query, normalize_scores
from app.retrieval.models import RetrievedChunk, RetrievalRequest
from app.retrieval.query_intent import infer_section_groups_from_query
from app.retrieval.retriever import _semantic_score_from_cosine_distance
from app.vectorstores.base import VectorStore
from app.vectorstores.errors import RetrievalError


@dataclass(frozen=True)
class HybridScoreWeights:
    semantic: float = 0.65
    bm25: float = 0.25
    metadata: float = 0.10

    def total(self) -> float:
        return self.semantic + self.bm25 + self.metadata


class HybridRetriever:
    def __init__(
        self,
        embedder: ExistingEmbedderInterface,
        vector_store: VectorStore,
        weights: HybridScoreWeights | None = None,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.weights = weights or HybridScoreWeights()
        if self.weights.total() <= 0.0:
            raise ValueError("Hybrid score weights must sum to a positive value.")

    def retrieve(self, request: RetrievalRequest) -> list[RetrievedChunk]:
        try:
            query_embedding = self.embedder.embed_query(request.query)
            candidates = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=request.resolved_candidate_k,
                filters=request.filters,
                include_embeddings=False,
            )
        except Exception as exc:
            raise RetrievalError("Failed to retrieve chunks with hybrid retrieval.") from exc

        candidate_documents = {
            candidate.id: _embedding_text_for_candidate(candidate.document, candidate.metadata)
            for candidate in candidates
        }
        bm25_scores = normalize_scores(
            bm25_scores_for_query(
                query=request.query,
                documents=candidate_documents,
            )
        )
        section_groups = infer_section_groups_from_query(request.query)

        ranked: list[RetrievedChunk] = []
        for candidate in candidates:
            semantic_score = _semantic_score_from_cosine_distance(candidate.distance)
            bm25_score = bm25_scores.get(candidate.id, 0.0)
            metadata_score = _section_intent_score(
                expected_section_groups=section_groups,
                metadata=candidate.metadata,
            )
            final_score = _hybrid_score(
                semantic_score=semantic_score,
                bm25_score=bm25_score,
                metadata_score=metadata_score,
                weights=self.weights,
            )
            metadata = dict(candidate.metadata)
            metadata["bm25_score"] = bm25_score
            metadata["hybrid_metadata_score"] = metadata_score
            ranked.append(
                RetrievedChunk(
                    chunk_id=candidate.id,
                    paper_id=str(candidate.metadata["paper_id"]),
                    document=candidate.document,
                    metadata=metadata,
                    distance=candidate.distance,
                    semantic_score=semantic_score,
                    metadata_score=metadata_score,
                    final_score=final_score,
                    rank=0,
                )
            )

        ranked.sort(
            key=lambda result: (
                -result.final_score,
                -result.semantic_score,
                result.chunk_id,
            )
        )
        return [
            RetrievedChunk(
                chunk_id=result.chunk_id,
                paper_id=result.paper_id,
                document=result.document,
                metadata=result.metadata,
                distance=result.distance,
                semantic_score=result.semantic_score,
                metadata_score=result.metadata_score,
                final_score=result.final_score,
                rank=index,
            )
            for index, result in enumerate(ranked[:request.top_k], start=1)
        ]


def _hybrid_score(
    semantic_score: float,
    bm25_score: float,
    metadata_score: float,
    weights: HybridScoreWeights,
) -> float:
    return (
        weights.semantic * semantic_score
        + weights.bm25 * bm25_score
        + weights.metadata * metadata_score
    ) / weights.total()


def _section_intent_score(
    expected_section_groups: tuple[str, ...],
    metadata: dict,
) -> float:
    if not expected_section_groups:
        return 0.0
    return 1.0 if metadata.get("section_group") in expected_section_groups else 0.0


def _embedding_text_for_candidate(document: str, metadata: dict) -> str:
    return (
        f"Section: {metadata.get('section', '')}\n"
        "Content:\n"
        f"{document}"
    )
