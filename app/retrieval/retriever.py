from __future__ import annotations

from app.retrieval.embedding_adapter import ExistingEmbedderInterface
from app.retrieval.metadata_scoring import has_metadata_hints, metadata_match_score
from app.retrieval.models import RetrievedChunk, RetrievalRequest
from app.vectorstores.base import VectorStore
from app.vectorstores.errors import RetrievalError


class MetadataAwareRetriever:
    def __init__(
        self,
        embedder: ExistingEmbedderInterface,
        vector_store: VectorStore,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store

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
            raise RetrievalError("Failed to retrieve chunks.") from exc

        use_hints = has_metadata_hints(request.metadata_hints)
        ranked: list[RetrievedChunk] = []
        for candidate in candidates:
            semantic_score = _semantic_score_from_cosine_distance(candidate.distance)
            metadata_score = (
                metadata_match_score(candidate.metadata, request.metadata_hints)
                if use_hints
                else 0.0
            )
            final_score = (
                (1.0 - request.metadata_weight) * semantic_score
                + request.metadata_weight * metadata_score
                if use_hints
                else semantic_score
            )
            ranked.append(
                RetrievedChunk(
                    chunk_id=candidate.id,
                    paper_id=str(candidate.metadata["paper_id"]),
                    document=candidate.document,
                    metadata=candidate.metadata,
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


def _semantic_score_from_cosine_distance(distance: float) -> float:
    return max(0.0, min(1.0, 1.0 - distance / 2.0))
