from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RetrievalFilters:
    knowledge_base_ids: tuple[str, ...] = ()
    paper_ids: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()
    languages: tuple[str, ...] = ()
    sections: tuple[str, ...] = ()
    section_groups: tuple[str, ...] = ()
    chunk_types: tuple[str, ...] = ()
    text_sources: tuple[str, ...] = ()
    published_from_yyyymmdd: int | None = None
    published_to_yyyymmdd: int | None = None

    def __post_init__(self) -> None:
        if (
            self.published_from_yyyymmdd is not None
            and self.published_to_yyyymmdd is not None
            and self.published_from_yyyymmdd > self.published_to_yyyymmdd
        ):
            raise ValueError("published_from_yyyymmdd must be <= published_to_yyyymmdd.")


@dataclass(frozen=True)
class SemanticMetadataHints:
    topics: tuple[str, ...] = ()
    methods: tuple[str, ...] = ()
    datasets: tuple[str, ...] = ()
    tasks: tuple[str, ...] = ()
    models: tuple[str, ...] = ()
    evaluation_metrics: tuple[str, ...] = ()


@dataclass(frozen=True)
class RetrievalRequest:
    query: str
    top_k: int = 5
    candidate_k: int | None = None
    filters: RetrievalFilters = field(default_factory=RetrievalFilters)
    metadata_hints: SemanticMetadataHints = field(default_factory=SemanticMetadataHints)
    metadata_weight: float = 0.15

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("query cannot be blank.")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive.")
        if self.candidate_k is not None and self.candidate_k < self.top_k:
            raise ValueError("candidate_k must be >= top_k.")
        if not 0.0 <= self.metadata_weight <= 1.0:
            raise ValueError("metadata_weight must be between 0 and 1.")

    @property
    def resolved_candidate_k(self) -> int:
        return self.candidate_k or max(self.top_k * 4, 20)


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    paper_id: str
    document: str
    metadata: dict[str, Any]
    distance: float
    semantic_score: float
    metadata_score: float
    final_score: float
    rank: int
