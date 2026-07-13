from __future__ import annotations

from typing import Protocol, Sequence

from app.retrieval.models import RetrievalFilters
from app.vectorstores.models import UpsertResult, VectorRecord, VectorSearchResult


class VectorStore(Protocol):
    def upsert_records(
        self,
        records: Sequence[VectorRecord],
        *,
        batch_size: int = 64,
    ) -> UpsertResult:
        ...

    def search(
        self,
        *,
        query_embedding: Sequence[float],
        top_k: int,
        filters: RetrievalFilters | None = None,
        include_embeddings: bool = False,
    ) -> list[VectorSearchResult]:
        ...

    def get_by_ids(self, ids: Sequence[str]) -> list[VectorRecord]:
        ...

    def get_by_paper(self, paper_id: str) -> list[VectorRecord]:
        ...

    def delete_by_ids(self, ids: Sequence[str]) -> int:
        ...

    def delete_by_paper(self, paper_id: str) -> int:
        ...

    def count(self) -> int:
        ...
