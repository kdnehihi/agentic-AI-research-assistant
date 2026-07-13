from __future__ import annotations

from typing import Protocol, Sequence

from app.retrieval.models import RetrievalFilters
from app.vectorstores.models import UpsertResult, VectorRecord, VectorSearchResult


class VectorStore(Protocol):
    """Minimal interface implemented by vector database backends."""

    def upsert_records(
        self,
        records: Sequence[VectorRecord],
        *,
        batch_size: int = 64,
    ) -> UpsertResult:
        """Insert or replace vector records in the backend."""

        ...

    def search(
        self,
        *,
        query_embedding: Sequence[float],
        top_k: int,
        filters: RetrievalFilters | None = None,
        include_embeddings: bool = False,
    ) -> list[VectorSearchResult]:
        """Search the backend by query embedding and optional filters."""

        ...

    def get_by_ids(self, ids: Sequence[str]) -> list[VectorRecord]:
        """Load stored vector records by exact ids."""

        ...

    def get_by_paper(self, paper_id: str) -> list[VectorRecord]:
        """Load all vector records associated with one paper."""

        ...

    def delete_by_ids(self, ids: Sequence[str]) -> int:
        """Delete records by exact ids and return the number removed."""

        ...

    def delete_by_paper(self, paper_id: str) -> int:
        """Delete all records for one paper and return the number removed."""

        ...

    def count(self) -> int:
        """Return the number of stored vector records."""

        ...
