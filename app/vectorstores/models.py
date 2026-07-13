from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class VectorRecord:
    """A complete vector-store record ready for indexing or retrieval."""

    id: str
    document: str
    embedding: list[float]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class VectorSearchResult:
    """One vector search result returned by a vector-store backend."""

    id: str
    document: str
    metadata: dict[str, Any]
    distance: float
    embedding: list[float] | None = None


@dataclass(frozen=True)
class UpsertResult:
    """Summary counts returned after vector-store upsert operations."""

    attempted: int
    upserted: int
    skipped: int = 0
    failed: int = 0
