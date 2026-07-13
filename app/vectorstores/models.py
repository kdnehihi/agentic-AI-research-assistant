from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class VectorRecord:
    id: str
    document: str
    embedding: list[float]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class VectorSearchResult:
    id: str
    document: str
    metadata: dict[str, Any]
    distance: float
    embedding: list[float] | None = None


@dataclass(frozen=True)
class UpsertResult:
    attempted: int
    upserted: int
    skipped: int = 0
    failed: int = 0
