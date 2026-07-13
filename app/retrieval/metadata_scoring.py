from __future__ import annotations

import re
from typing import Any

from app.retrieval.models import SemanticMetadataHints


HINT_FIELDS = (
    "topics",
    "methods",
    "datasets",
    "tasks",
    "models",
    "evaluation_metrics",
)


def metadata_match_score(
    metadata: dict[str, Any],
    hints: SemanticMetadataHints,
) -> float:
    """Score how well chunk metadata matches soft semantic hints."""

    scores: list[float] = []

    for field in HINT_FIELDS:
        requested = getattr(hints, field)
        if not requested:
            continue
        requested_values = {_normalize_tag(value) for value in requested if value}
        metadata_values = {
            _normalize_tag(value)
            for value in _metadata_values(metadata.get(field))
            if value
        }
        if not requested_values:
            continue
        scores.append(
            len(requested_values & metadata_values) / len(requested_values)
        )

    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def has_metadata_hints(hints: SemanticMetadataHints) -> bool:
    """Return whether any metadata hint field contains requested values."""

    return any(getattr(hints, field) for field in HINT_FIELDS)


def _metadata_values(value: Any) -> list[str]:
    """Normalize a scalar or iterable metadata value into a string list."""

    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value if item is not None]


def _normalize_tag(value: Any) -> str:
    """Normalize free-form labels into stable comparable tags."""

    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", str(value).lower())).strip("_")
