from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

from app.retrieval.models import RetrievalFilters
from app.vectorstores.errors import InvalidChunkMetadataError


REQUIRED_METADATA_FIELDS = {
    "paper_id",
    "knowledge_base_ids",
    "source",
    "language",
    "title",
    "published_year",
    "published_yyyymmdd",
    "section",
    "section_group",
    "chunk_type",
    "chunk_index",
    "word_count",
    "text_source",
    "metadata_schema_version",
    "embedding_model_id",
    "embedding_dimension",
}

ARRAY_METADATA_FIELDS = {
    "knowledge_base_ids",
    "authors",
    "topics",
    "methods",
    "datasets",
    "tasks",
    "models",
    "evaluation_metrics",
}

SEMANTIC_TAG_FIELDS = {
    "topics",
    "methods",
    "datasets",
    "tasks",
    "models",
    "evaluation_metrics",
}

SECTION_GROUP_ALIASES = {
    "abstract": "abstract",
    "front matter": "other",
    "full text": "other",
    "introduction": "introduction",
    "related work": "related_work",
    "background": "background",
    "method": "method",
    "methods": "method",
    "methodology": "method",
    "approach": "method",
    "model": "method",
    "experiments": "experiments",
    "experimental setup": "experiments",
    "experimental details": "experiments",
    "results": "results",
    "experimental results": "results",
    "discussion": "discussion",
    "discussion and analysis": "discussion",
    "analysis": "discussion",
    "limitations": "limitations",
    "ethical considerations": "limitations",
    "conclusion": "conclusion",
    "appendix": "other",
}


def normalize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Normalize chunk metadata and ensure required vector-store fields exist."""

    normalized: dict[str, Any] = {}

    for key, value in metadata.items():
        if value is None:
            continue
        if key in ARRAY_METADATA_FIELDS:
            values = _normalize_array(value, semantic=key in SEMANTIC_TAG_FIELDS)
            if values:
                normalized[key] = values
            continue
        normalized[key] = value

    if "section" in normalized:
        normalized.setdefault("section_group", section_group_for(normalized["section"]))

    validate_required_metadata(normalized)
    return normalized


def validate_required_metadata(metadata: dict[str, Any]) -> None:
    """Validate the metadata contract required before vector indexing."""

    missing = sorted(field for field in REQUIRED_METADATA_FIELDS if field not in metadata)
    if missing:
        raise InvalidChunkMetadataError(
            f"Chunk metadata is missing required fields: {', '.join(missing)}"
        )

    if not isinstance(metadata["knowledge_base_ids"], list):
        raise InvalidChunkMetadataError("knowledge_base_ids must be a list.")
    if not metadata["knowledge_base_ids"]:
        raise InvalidChunkMetadataError("knowledge_base_ids must not be empty.")


def section_group_for(section: str) -> str:
    """Map a raw section label into a normalized retrieval section group."""

    normalized = _normalize_label(section)
    return SECTION_GROUP_ALIASES.get(normalized.replace("_", " "), "other")


def build_chroma_where(filters: RetrievalFilters) -> dict[str, Any] | None:
    """Build a Chroma where clause for metadata fields that support arrays."""

    conditions: list[dict[str, Any]] = []

    if filters.paper_ids:
        conditions.append(_exact_or_in("paper_id", filters.paper_ids))
    if filters.sources:
        conditions.append(_exact_or_in("source", filters.sources))
    if filters.languages:
        conditions.append(_exact_or_in("language", filters.languages))
    if filters.sections:
        conditions.append(_exact_or_in("section", filters.sections))
    if filters.section_groups:
        conditions.append(_exact_or_in("section_group", filters.section_groups))
    if filters.chunk_types:
        conditions.append(_exact_or_in("chunk_type", filters.chunk_types))
    if filters.text_sources:
        conditions.append(_exact_or_in("text_source", filters.text_sources))
    if filters.knowledge_base_ids:
        kb_conditions = [
            {"knowledge_base_ids": {"$contains": knowledge_base_id}}
            for knowledge_base_id in filters.knowledge_base_ids
        ]
        if len(kb_conditions) == 1:
            conditions.append(kb_conditions[0])
        else:
            conditions.append({"$or": kb_conditions})

    date_conditions: list[dict[str, Any]] = []
    if filters.published_from_yyyymmdd is not None:
        date_conditions.append(
            {"published_yyyymmdd": {"$gte": filters.published_from_yyyymmdd}}
        )
    if filters.published_to_yyyymmdd is not None:
        date_conditions.append(
            {"published_yyyymmdd": {"$lte": filters.published_to_yyyymmdd}}
        )
    conditions.extend(date_conditions)

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def build_chroma_runtime_where(filters: RetrievalFilters | None) -> dict[str, Any] | None:
    """Build a runtime Chroma filter using flattened array membership flags."""

    if filters is None:
        return None

    conditions: list[dict[str, Any]] = []
    if filters.paper_ids:
        conditions.append(_exact_or_in("paper_id", filters.paper_ids))
    if filters.sources:
        conditions.append(_exact_or_in("source", filters.sources))
    if filters.languages:
        conditions.append(_exact_or_in("language", filters.languages))
    if filters.sections:
        conditions.append(_exact_or_in("section", filters.sections))
    if filters.section_groups:
        conditions.append(_exact_or_in("section_group", filters.section_groups))
    if filters.chunk_types:
        conditions.append(_exact_or_in("chunk_type", filters.chunk_types))
    if filters.text_sources:
        conditions.append(_exact_or_in("text_source", filters.text_sources))
    if filters.knowledge_base_ids:
        kb_conditions = [
            {_membership_key("knowledge_base_ids", knowledge_base_id): True}
            for knowledge_base_id in filters.knowledge_base_ids
        ]
        if len(kb_conditions) == 1:
            conditions.append(kb_conditions[0])
        else:
            conditions.append({"$or": kb_conditions})
    if filters.published_from_yyyymmdd is not None:
        conditions.append(
            {"published_yyyymmdd": {"$gte": filters.published_from_yyyymmdd}}
        )
    if filters.published_to_yyyymmdd is not None:
        conditions.append(
            {"published_yyyymmdd": {"$lte": filters.published_to_yyyymmdd}}
        )

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def to_chroma_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Serialize project metadata into Chroma-compatible scalar fields."""

    chroma_metadata: dict[str, Any] = {}
    for key, value in metadata.items():
        if key in ARRAY_METADATA_FIELDS:
            values = _normalize_array(value, semantic=key in SEMANTIC_TAG_FIELDS)
            if not values:
                continue
            chroma_metadata[key] = json.dumps(values)
            for item in values:
                chroma_metadata[_membership_key(key, item)] = True
            continue
        chroma_metadata[key] = value

    return chroma_metadata


def from_chroma_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Restore project metadata from Chroma scalar storage format."""

    restored: dict[str, Any] = {}
    for key, value in metadata.items():
        if "__has__" in key:
            continue
        if key in ARRAY_METADATA_FIELDS and isinstance(value, str):
            restored[key] = json.loads(value)
            continue
        restored[key] = value
    return restored


def published_date_to_yyyymmdd(value: str | None) -> int:
    """Convert a loose date string into an integer YYYYMMDD value."""

    if not value:
        return 0
    try:
        return int(datetime.fromisoformat(value[:10]).strftime("%Y%m%d"))
    except ValueError:
        digits = re.sub(r"\D", "", value)
        return int((digits + "00000000")[:8]) if digits else 0


def published_date_to_year(value: str | None) -> int:
    """Extract a four-digit year from a loose publication date string."""

    yyyymmdd = published_date_to_yyyymmdd(value)
    return yyyymmdd // 10000 if yyyymmdd else 0


def _exact_or_in(field: str, values: tuple[str, ...]) -> dict[str, Any]:
    """Use equality for one value and $in for multiple values."""

    if len(values) == 1:
        return {field: values[0]}
    return {field: {"$in": list(values)}}


def _normalize_array(value: Any, semantic: bool) -> list[str]:
    """Normalize array-like metadata while preserving stable unique order."""

    if isinstance(value, str):
        raw_values = [value]
    else:
        raw_values = list(value)

    values: list[str] = []
    for item in raw_values:
        if item is None:
            continue
        normalized = _normalize_tag(item) if semantic else str(item).strip()
        if normalized:
            values.append(normalized)
    return sorted(dict.fromkeys(values))


def _normalize_tag(value: Any) -> str:
    """Normalize semantic metadata labels into lowercase tags."""

    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", str(value).lower())).strip("_")


def _normalize_label(value: Any) -> str:
    """Normalize display labels for alias lookup."""

    return re.sub(r"\s+", " ", str(value).strip().lower())


def _membership_key(field: str, value: str) -> str:
    """Build a flattened boolean key used for Chroma array membership filters."""

    return f"{field}__has__{_normalize_tag(value)}"
