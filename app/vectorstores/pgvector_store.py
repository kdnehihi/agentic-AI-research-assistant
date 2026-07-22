from __future__ import annotations

import json
import re
from typing import Any, Sequence

from sqlalchemy import bindparam, create_engine, text
from sqlalchemy.engine import Engine

from app.config import get_settings
from app.retrieval.models import RetrievalFilters
from app.vectorstores.errors import (
    EmbeddingDimensionMismatchError,
    VectorStoreConfigurationError,
    VectorStoreError,
)
from app.vectorstores.metadata import validate_required_metadata
from app.vectorstores.models import UpsertResult, VectorRecord, VectorSearchResult


class PgVectorStore:
    """Postgres/pgvector implementation of the project VectorStore interface."""

    def __init__(
        self,
        *,
        database_url: str | None = None,
        table_name: str | None = None,
        embedding_model_id: str = "BAAI/bge-small-en-v1.5",
        embedding_dimension: int = 384,
        distance_metric: str | None = None,
        metadata_schema_version: int | None = None,
        initialize: bool = True,
    ) -> None:
        settings = get_settings()
        self.database_url = database_url or settings.database_url
        if not self.database_url:
            raise VectorStoreConfigurationError(
                "PgVectorStore requires DATABASE_URL to be configured."
            )
        self.table_name = _safe_identifier(table_name or settings.pgvector_table_name)
        self.embedding_model_id = embedding_model_id
        self.embedding_dimension = embedding_dimension
        self.distance_metric = distance_metric or settings.vector_distance_metric
        self.metadata_schema_version = (
            metadata_schema_version or settings.metadata_schema_version
        )
        self.engine = create_engine(self.database_url, pool_pre_ping=True, future=True)
        if initialize:
            self._init_db()

    def upsert_records(
        self,
        records: Sequence[VectorRecord],
        *,
        batch_size: int = 64,
    ) -> UpsertResult:
        """Validate and upsert vector records into Postgres."""

        if not records:
            return UpsertResult(attempted=0, upserted=0)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        attempted = len(records)
        statement = text(
            f"""
            INSERT INTO {self.table_name} (
                id, document, embedding, metadata_json, embedding_model_id,
                embedding_dimension, metadata_schema_version, updated_at
            )
            VALUES (
                :id, :document, CAST(:embedding AS vector),
                CAST(:metadata_json AS jsonb), :embedding_model_id,
                :embedding_dimension, :metadata_schema_version, now()
            )
            ON CONFLICT (id) DO UPDATE SET
                document = EXCLUDED.document,
                embedding = EXCLUDED.embedding,
                metadata_json = EXCLUDED.metadata_json,
                embedding_model_id = EXCLUDED.embedding_model_id,
                embedding_dimension = EXCLUDED.embedding_dimension,
                metadata_schema_version = EXCLUDED.metadata_schema_version,
                updated_at = now()
            """
        )
        try:
            with self.engine.begin() as conn:
                for start in range(0, len(records), batch_size):
                    batch = list(records[start:start + batch_size])
                    self._validate_records(batch)
                    conn.execute(
                        statement,
                        [
                            {
                                "id": record.id,
                                "document": record.document,
                                "embedding": _vector_literal(record.embedding),
                                "metadata_json": json.dumps(record.metadata),
                                "embedding_model_id": self.embedding_model_id,
                                "embedding_dimension": self.embedding_dimension,
                                "metadata_schema_version": self.metadata_schema_version,
                            }
                            for record in batch
                        ],
                    )
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError("PgVector upsert failed.") from exc

        return UpsertResult(attempted=attempted, upserted=attempted)

    def search(
        self,
        *,
        query_embedding: Sequence[float],
        top_k: int,
        filters: RetrievalFilters | None = None,
        include_embeddings: bool = False,
    ) -> list[VectorSearchResult]:
        """Search Postgres by pgvector distance and metadata filters."""

        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        self._validate_embedding(query_embedding)

        where_sql, params = _where_clause(filters)
        embedding_column = ", embedding::text AS embedding_text" if include_embeddings else ""
        statement = text(
            f"""
            SELECT id, document, metadata_json,
                   embedding <=> CAST(:query_embedding AS vector) AS distance
                   {embedding_column}
            FROM {self.table_name}
            {where_sql}
            ORDER BY embedding <=> CAST(:query_embedding AS vector)
            LIMIT :top_k
            """
        )
        params.update(
            {
                "query_embedding": _vector_literal(query_embedding),
                "top_k": top_k,
            }
        )
        try:
            with self.engine.connect() as conn:
                rows = conn.execute(statement, params).mappings().all()
        except Exception as exc:
            raise VectorStoreError("PgVector search failed.") from exc

        return [
            VectorSearchResult(
                id=row["id"],
                document=row["document"],
                metadata=_json_mapping(row["metadata_json"]),
                distance=float(row["distance"]),
                embedding=(
                    _parse_vector_literal(row["embedding_text"])
                    if include_embeddings
                    else None
                ),
            )
            for row in rows
        ]

    def get_by_ids(self, ids: Sequence[str]) -> list[VectorRecord]:
        """Load stored vector records by exact ids."""

        if not ids:
            return []
        statement = text(
            f"""
            SELECT id, document, embedding::text AS embedding_text, metadata_json
            FROM {self.table_name}
            WHERE id IN :ids
            ORDER BY id
            """
        ).bindparams(bindparam("ids", expanding=True))
        try:
            with self.engine.connect() as conn:
                rows = conn.execute(statement, {"ids": list(ids)}).mappings().all()
        except Exception as exc:
            raise VectorStoreError("PgVector get_by_ids failed.") from exc
        return [_record_from_row(row) for row in rows]

    def get_by_paper(self, paper_id: str) -> list[VectorRecord]:
        """Load all stored vector records for one paper id."""

        statement = text(
            f"""
            SELECT id, document, embedding::text AS embedding_text, metadata_json
            FROM {self.table_name}
            WHERE metadata_json->>'paper_id' = :paper_id
            ORDER BY id
            """
        )
        try:
            with self.engine.connect() as conn:
                rows = conn.execute(statement, {"paper_id": paper_id}).mappings().all()
        except Exception as exc:
            raise VectorStoreError("PgVector get_by_paper failed.") from exc
        return [_record_from_row(row) for row in rows]

    def delete_by_ids(self, ids: Sequence[str]) -> int:
        """Delete records by exact ids and return affected count."""

        if not ids:
            return 0
        statement = text(
            f"DELETE FROM {self.table_name} WHERE id IN :ids"
        ).bindparams(bindparam("ids", expanding=True))
        try:
            with self.engine.begin() as conn:
                result = conn.execute(statement, {"ids": list(ids)})
        except Exception as exc:
            raise VectorStoreError("PgVector delete_by_ids failed.") from exc
        return int(result.rowcount or 0)

    def delete_by_paper(self, paper_id: str) -> int:
        """Delete all vector records for one paper id."""

        statement = text(
            f"DELETE FROM {self.table_name} WHERE metadata_json->>'paper_id' = :paper_id"
        )
        try:
            with self.engine.begin() as conn:
                result = conn.execute(statement, {"paper_id": paper_id})
        except Exception as exc:
            raise VectorStoreError("PgVector delete_by_paper failed.") from exc
        return int(result.rowcount or 0)

    def count(self) -> int:
        """Return the number of stored vector records."""

        try:
            with self.engine.connect() as conn:
                value = conn.execute(
                    text(f"SELECT COUNT(*) FROM {self.table_name}")
                ).scalar_one()
        except Exception as exc:
            raise VectorStoreError("PgVector count failed.") from exc
        return int(value)

    def _init_db(self) -> None:
        """Create pgvector extension, table, and retrieval indexes."""

        distance_ops = {
            "cosine": "vector_cosine_ops",
            "l2": "vector_l2_ops",
            "ip": "vector_ip_ops",
        }.get(self.distance_metric, "vector_cosine_ops")
        try:
            with self.engine.begin() as conn:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                conn.execute(
                    text(
                        f"""
                        CREATE TABLE IF NOT EXISTS {self.table_name} (
                            id TEXT PRIMARY KEY,
                            document TEXT NOT NULL,
                            embedding vector({self.embedding_dimension}) NOT NULL,
                            metadata_json JSONB NOT NULL,
                            embedding_model_id TEXT NOT NULL,
                            embedding_dimension INTEGER NOT NULL,
                            metadata_schema_version INTEGER NOT NULL,
                            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                        )
                        """
                    )
                )
                conn.execute(
                    text(
                        f"""
                        CREATE INDEX IF NOT EXISTS idx_{self.table_name}_embedding
                        ON {self.table_name}
                        USING hnsw (embedding {distance_ops})
                        """
                    )
                )
                conn.execute(
                    text(
                        f"""
                        CREATE INDEX IF NOT EXISTS idx_{self.table_name}_metadata
                        ON {self.table_name}
                        USING gin (metadata_json)
                        """
                    )
                )
                conn.execute(
                    text(
                        f"""
                        CREATE INDEX IF NOT EXISTS idx_{self.table_name}_paper_id
                        ON {self.table_name} ((metadata_json->>'paper_id'))
                        """
                    )
                )
        except Exception as exc:
            raise VectorStoreConfigurationError(
                "Could not initialize PgVectorStore. Ensure DATABASE_URL points "
                "to Postgres and the pgvector extension is available."
            ) from exc

    def _validate_records(self, records: Sequence[VectorRecord]) -> None:
        for record in records:
            if not record.id:
                raise VectorStoreError("Vector record id must not be blank.")
            if not record.document:
                raise VectorStoreError(f"Vector record {record.id} has empty document.")
            self._validate_embedding(record.embedding)
            validate_required_metadata(record.metadata)
            metadata_dimension = record.metadata.get("embedding_dimension")
            if metadata_dimension != self.embedding_dimension:
                raise EmbeddingDimensionMismatchError(
                    f"Record {record.id} metadata dimension {metadata_dimension} "
                    f"does not match collection dimension {self.embedding_dimension}."
                )

    def _validate_embedding(self, embedding: Sequence[float]) -> None:
        if len(embedding) != self.embedding_dimension:
            raise EmbeddingDimensionMismatchError(
                f"Embedding dimension {len(embedding)} does not match collection "
                f"dimension {self.embedding_dimension}."
            )


def _where_clause(filters: RetrievalFilters | None) -> tuple[str, dict[str, Any]]:
    if filters is None:
        return "", {}

    clauses: list[str] = []
    params: dict[str, Any] = {}
    _add_text_in_clause(clauses, params, "paper_id", filters.paper_ids)
    _add_text_in_clause(clauses, params, "source", filters.sources)
    _add_text_in_clause(clauses, params, "language", filters.languages)
    _add_text_in_clause(clauses, params, "section", filters.sections)
    _add_text_in_clause(clauses, params, "section_group", filters.section_groups)
    _add_text_in_clause(clauses, params, "chunk_type", filters.chunk_types)
    _add_text_in_clause(clauses, params, "text_source", filters.text_sources)
    if filters.knowledge_base_ids:
        name = "knowledge_base_ids"
        params[name] = list(filters.knowledge_base_ids)
        clauses.append(
            "EXISTS ("
            "SELECT 1 FROM jsonb_array_elements_text("
            "metadata_json->'knowledge_base_ids'"
            ") AS kb(value) WHERE kb.value = ANY(:knowledge_base_ids)"
            ")"
        )
    if filters.published_from_yyyymmdd is not None:
        params["published_from_yyyymmdd"] = filters.published_from_yyyymmdd
        clauses.append(
            "CAST(metadata_json->>'published_yyyymmdd' AS integer) "
            ">= :published_from_yyyymmdd"
        )
    if filters.published_to_yyyymmdd is not None:
        params["published_to_yyyymmdd"] = filters.published_to_yyyymmdd
        clauses.append(
            "CAST(metadata_json->>'published_yyyymmdd' AS integer) "
            "<= :published_to_yyyymmdd"
        )
    if not clauses:
        return "", params
    return "WHERE " + " AND ".join(clauses), params


def _add_text_in_clause(
    clauses: list[str],
    params: dict[str, Any],
    metadata_key: str,
    values: Sequence[str],
) -> None:
    if not values:
        return
    params[metadata_key] = list(values)
    clauses.append(f"metadata_json->>'{metadata_key}' = ANY(:{metadata_key})")


def _record_from_row(row) -> VectorRecord:
    return VectorRecord(
        id=row["id"],
        document=row["document"],
        embedding=_parse_vector_literal(row["embedding_text"]),
        metadata=_json_mapping(row["metadata_json"]),
    )


def _json_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        return json.loads(value)
    return dict(value or {})


def _safe_identifier(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise VectorStoreConfigurationError(f"Unsafe SQL identifier: {value!r}")
    return value


def _vector_literal(values: Sequence[float]) -> str:
    return "[" + ",".join(str(float(value)) for value in values) + "]"


def _parse_vector_literal(value: str) -> list[float]:
    return [
        float(item)
        for item in value.strip().removeprefix("[").removesuffix("]").split(",")
        if item
    ]
