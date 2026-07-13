from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from app.config import get_settings
from app.retrieval.models import RetrievalFilters
from app.vectorstores.errors import (
    EmbeddingDimensionMismatchError,
    VectorStoreConfigurationError,
    VectorStoreError,
)
from app.vectorstores.metadata import (
    build_chroma_runtime_where,
    from_chroma_metadata,
    to_chroma_metadata,
    validate_required_metadata,
)
from app.vectorstores.models import UpsertResult, VectorRecord, VectorSearchResult


class ChromaVectorStore:
    def __init__(
        self,
        persist_path: str | Path | None = None,
        collection_name: str | None = None,
        embedding_model_id: str = "BAAI/bge-small-en-v1.5",
        embedding_dimension: int = 384,
        distance_metric: str | None = None,
        metadata_schema_version: int | None = None,
    ):
        settings = get_settings()
        self.persist_path = Path(persist_path or settings.chroma_path)
        self.collection_name = collection_name or settings.chroma_collection_name
        self.embedding_model_id = embedding_model_id
        self.embedding_dimension = embedding_dimension
        self.distance_metric = distance_metric or settings.vector_distance_metric
        self.metadata_schema_version = (
            metadata_schema_version or settings.metadata_schema_version
        )
        self.persist_path.mkdir(parents=True, exist_ok=True)

        try:
            import chromadb
        except ImportError as exc:
            raise VectorStoreConfigurationError(
                "chromadb is required for ChromaVectorStore. "
                "Install it with `pip install chromadb`."
            ) from exc

        self.client = chromadb.PersistentClient(path=str(self.persist_path))
        self.collection = self._get_or_create_collection()
        self._validate_collection_metadata()

    def upsert_records(
        self,
        records: Sequence[VectorRecord],
        *,
        batch_size: int = 64,
    ) -> UpsertResult:
        if not records:
            return UpsertResult(attempted=0, upserted=0)

        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        attempted = len(records)
        try:
            for start in range(0, len(records), batch_size):
                batch = list(records[start:start + batch_size])
                self._validate_records(batch)
                self.collection.upsert(
                    ids=[record.id for record in batch],
                    documents=[record.document for record in batch],
                    embeddings=[record.embedding for record in batch],
                    metadatas=[
                        to_chroma_metadata(record.metadata)
                        for record in batch
                    ],
                )
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError("Chroma upsert failed.") from exc

        return UpsertResult(attempted=attempted, upserted=attempted)

    def search(
        self,
        *,
        query_embedding: Sequence[float],
        top_k: int,
        filters: RetrievalFilters | None = None,
        include_embeddings: bool = False,
    ) -> list[VectorSearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        self._validate_embedding(query_embedding)

        if self.count() == 0:
            return []

        include = ["documents", "metadatas", "distances"]
        if include_embeddings:
            include.append("embeddings")

        try:
            response = self.collection.query(
                query_embeddings=[list(query_embedding)],
                n_results=top_k,
                where=build_chroma_runtime_where(filters),
                include=include,
            )
        except Exception as exc:
            raise VectorStoreError("Chroma search failed.") from exc

        return self._search_results_from_response(response, include_embeddings)

    def get_by_ids(self, ids: Sequence[str]) -> list[VectorRecord]:
        if not ids:
            return []

        try:
            response = self.collection.get(
                ids=list(ids),
                include=["documents", "metadatas", "embeddings"],
            )
        except Exception as exc:
            raise VectorStoreError("Chroma get_by_ids failed.") from exc

        return self._records_from_get_response(response)

    def get_by_paper(self, paper_id: str) -> list[VectorRecord]:
        try:
            response = self.collection.get(
                where={"paper_id": paper_id},
                include=["documents", "metadatas", "embeddings"],
            )
        except Exception as exc:
            raise VectorStoreError("Chroma get_by_paper failed.") from exc

        return self._records_from_get_response(response)

    def delete_by_ids(self, ids: Sequence[str]) -> int:
        if not ids:
            return 0
        existing_count = len(self.get_by_ids(ids))
        try:
            self.collection.delete(ids=list(ids))
        except Exception as exc:
            raise VectorStoreError("Chroma delete_by_ids failed.") from exc
        return existing_count

    def delete_by_paper(self, paper_id: str) -> int:
        existing_count = len(self.get_by_paper(paper_id))
        if existing_count == 0:
            return 0
        try:
            self.collection.delete(where={"paper_id": paper_id})
        except Exception as exc:
            raise VectorStoreError("Chroma delete_by_paper failed.") from exc
        return existing_count

    def count(self) -> int:
        try:
            return int(self.collection.count())
        except Exception as exc:
            raise VectorStoreError("Chroma count failed.") from exc

    def _get_or_create_collection(self) -> Any:
        collection_metadata = {
            "hnsw:space": self.distance_metric,
            "distance_metric": self.distance_metric,
            "embedding_model_id": self.embedding_model_id,
            "embedding_dimension": self.embedding_dimension,
            "metadata_schema_version": self.metadata_schema_version,
        }
        try:
            return self.client.get_or_create_collection(
                name=self.collection_name,
                metadata=collection_metadata,
                embedding_function=None,
            )
        except Exception as exc:
            raise VectorStoreConfigurationError(
                f"Could not open Chroma collection '{self.collection_name}'."
            ) from exc

    def _validate_collection_metadata(self) -> None:
        metadata = self.collection.metadata or {}
        expected = {
            "distance_metric": self.distance_metric,
            "embedding_model_id": self.embedding_model_id,
            "embedding_dimension": self.embedding_dimension,
            "metadata_schema_version": self.metadata_schema_version,
        }
        mismatches = [
            f"{key}: existing={metadata.get(key)!r}, expected={value!r}"
            for key, value in expected.items()
            if metadata.get(key) != value
        ]
        if mismatches:
            raise VectorStoreConfigurationError(
                "Existing Chroma collection is incompatible with current vector "
                "configuration. Use a new collection name/path or delete the "
                "test collection intentionally. Mismatches: "
                + "; ".join(mismatches)
            )

    def _validate_records(self, records: Sequence[VectorRecord]) -> None:
        ids = [record.id for record in records]
        documents = [record.document for record in records]
        embeddings = [record.embedding for record in records]
        metadatas = [record.metadata for record in records]
        lengths = {len(ids), len(documents), len(embeddings), len(metadatas)}
        if len(lengths) != 1:
            raise VectorStoreError("Vector record field lengths are misaligned.")

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

    def _search_results_from_response(
        self,
        response: dict[str, Any],
        include_embeddings: bool,
    ) -> list[VectorSearchResult]:
        ids = (response.get("ids") or [[]])[0]
        documents = (response.get("documents") or [[]])[0]
        metadatas = (response.get("metadatas") or [[]])[0]
        distances = (response.get("distances") or [[]])[0]
        raw_embeddings = response.get("embeddings")
        embeddings = raw_embeddings[0] if include_embeddings and raw_embeddings is not None else []

        results: list[VectorSearchResult] = []
        for index, record_id in enumerate(ids):
            results.append(
                VectorSearchResult(
                    id=record_id,
                    document=documents[index],
                    metadata=from_chroma_metadata(metadatas[index] or {}),
                    distance=float(distances[index]),
                    embedding=list(embeddings[index]) if include_embeddings else None,
                )
            )
        return results

    def _records_from_get_response(self, response: dict[str, Any]) -> list[VectorRecord]:
        ids = response.get("ids") or []
        documents = response.get("documents") or []
        metadatas = response.get("metadatas") or []
        embeddings = response.get("embeddings")
        if embeddings is None:
            embeddings = []
        records: list[VectorRecord] = []
        for index, record_id in enumerate(ids):
            records.append(
                VectorRecord(
                    id=record_id,
                    document=documents[index],
                    embedding=[float(value) for value in embeddings[index]],
                    metadata=from_chroma_metadata(metadatas[index] or {}),
                )
            )
        return records
