from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Sequence

from app.config import get_settings
from app.retrieval.embedding_adapter import ExistingEmbedderInterface
from app.vectorstores.base import VectorStore
from app.vectorstores.errors import (
    EmbeddingDimensionMismatchError,
    InvalidChunkMetadataError,
)
from app.vectorstores.metadata import (
    normalize_metadata,
    published_date_to_year,
    published_date_to_yyyymmdd,
    section_group_for,
)
from app.vectorstores.models import VectorRecord


@dataclass(frozen=True)
class PaperIndexMetadata:
    paper_id: str
    title: str
    source: str
    language: str = "en"
    knowledge_base_ids: tuple[str, ...] = ("default",)
    published_date: str | None = None
    authors: tuple[str, ...] = ()
    source_url: str | None = None
    topics: tuple[str, ...] = ()
    methods: tuple[str, ...] = ()
    datasets: tuple[str, ...] = ()
    tasks: tuple[str, ...] = ()
    models: tuple[str, ...] = ()
    evaluation_metrics: tuple[str, ...] = ()
    text_source: str = "clean_text"
    chunk_type: str = "paper_section_chunk"
    extraction_quality: str | None = None


@dataclass(frozen=True)
class ChunkIndexingResult:
    attempted: int
    upserted: int
    skipped: int
    failed: int
    errors: list[str]


def index_chunks(
    chunks: Sequence[Any],
    paper_metadata: PaperIndexMetadata,
    embedder: ExistingEmbedderInterface,
    vector_store: VectorStore,
    *,
    batch_size: int | None = None,
    metadata_schema_version: int | None = None,
) -> ChunkIndexingResult:
    settings = get_settings()
    resolved_batch_size = batch_size or settings.vector_upsert_batch_size
    schema_version = metadata_schema_version or settings.metadata_schema_version

    if not chunks:
        return ChunkIndexingResult(
            attempted=0,
            upserted=0,
            skipped=0,
            failed=0,
            errors=[],
        )

    attempted = len(chunks)
    upserted = 0
    failed = 0
    errors: list[str] = []

    for start in range(0, len(chunks), resolved_batch_size):
        batch = list(chunks[start:start + resolved_batch_size])
        try:
            records = _records_for_batch(
                chunks=batch,
                paper_metadata=paper_metadata,
                embedder=embedder,
                metadata_schema_version=schema_version,
            )
            result = vector_store.upsert_records(
                records,
                batch_size=resolved_batch_size,
            )
            upserted += result.upserted
            failed += result.failed
        except Exception as exc:
            failed += len(batch)
            errors.append(str(exc))

    return ChunkIndexingResult(
        attempted=attempted,
        upserted=upserted,
        skipped=0,
        failed=failed,
        errors=errors,
    )


def build_vector_records(
    chunks: Sequence[Any],
    paper_metadata: PaperIndexMetadata,
    embeddings: Sequence[Sequence[float]],
    embedding_model_id: str,
    metadata_schema_version: int,
) -> list[VectorRecord]:
    if len(chunks) != len(embeddings):
        raise EmbeddingDimensionMismatchError(
            f"Got {len(chunks)} chunks but {len(embeddings)} embeddings."
        )

    records: list[VectorRecord] = []
    for chunk, embedding in zip(chunks, embeddings):
        chunk_data = _chunk_to_dict(chunk)
        document = str(chunk_data.get("text", "")).strip()
        if not document:
            raise InvalidChunkMetadataError("Chunk text must not be empty.")
        metadata = _metadata_for_chunk(
            chunk=chunk_data,
            paper_metadata=paper_metadata,
            embedding_model_id=embedding_model_id,
            embedding_dimension=len(embedding),
            metadata_schema_version=metadata_schema_version,
        )
        records.append(
            VectorRecord(
                id=str(chunk_data["chunk_id"]),
                document=document,
                embedding=[float(value) for value in embedding],
                metadata=metadata,
            )
        )

    return records


def _records_for_batch(
    chunks: Sequence[Any],
    paper_metadata: PaperIndexMetadata,
    embedder: ExistingEmbedderInterface,
    metadata_schema_version: int,
) -> list[VectorRecord]:
    embedding_texts = [
        _embedding_text(_chunk_to_dict(chunk), paper_metadata)
        for chunk in chunks
    ]
    embeddings = embedder.embed_documents(embedding_texts)
    return build_vector_records(
        chunks=chunks,
        paper_metadata=paper_metadata,
        embeddings=embeddings,
        embedding_model_id=embedder.model_name,
        metadata_schema_version=metadata_schema_version,
    )


def _metadata_for_chunk(
    chunk: dict[str, Any],
    paper_metadata: PaperIndexMetadata,
    embedding_model_id: str,
    embedding_dimension: int,
    metadata_schema_version: int,
) -> dict[str, Any]:
    section = str(chunk.get("section", ""))
    metadata = {
        "paper_id": paper_metadata.paper_id,
        "knowledge_base_ids": list(paper_metadata.knowledge_base_ids),
        "source": paper_metadata.source,
        "language": paper_metadata.language,
        "title": paper_metadata.title,
        "published_year": published_date_to_year(paper_metadata.published_date),
        "published_yyyymmdd": published_date_to_yyyymmdd(paper_metadata.published_date),
        "section": section,
        "section_group": section_group_for(section),
        "chunk_type": paper_metadata.chunk_type,
        "chunk_index": int(chunk["chunk_index"]),
        "word_count": int(chunk["word_count"]),
        "text_source": paper_metadata.text_source,
        "metadata_schema_version": metadata_schema_version,
        "embedding_model_id": embedding_model_id,
        "embedding_dimension": embedding_dimension,
        "authors": list(paper_metadata.authors),
        "source_url": paper_metadata.source_url,
        "extraction_quality": paper_metadata.extraction_quality,
        "topics": list(paper_metadata.topics),
        "methods": list(paper_metadata.methods),
        "datasets": list(paper_metadata.datasets),
        "tasks": list(paper_metadata.tasks),
        "models": list(paper_metadata.models),
        "evaluation_metrics": list(paper_metadata.evaluation_metrics),
    }
    return normalize_metadata(metadata)


def _embedding_text(chunk: dict[str, Any], paper_metadata: PaperIndexMetadata) -> str:
    return (
        f"Title: {paper_metadata.title}\n"
        f"Section: {chunk.get('section', '')}\n"
        "Content:\n"
        f"{chunk.get('text', '')}"
    )


def _chunk_to_dict(chunk: Any) -> dict[str, Any]:
    if isinstance(chunk, dict):
        return dict(chunk)
    if is_dataclass(chunk):
        return asdict(chunk)
    raise InvalidChunkMetadataError(
        f"Unsupported chunk type for indexing: {type(chunk).__name__}"
    )
