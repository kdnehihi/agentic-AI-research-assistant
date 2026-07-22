from __future__ import annotations

from typing import Any

from app.config import AppSettings, get_settings
from app.conversations.repository import ConversationRunRepository
from app.conversations.sqlite_repository import SQLiteConversationRepository
from app.storage.paper_store import PaperStore
from app.vectorstores.base import VectorStore
from app.vectorstores.chroma_store import ChromaVectorStore


LOCAL_CONVERSATION_BACKEND = "sqlite"
LOCAL_PAPER_STORE_BACKEND = "sqlite"
LOCAL_VECTOR_STORE_BACKEND = "chroma"


class StorageBackendConfigurationError(RuntimeError):
    """Raised when a configured storage backend cannot be created."""


def create_conversation_repository(
    settings: AppSettings | None = None,
) -> ConversationRunRepository:
    """Create the configured conversation and agent-run repository."""

    settings = settings or get_settings()
    backend = _normalized_backend(settings.conversation_backend)
    if backend == LOCAL_CONVERSATION_BACKEND:
        return SQLiteConversationRepository()
    if backend == "postgres":
        raise StorageBackendConfigurationError(
            "CONVERSATION_BACKEND=postgres is reserved for cloud deployment, "
            "but the Postgres conversation repository is not implemented yet."
        )
    raise StorageBackendConfigurationError(
        f"Unsupported CONVERSATION_BACKEND={settings.conversation_backend!r}. "
        "Supported values: sqlite, postgres."
    )


def create_paper_store(settings: AppSettings | None = None) -> PaperStore:
    """Create the configured paper metadata/artifact store."""

    settings = settings or get_settings()
    backend = _normalized_backend(settings.paper_store_backend)
    if backend == LOCAL_PAPER_STORE_BACKEND:
        return PaperStore()
    if backend == "postgres":
        raise StorageBackendConfigurationError(
            "PAPER_STORE_BACKEND=postgres is reserved for cloud deployment, "
            "but the Postgres paper store is not implemented yet."
        )
    raise StorageBackendConfigurationError(
        f"Unsupported PAPER_STORE_BACKEND={settings.paper_store_backend!r}. "
        "Supported values: sqlite, postgres."
    )


def create_vector_store(
    *,
    embedding_model_id: str = "BAAI/bge-small-en-v1.5",
    embedding_dimension: int = 384,
    settings: AppSettings | None = None,
) -> VectorStore:
    """Create the configured vector-store backend."""

    settings = settings or get_settings()
    backend = _normalized_backend(settings.vector_store_backend)
    if backend == LOCAL_VECTOR_STORE_BACKEND:
        return ChromaVectorStore(
            embedding_model_id=embedding_model_id,
            embedding_dimension=embedding_dimension,
        )
    if backend == "pgvector":
        from app.vectorstores.pgvector_store import PgVectorStore

        return PgVectorStore(
            embedding_model_id=embedding_model_id,
            embedding_dimension=embedding_dimension,
        )
    if backend == "qdrant":
        raise StorageBackendConfigurationError(
            "VECTOR_STORE_BACKEND=qdrant is reserved for cloud deployment, "
            "but the Qdrant vector-store adapter is not implemented yet."
        )
    raise StorageBackendConfigurationError(
        f"Unsupported VECTOR_STORE_BACKEND={settings.vector_store_backend!r}. "
        "Supported values: chroma, pgvector, qdrant."
    )


def storage_backend_summary(settings: AppSettings | None = None) -> dict[str, Any]:
    """Return a compact, non-secret storage-backend summary for readiness output."""

    settings = settings or get_settings()
    return {
        "conversation_backend": _normalized_backend(settings.conversation_backend),
        "paper_store_backend": _normalized_backend(settings.paper_store_backend),
        "vector_store_backend": _normalized_backend(settings.vector_store_backend),
        "database_url_configured": bool(settings.database_url),
    }


def _normalized_backend(value: str | None) -> str:
    return (value or "").strip().lower()
