import pytest

from app.config import get_settings
from app.storage.factory import (
    StorageBackendConfigurationError,
    create_conversation_repository,
    create_paper_store,
    create_vector_store,
    storage_backend_summary,
)
from app.storage.paper_store import PaperStore
from app.vectorstores.errors import VectorStoreConfigurationError


def test_storage_factories_keep_local_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("CONVERSATION_DB_PATH", str(tmp_path / "conversations.sqlite3"))
    monkeypatch.setenv("PAPER_DB_PATH", str(tmp_path / "papers.sqlite3"))
    monkeypatch.setenv("PAPERS_DIR", str(tmp_path / "papers"))
    monkeypatch.setenv("CHROMA_PATH", str(tmp_path / "chroma"))

    conversation_repo = create_conversation_repository()
    paper_store = create_paper_store()
    vector_store = create_vector_store(embedding_model_id="fake", embedding_dimension=3)

    assert conversation_repo.health_check()["status"] == "ok"
    assert isinstance(paper_store, PaperStore)
    assert vector_store.count() == 0


def test_storage_factory_reports_backend_summary(monkeypatch):
    monkeypatch.setenv("CONVERSATION_BACKEND", "sqlite")
    monkeypatch.setenv("PAPER_STORE_BACKEND", "sqlite")
    monkeypatch.setenv("VECTOR_STORE_BACKEND", "chroma")
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@example/db")

    assert storage_backend_summary() == {
        "conversation_backend": "sqlite",
        "paper_store_backend": "sqlite",
        "vector_store_backend": "chroma",
        "database_url_configured": True,
    }


def test_storage_factories_fail_fast_for_unimplemented_cloud_backends(monkeypatch):
    monkeypatch.setenv("CONVERSATION_BACKEND", "postgres")
    with pytest.raises(StorageBackendConfigurationError, match="Postgres"):
        create_conversation_repository()

    get_settings.cache_clear()
    monkeypatch.setenv("CONVERSATION_BACKEND", "sqlite")
    monkeypatch.setenv("PAPER_STORE_BACKEND", "postgres")
    with pytest.raises(StorageBackendConfigurationError, match="Postgres"):
        create_paper_store()

    get_settings.cache_clear()
    monkeypatch.setenv("PAPER_STORE_BACKEND", "sqlite")
    monkeypatch.setenv("VECTOR_STORE_BACKEND", "pgvector")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(VectorStoreConfigurationError, match="DATABASE_URL"):
        create_vector_store()
