import pytest

from app.config import get_settings
from app.conversations.postgres_repository import PostgresConversationRepository
from app.storage.factory import (
    StorageBackendConfigurationError,
    create_conversation_repository,
    create_paper_store,
    create_vector_store,
    storage_backend_summary,
)
from app.storage.paper_store import PaperStore
from app.storage.postgres_paper_store import PostgresPaperStore
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


def test_storage_factories_require_database_url_for_cloud_backends(monkeypatch):
    monkeypatch.setenv("CONVERSATION_BACKEND", "postgres")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(StorageBackendConfigurationError, match="DATABASE_URL"):
        create_conversation_repository()

    get_settings.cache_clear()
    monkeypatch.setenv("CONVERSATION_BACKEND", "sqlite")
    monkeypatch.setenv("PAPER_STORE_BACKEND", "postgres")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(StorageBackendConfigurationError, match="DATABASE_URL"):
        create_paper_store()

    get_settings.cache_clear()
    monkeypatch.setenv("PAPER_STORE_BACKEND", "sqlite")
    monkeypatch.setenv("VECTOR_STORE_BACKEND", "pgvector")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(VectorStoreConfigurationError, match="DATABASE_URL"):
        create_vector_store()


def test_storage_factories_route_to_postgres_backends(monkeypatch):
    created = {}

    class FakeConversationRepo:
        def __init__(self):
            created["conversation"] = True

    class FakePaperStore:
        def __init__(self):
            created["paper"] = True

    import app.conversations.postgres_repository as conversation_module
    import app.storage.factory as factory_module

    monkeypatch.setenv("CONVERSATION_BACKEND", "postgres")
    monkeypatch.setenv("PAPER_STORE_BACKEND", "postgres")
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://user:pass@host/db")
    monkeypatch.setattr(
        conversation_module,
        "PostgresConversationRepository",
        FakeConversationRepo,
    )
    monkeypatch.setattr(factory_module, "PostgresPaperStore", FakePaperStore)

    assert isinstance(create_conversation_repository(), FakeConversationRepo)
    assert isinstance(create_paper_store(), FakePaperStore)
    assert created == {"conversation": True, "paper": True}


def test_postgres_adapters_can_be_constructed_without_initializing_db(tmp_path):
    fake_engine = object()
    conversation_repo = PostgresConversationRepository(
        engine=fake_engine,
        initialize=False,
    )
    paper_store = PostgresPaperStore(
        engine=fake_engine,
        papers_dir=tmp_path / "papers",
        initialize=False,
    )

    assert conversation_repo.engine is fake_engine
    assert paper_store.engine is fake_engine
    assert paper_store.papers_dir == tmp_path / "papers"
