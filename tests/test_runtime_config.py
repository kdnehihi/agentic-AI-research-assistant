from app.conversations.sqlite_repository import SQLiteConversationRepository
from app.storage.paper_store import PaperStore


def test_repositories_use_configured_storage_paths(tmp_path, monkeypatch):
    conversation_db = tmp_path / "runtime" / "conversations.sqlite3"
    paper_db = tmp_path / "runtime" / "papers.sqlite3"
    papers_dir = tmp_path / "runtime" / "papers"
    monkeypatch.setenv("CONVERSATION_DB_PATH", str(conversation_db))
    monkeypatch.setenv("PAPER_DB_PATH", str(paper_db))
    monkeypatch.setenv("PAPERS_DIR", str(papers_dir))

    conversation_repo = SQLiteConversationRepository()
    paper_store = PaperStore()

    assert conversation_repo.db_path == conversation_db
    assert paper_store.db_path == paper_db
    assert paper_store.papers_dir == papers_dir
    assert conversation_repo.health_check()["status"] == "ok"


def test_runtime_settings_include_model_cache_and_api_payload_options(
    tmp_path,
    monkeypatch,
):
    hf_home = tmp_path / "hf"
    st_home = tmp_path / "sentence-transformers"
    monkeypatch.setenv("HF_HOME", str(hf_home))
    monkeypatch.setenv("SENTENCE_TRANSFORMERS_HOME", str(st_home))
    monkeypatch.setenv("BGE_PRELOAD_ON_STARTUP", "true")
    monkeypatch.setenv("API_INCLUDE_FULL_EVIDENCE_TEXT", "true")
    monkeypatch.setenv("API_EVIDENCE_TEXT_MAX_CHARS", "123")

    from app.config import get_settings

    settings = get_settings()

    assert settings.hf_home == str(hf_home)
    assert settings.sentence_transformers_home == str(st_home)
    assert settings.bge_preload_on_startup is True
    assert settings.api_include_full_evidence_text is True
    assert settings.api_evidence_text_max_chars == 123


def test_runtime_settings_include_storage_backend_options(monkeypatch):
    monkeypatch.setenv("CONVERSATION_BACKEND", "postgres")
    monkeypatch.setenv("PAPER_STORE_BACKEND", "postgres")
    monkeypatch.setenv("VECTOR_STORE_BACKEND", "pgvector")
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://user:pass@host/db")

    from app.config import get_settings

    settings = get_settings()

    assert settings.conversation_backend == "postgres"
    assert settings.paper_store_backend == "postgres"
    assert settings.vector_store_backend == "pgvector"
    assert settings.database_url == "postgresql+psycopg://user:pass@host/db"
