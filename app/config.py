from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Local runtime configuration with safe defaults."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    chroma_path: str = Field(default="data/vector_store/chroma", alias="CHROMA_PATH")
    data_dir: str = Field(default="data", alias="DATA_DIR")
    conversation_db_path: str = Field(
        default="data/metadata/conversations.sqlite3",
        alias="CONVERSATION_DB_PATH",
    )
    paper_db_path: str = Field(
        default="data/metadata/papers.sqlite3",
        alias="PAPER_DB_PATH",
    )
    papers_dir: str = Field(default="data/papers", alias="PAPERS_DIR")
    chroma_collection_name: str = Field(
        default="research_paper_chunks_v1",
        alias="CHROMA_COLLECTION_NAME",
    )
    vector_distance_metric: str = Field(default="cosine", alias="VECTOR_DISTANCE_METRIC")
    vector_upsert_batch_size: int = Field(default=64, alias="VECTOR_UPSERT_BATCH_SIZE")
    retrieval_default_top_k: int = Field(default=5, alias="RETRIEVAL_DEFAULT_TOP_K")
    retrieval_default_candidate_k: int = Field(
        default=20,
        alias="RETRIEVAL_DEFAULT_CANDIDATE_K",
    )
    retrieval_metadata_weight: float = Field(
        default=0.15,
        alias="RETRIEVAL_METADATA_WEIGHT",
    )
    metadata_schema_version: int = Field(default=1, alias="METADATA_SCHEMA_VERSION")
    bge_model_path: str | None = Field(default=None, alias="BGE_MODEL_PATH")
    bge_offline: bool = Field(default=False, alias="BGE_OFFLINE")
    bge_preload_on_startup: bool = Field(
        default=False,
        alias="BGE_PRELOAD_ON_STARTUP",
    )
    hf_home: str = Field(default="data/hf_cache", alias="HF_HOME")
    sentence_transformers_home: str = Field(
        default="data/sentence_transformers",
        alias="SENTENCE_TRANSFORMERS_HOME",
    )
    api_include_full_evidence_text: bool = Field(
        default=False,
        alias="API_INCLUDE_FULL_EVIDENCE_TEXT",
    )
    api_evidence_text_max_chars: int = Field(
        default=600,
        alias="API_EVIDENCE_TEXT_MAX_CHARS",
    )
    llm_provider: str = Field(default="langchain_openai", alias="LLM_PROVIDER")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    readiness_check_vector_store: bool = Field(
        default=False,
        alias="READINESS_CHECK_VECTOR_STORE",
    )
    conversation_recent_message_limit: int = Field(
        default=8,
        alias="CONVERSATION_RECENT_MESSAGE_LIMIT",
    )
    conversation_summary_trigger_messages: int = Field(
        default=12,
        alias="CONVERSATION_SUMMARY_TRIGGER_MESSAGES",
    )
    conversation_summary_keep_recent: int = Field(
        default=6,
        alias="CONVERSATION_SUMMARY_KEEP_RECENT",
    )


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Load runtime settings from defaults plus optional .env values."""

    return AppSettings()
