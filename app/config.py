from __future__ import annotations

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


def get_settings() -> AppSettings:
    return AppSettings()
