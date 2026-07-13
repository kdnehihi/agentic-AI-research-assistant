from __future__ import annotations


class VectorStoreError(Exception):
    """Base error for vector-store operations."""


class VectorStoreConfigurationError(VectorStoreError):
    """Raised when a vector collection is incompatible with current settings."""


class EmbeddingDimensionMismatchError(VectorStoreError):
    """Raised when record or query embeddings do not match collection dimension."""


class InvalidChunkMetadataError(VectorStoreError):
    """Raised when chunk metadata is missing or malformed."""


class RetrievalError(Exception):
    """Raised when retrieval cannot be completed."""
