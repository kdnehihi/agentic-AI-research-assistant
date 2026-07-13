from __future__ import annotations

from typing import Any, Protocol

from app.tools.embedding_tools import (
    DEFAULT_BGE_QUERY_INSTRUCTION,
    DEFAULT_EMBEDDING_BATCH_SIZE,
    TextEmbedder,
    _normalize_vector,
    _to_float_vectors,
)


class ExistingEmbedderInterface(Protocol):
    model_name: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, query: str) -> list[float]:
        ...


class ExistingEmbeddingAdapter:
    """Thin adapter over the existing BGE TextEmbedder implementation."""

    def __init__(
        self,
        embedder: TextEmbedder,
        model_name: str,
        batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
        normalize_embeddings: bool = True,
        query_instruction: str = DEFAULT_BGE_QUERY_INSTRUCTION,
    ):
        self.embedder = embedder
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.query_instruction = query_instruction

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raw_embeddings = self.embedder.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )
        embeddings = _to_float_vectors(raw_embeddings)
        if self.normalize_embeddings:
            return [_normalize_vector(vector) for vector in embeddings]
        return embeddings

    def embed_query(self, query: str) -> list[float]:
        query_text = (
            f"{self.query_instruction}{query.strip()}"
            if self.query_instruction
            else query.strip()
        )
        raw_embedding = self.embedder.encode(
            [query_text],
            batch_size=1,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )
        vectors = _to_float_vectors(raw_embedding)
        if not vectors:
            raise ValueError("Embedder returned no query vector.")
        if self.normalize_embeddings:
            return _normalize_vector(vectors[0])
        return vectors[0]


class DeterministicKeywordEmbedder:
    """Small fake embedder for local smoke tests and unit tests."""

    model_name = "fake-keyword-embedder"

    def __init__(self, vocabulary: tuple[str, ...]):
        self.vocabulary = vocabulary

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._embed(query)

    def _embed(self, text: str) -> list[float]:
        lowered = text.lower()
        vector = [1.0 if term in lowered else 0.0 for term in self.vocabulary]
        norm = sum(value * value for value in vector) ** 0.5
        if norm == 0:
            return vector
        return [value / norm for value in vector]
