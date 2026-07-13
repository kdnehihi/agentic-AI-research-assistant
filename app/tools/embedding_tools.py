from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from app.agent.state import AgentState
from app.storage.paper_store import PaperStore


DEFAULT_BGE_MODEL_NAME = "BAAI/bge-small-en-v1.5"
DEFAULT_EMBEDDING_BATCH_SIZE = 16
DEFAULT_BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class TextEmbedder(Protocol):
    """Protocol shared by sentence-transformers and fallback embedders."""

    def encode(
        self,
        texts: list[str],
        batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> Any:
        """Encode a batch of texts into dense vectors."""

        ...


class TransformersBgeEmbedder:
    """Transformers-based BGE fallback when sentence-transformers cannot load."""

    def __init__(self, model_name: str):
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers and torch are required for the BGE fallback embedder."
            ) from exc

        self.model_name = model_name
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def encode(
        self,
        texts: list[str],
        batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> list[list[float]]:
        """Encode texts with mean pooling over transformer token embeddings."""

        del show_progress_bar

        vectors: list[list[float]] = []
        with self.torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = texts[start:start + batch_size]
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                output = self.model(**encoded)
                pooled = self._mean_pool(
                    token_embeddings=output.last_hidden_state,
                    attention_mask=encoded["attention_mask"],
                )
                if normalize_embeddings:
                    pooled = self.torch.nn.functional.normalize(pooled, p=2, dim=1)
                vectors.extend(pooled.cpu().tolist())

        return vectors

    def _mean_pool(self, token_embeddings: Any, attention_mask: Any) -> Any:
        """Mean-pool token embeddings while ignoring padding tokens."""

        expanded_mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = (token_embeddings * expanded_mask).sum(dim=1)
        counts = expanded_mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts


@dataclass(frozen=True)
class EmbeddedChunk:
    """One chunk plus its embedding vector and chunk metadata."""

    chunk_id: str
    paper_id: str
    section: str
    section_index: int
    chunk_index: int
    section_chunk_index: int
    start_word: int
    end_word: int
    word_count: int
    embedding_model: str
    embedding_dim: int
    embedding: list[float]
    text: str


@dataclass(frozen=True)
class ChunkSearchResult:
    """Search result returned by local JSONL embedding search."""

    score: float
    chunk_id: str
    paper_id: str
    section: str
    section_index: int
    chunk_index: int
    section_chunk_index: int
    start_word: int
    end_word: int
    word_count: int
    text: str


def embed_selected_paper_chunks(
    state: AgentState,
    file_store: PaperStore | None = None,
    embedder: TextEmbedder | None = None,
    model_name: str = DEFAULT_BGE_MODEL_NAME,
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    normalize_embeddings: bool = True,
) -> dict[str, Any]:
    """
    Embed selected paper chunks with a BAAI BGE model and save embeddings.jsonl.
    """
    # Agent entrypoint: use chunk paths from state when present, otherwise fall
    # back to the conventional PaperStore chunks.jsonl location.
    file_store = file_store or PaperStore()

    if not state.selected_papers:
        return {
            "status": "skipped",
            "processed": 0,
            "failed": 0,
            "embedded_chunks": 0,
            "summary": "No selected papers available for embedding.",
        }

    embedder = embedder or load_bge_embedder(model_name=model_name)
    paper_embedding_paths = dict(state.paper_embedding_paths)
    processed = 0
    failed = 0
    total_embedded_chunks = 0
    errors: list[dict[str, str]] = []

    for paper in state.selected_papers:
        paper_id = paper.paper_id
        if not paper_id:
            failed += 1
            errors.append(
                {
                    "paper_id": "",
                    "title": paper.title,
                    "error": "Paper is missing paper_id.",
                }
            )
            continue

        try:
            chunks_path = _chunks_path_for_paper(
                state=state,
                file_store=file_store,
                paper_id=paper_id,
            )
            chunks = load_chunks_jsonl(chunks_path)
            embedded_chunks = embed_chunks(
                chunks=chunks,
                embedder=embedder,
                model_name=model_name,
                batch_size=batch_size,
                normalize_embeddings=normalize_embeddings,
            )
            embeddings_path = save_embeddings_jsonl(
                embedded_chunks=embedded_chunks,
                path=file_store.embeddings_path(paper_id),
            )

            paper_embedding_paths[paper_id] = str(embeddings_path)
            processed += 1
            total_embedded_chunks += len(embedded_chunks)
        except Exception as exc:
            failed += 1
            errors.append(
                {
                    "paper_id": paper_id,
                    "title": paper.title,
                    "error": str(exc),
                }
            )

    state.set_paper_embedding_paths(paper_embedding_paths)

    if failed == 0:
        status = "success"
    elif processed > 0:
        status = "partial_success"
    else:
        status = "failed"

    return {
        "status": status,
        "processed": processed,
        "failed": failed,
        "embedded_chunks": total_embedded_chunks,
        "errors": errors,
        "model_name": model_name,
        "summary": (
            f"Embedded {total_embedded_chunks} chunks from {processed} papers "
            f"using {model_name}. Failed: {failed}."
        ),
    }


def load_bge_embedder(model_name: str = DEFAULT_BGE_MODEL_NAME) -> TextEmbedder:
    """Load a BGE embedder, preferring sentence-transformers when available."""

    # Keep sentence-transformers optional so tests and non-embedding workflows do
    # not need to download a model.
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for real BGE embeddings. "
            "Install it with `pip install sentence-transformers`."
        ) from exc

    try:
        return SentenceTransformer(model_name)
    except Exception:
        return TransformersBgeEmbedder(model_name=model_name)


def load_chunks_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load and validate section chunks from a JSONL file."""

    chunk_path = Path(path)
    if not chunk_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunk_path}")

    chunks: list[dict[str, Any]] = []
    for line_number, line in enumerate(chunk_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue

        row = json.loads(line)
        if not row.get("text"):
            raise ValueError(f"Chunk at line {line_number} is missing text.")
        chunks.append(row)

    if not chunks:
        raise ValueError(f"No chunks found in {chunk_path}")

    return chunks


def load_embeddings_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load and validate embedded chunks from a JSONL file."""

    embeddings_path = Path(path)
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        embeddings_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        row = json.loads(line)
        if not row.get("embedding"):
            raise ValueError(f"Embedding at line {line_number} is missing vector.")
        if not row.get("text"):
            raise ValueError(f"Embedding at line {line_number} is missing text.")
        rows.append(row)

    if not rows:
        raise ValueError(f"No embeddings found in {embeddings_path}")

    return rows


def embed_chunks(
    chunks: list[dict[str, Any]],
    embedder: TextEmbedder,
    model_name: str = DEFAULT_BGE_MODEL_NAME,
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    normalize_embeddings: bool = True,
) -> list[EmbeddedChunk]:
    """Embed chunk text and return structured EmbeddedChunk records."""

    # BGE retrieval works well with normalized embeddings because dot product
    # then behaves like cosine similarity.
    texts = [str(chunk["text"]) for chunk in chunks]
    raw_embeddings = embedder.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
        show_progress_bar=False,
    )
    embeddings = _to_float_vectors(raw_embeddings)

    if len(embeddings) != len(chunks):
        raise ValueError(
            f"Embedder returned {len(embeddings)} vectors for {len(chunks)} chunks."
        )

    if normalize_embeddings:
        embeddings = [_normalize_vector(vector) for vector in embeddings]

    return [
        _build_embedded_chunk(
            chunk=chunk,
            embedding=embedding,
            model_name=model_name,
        )
        for chunk, embedding in zip(chunks, embeddings)
    ]


def search_embedded_chunks(
    query: str,
    embeddings: list[dict[str, Any]],
    embedder: TextEmbedder,
    top_k: int = 5,
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    normalize_embeddings: bool = True,
    query_instruction: str = DEFAULT_BGE_QUERY_INSTRUCTION,
    excluded_sections: tuple[str, ...] = ("Front Matter",),
) -> list[ChunkSearchResult]:
    """
    Search embedded chunks for a query using BGE-style dense retrieval.
    """
    # Prefixing the query follows the common BGE retrieval recipe. It is applied
    # only to the query, while stored chunk text remains the passage side.
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    if not query.strip():
        raise ValueError("query must not be empty.")

    query_text = f"{query_instruction}{query.strip()}" if query_instruction else query.strip()
    raw_query_embedding = embedder.encode(
        [query_text],
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
        show_progress_bar=False,
    )
    query_vectors = _to_float_vectors(raw_query_embedding)
    if not query_vectors:
        raise ValueError("Embedder returned no query vector.")

    query_vector = query_vectors[0]
    if normalize_embeddings:
        query_vector = _normalize_vector(query_vector)

    searchable_embeddings = [
        row
        for row in embeddings
        if str(row.get("section", "")) not in excluded_sections
    ]
    if not searchable_embeddings:
        raise ValueError("No embeddings remain after section filtering.")

    scored_rows = [
        (
            _dot_product(query_vector, _embedding_vector(row, normalize=normalize_embeddings)),
            row,
        )
        for row in searchable_embeddings
    ]
    scored_rows.sort(key=lambda item: item[0], reverse=True)

    return [
        _build_search_result(score=score, row=row)
        for score, row in scored_rows[:top_k]
    ]


def save_embeddings_jsonl(
    embedded_chunks: list[EmbeddedChunk],
    path: str | Path,
) -> Path:
    """Persist embedded chunks as JSON Lines."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for embedded_chunk in embedded_chunks:
            file.write(json.dumps(asdict(embedded_chunk), ensure_ascii=False) + "\n")

    return output_path


def _build_search_result(score: float, row: dict[str, Any]) -> ChunkSearchResult:
    """Convert a scored embedding row into a search result object."""

    return ChunkSearchResult(
        score=float(score),
        chunk_id=str(row["chunk_id"]),
        paper_id=str(row["paper_id"]),
        section=str(row["section"]),
        section_index=int(row["section_index"]),
        chunk_index=int(row["chunk_index"]),
        section_chunk_index=int(row["section_chunk_index"]),
        start_word=int(row["start_word"]),
        end_word=int(row["end_word"]),
        word_count=int(row["word_count"]),
        text=str(row["text"]),
    )


def _build_embedded_chunk(
    chunk: dict[str, Any],
    embedding: list[float],
    model_name: str,
) -> EmbeddedChunk:
    """Combine one raw chunk and embedding vector into an EmbeddedChunk."""

    return EmbeddedChunk(
        chunk_id=str(chunk["chunk_id"]),
        paper_id=str(chunk["paper_id"]),
        section=str(chunk["section"]),
        section_index=int(chunk["section_index"]),
        chunk_index=int(chunk["chunk_index"]),
        section_chunk_index=int(chunk["section_chunk_index"]),
        start_word=int(chunk["start_word"]),
        end_word=int(chunk["end_word"]),
        word_count=int(chunk["word_count"]),
        embedding_model=model_name,
        embedding_dim=len(embedding),
        embedding=embedding,
        text=str(chunk["text"]),
    )


def _chunks_path_for_paper(
    state: AgentState,
    file_store: PaperStore,
    paper_id: str,
) -> Path:
    """Resolve a paper chunk path from state or the conventional store path."""

    if paper_id in state.paper_chunk_paths:
        return Path(state.paper_chunk_paths[paper_id])

    return file_store.chunks_path(paper_id)


def _to_float_vectors(raw_embeddings: Any) -> list[list[float]]:
    """Convert numpy/torch/list embedding outputs to nested float lists."""

    if hasattr(raw_embeddings, "tolist"):
        raw_embeddings = raw_embeddings.tolist()

    return [
        [float(value) for value in vector]
        for vector in raw_embeddings
    ]


def _normalize_vector(vector: list[float]) -> list[float]:
    """L2-normalize one vector, leaving zero vectors unchanged."""

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector

    return [value / norm for value in vector]


def _embedding_vector(row: dict[str, Any], normalize: bool) -> list[float]:
    """Load and optionally normalize one embedding vector from a JSON row."""

    vector = [float(value) for value in row["embedding"]]
    if normalize:
        return _normalize_vector(vector)
    return vector


def _dot_product(left: list[float], right: list[float]) -> float:
    """Compute dot product after validating both vectors have equal length."""

    if len(left) != len(right):
        raise ValueError(
            f"Embedding dimension mismatch: query={len(left)} chunk={len(right)}"
        )

    return sum(left_value * right_value for left_value, right_value in zip(left, right))
