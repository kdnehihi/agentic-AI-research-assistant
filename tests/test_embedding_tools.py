import json
import math

import pytest

from app.agent.state import AgentState, Paper
from app.storage.paper_store import PaperStore
from app.tools.chunking_tools import Section, chunk_section, save_chunks_jsonl
from app.tools.embedding_tools import (
    DEFAULT_BGE_MODEL_NAME,
    embed_chunks,
    embed_selected_paper_chunks,
    load_embeddings_jsonl,
    load_chunks_jsonl,
    resolve_bge_model_source,
    search_embedded_chunks,
    save_embeddings_jsonl,
)


class FakeBgeEmbedder:
    def encode(
        self,
        texts,
        batch_size=16,
        normalize_embeddings=True,
        show_progress_bar=False,
    ):
        return [
            [float(index + 1), float(len(text.split())), 1.0]
            for index, text in enumerate(texts)
        ]


class KeywordBgeEmbedder:
    def encode(
        self,
        texts,
        batch_size=16,
        normalize_embeddings=True,
        show_progress_bar=False,
    ):
        return [
            [
                float("retrieval" in text.lower() or "rag" in text.lower()),
                float("diversity" in text.lower()),
                float("evaluation" in text.lower() or "benchmark" in text.lower()),
            ]
            for text in texts
        ]


def test_resolve_bge_model_source_uses_local_path(monkeypatch, tmp_path):
    model_dir = tmp_path / "bge"
    model_dir.mkdir()
    monkeypatch.setenv("BGE_MODEL_PATH", str(model_dir))
    monkeypatch.delenv("BGE_OFFLINE", raising=False)

    source, local_only = resolve_bge_model_source(DEFAULT_BGE_MODEL_NAME)

    assert source == str(model_dir)
    assert local_only is True


def test_resolve_bge_model_source_requires_path_when_offline(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("BGE_MODEL_PATH", raising=False)
    monkeypatch.setenv("BGE_OFFLINE", "true")

    with pytest.raises(FileNotFoundError):
        resolve_bge_model_source(DEFAULT_BGE_MODEL_NAME)


def test_resolve_bge_model_source_auto_detects_default_local_model(
    monkeypatch,
    tmp_path,
):
    model_dir = tmp_path / "data" / "models" / "bge-small-en-v1.5"
    model_dir.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("BGE_MODEL_PATH", raising=False)
    monkeypatch.delenv("BGE_OFFLINE", raising=False)

    source, local_only = resolve_bge_model_source(DEFAULT_BGE_MODEL_NAME)

    assert source == str(model_dir.relative_to(tmp_path))
    assert local_only is True


def test_resolve_bge_model_source_uses_remote_name_by_default(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("BGE_MODEL_PATH", raising=False)
    monkeypatch.delenv("BGE_OFFLINE", raising=False)

    source, local_only = resolve_bge_model_source(DEFAULT_BGE_MODEL_NAME)

    assert source == DEFAULT_BGE_MODEL_NAME
    assert local_only is False


def test_embed_chunks_keeps_chunk_metadata_and_normalizes_vectors():
    chunks = [
        {
            "chunk_id": "arxiv:test::chunk:0",
            "paper_id": "arxiv:test",
            "section": "Introduction",
            "section_index": 1,
            "chunk_index": 0,
            "section_chunk_index": 0,
            "start_word": 0,
            "end_word": 4,
            "word_count": 4,
            "text": "retrieval augmented generation test",
        }
    ]

    embedded_chunks = embed_chunks(
        chunks=chunks,
        embedder=FakeBgeEmbedder(),
        model_name=DEFAULT_BGE_MODEL_NAME,
    )

    embedded_chunk = embedded_chunks[0]
    vector_norm = math.sqrt(sum(value * value for value in embedded_chunk.embedding))

    assert embedded_chunk.chunk_id == "arxiv:test::chunk:0"
    assert embedded_chunk.paper_id == "arxiv:test"
    assert embedded_chunk.section == "Introduction"
    assert embedded_chunk.embedding_model == DEFAULT_BGE_MODEL_NAME
    assert embedded_chunk.embedding_dim == 3
    assert vector_norm == pytest.approx(1.0)


def test_save_embeddings_jsonl_writes_vectors_with_metadata(tmp_path):
    chunks = [
        {
            "chunk_id": "arxiv:test::chunk:0",
            "paper_id": "arxiv:test",
            "section": "Abstract",
            "section_index": 0,
            "chunk_index": 0,
            "section_chunk_index": 0,
            "start_word": 0,
            "end_word": 3,
            "word_count": 3,
            "text": "one two three",
        }
    ]
    embedded_chunks = embed_chunks(chunks=chunks, embedder=FakeBgeEmbedder())

    output_path = save_embeddings_jsonl(
        embedded_chunks=embedded_chunks,
        path=tmp_path / "embeddings.jsonl",
    )
    rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]

    assert len(rows) == 1
    assert rows[0]["chunk_id"] == "arxiv:test::chunk:0"
    assert rows[0]["embedding_dim"] == 3
    assert len(rows[0]["embedding"]) == 3
    assert rows[0]["text"] == "one two three"


def test_load_chunks_jsonl_reads_chunk_rows(tmp_path):
    chunks = chunk_section(
        section=Section(title="Abstract", text=" ".join(f"word{i}" for i in range(20))),
        paper_id="arxiv:load",
        min_chunk_words=10,
        target_chunk_words=10,
        max_chunk_words=12,
        overlap_words=2,
    )
    chunks_path = save_chunks_jsonl(chunks, tmp_path / "chunks.jsonl")

    rows = load_chunks_jsonl(chunks_path)

    assert len(rows) == len(chunks)
    assert rows[0]["paper_id"] == "arxiv:load"
    assert rows[0]["section"] == "Abstract"


def test_search_embedded_chunks_ranks_relevant_chunk_first(tmp_path):
    chunks = [
        {
            "chunk_id": "arxiv:test::chunk:0",
            "paper_id": "arxiv:test",
            "section": "Methodology",
            "section_index": 1,
            "chunk_index": 0,
            "section_chunk_index": 0,
            "start_word": 0,
            "end_word": 8,
            "word_count": 8,
            "text": "query aware diversity for retrieval augmented generation",
        },
        {
            "chunk_id": "arxiv:test::chunk:1",
            "paper_id": "arxiv:test",
            "section": "Related Work",
            "section_index": 2,
            "chunk_index": 1,
            "section_chunk_index": 0,
            "start_word": 0,
            "end_word": 8,
            "word_count": 8,
            "text": "generic background about unrelated language model training",
        },
        {
            "chunk_id": "arxiv:test::chunk:2",
            "paper_id": "arxiv:test",
            "section": "Results",
            "section_index": 3,
            "chunk_index": 2,
            "section_chunk_index": 0,
            "start_word": 0,
            "end_word": 8,
            "word_count": 8,
            "text": "benchmark evaluation shows retrieval performance improvements",
        },
    ]
    embedded_chunks = embed_chunks(chunks=chunks, embedder=KeywordBgeEmbedder())
    embeddings_path = save_embeddings_jsonl(
        embedded_chunks=embedded_chunks,
        path=tmp_path / "embeddings.jsonl",
    )

    results = search_embedded_chunks(
        query="query-aware diversity for RAG retrieval",
        embeddings=load_embeddings_jsonl(embeddings_path),
        embedder=KeywordBgeEmbedder(),
        top_k=2,
    )

    assert results[0].chunk_id == "arxiv:test::chunk:0"
    assert results[0].section == "Methodology"
    assert "retrieval augmented generation" in results[0].text
    assert results[0].score > results[1].score


def test_search_embedded_chunks_excludes_front_matter_by_default(tmp_path):
    chunks = [
        {
            "chunk_id": "arxiv:test::chunk:0",
            "paper_id": "arxiv:test",
            "section": "Front Matter",
            "section_index": 0,
            "chunk_index": 0,
            "section_chunk_index": 0,
            "start_word": 0,
            "end_word": 8,
            "word_count": 8,
            "text": "query aware diversity retrieval augmented generation title",
        },
        {
            "chunk_id": "arxiv:test::chunk:1",
            "paper_id": "arxiv:test",
            "section": "Abstract",
            "section_index": 1,
            "chunk_index": 1,
            "section_chunk_index": 0,
            "start_word": 0,
            "end_word": 8,
            "word_count": 8,
            "text": "query aware diversity for retrieval augmented generation",
        },
    ]
    embedded_chunks = embed_chunks(chunks=chunks, embedder=KeywordBgeEmbedder())
    embeddings_path = save_embeddings_jsonl(
        embedded_chunks=embedded_chunks,
        path=tmp_path / "embeddings.jsonl",
    )

    results = search_embedded_chunks(
        query="query-aware diversity for RAG retrieval",
        embeddings=load_embeddings_jsonl(embeddings_path),
        embedder=KeywordBgeEmbedder(),
        top_k=2,
    )

    assert [result.section for result in results] == ["Abstract"]


def test_embed_selected_paper_chunks_saves_embeddings_and_updates_state(tmp_path):
    store = PaperStore(
        db_path=tmp_path / "metadata" / "papers.sqlite3",
        papers_dir=tmp_path / "papers",
    )
    paper = Paper(
        paper_id="arxiv:embed",
        title="Embedding Paper",
        source="arxiv",
        url="https://arxiv.org/abs/embed",
    )
    state = AgentState(topic="rag embeddings", max_papers=1)
    state.set_selected_papers([paper])

    chunks = chunk_section(
        section=Section(
            title="Introduction",
            text=" ".join(f"retrieval{i}" for i in range(30)),
        ),
        paper_id=paper.paper_id,
        min_chunk_words=10,
        target_chunk_words=10,
        max_chunk_words=12,
        overlap_words=2,
    )
    chunks_path = save_chunks_jsonl(chunks, store.chunks_path(paper.paper_id))
    state.set_paper_chunk_paths({paper.paper_id: str(chunks_path)})

    observation = embed_selected_paper_chunks(
        state=state,
        file_store=store,
        embedder=FakeBgeEmbedder(),
    )
    embeddings_path = store.embeddings_path(paper.paper_id)
    rows = [
        json.loads(line)
        for line in embeddings_path.read_text(encoding="utf-8").splitlines()
    ]

    assert observation["status"] == "success"
    assert observation["processed"] == 1
    assert observation["embedded_chunks"] == len(chunks)
    assert state.paper_embedding_paths == {paper.paper_id: str(embeddings_path)}
    assert rows[0]["paper_id"] == paper.paper_id
    assert rows[0]["section"] == "Introduction"
    assert rows[0]["embedding_model"] == DEFAULT_BGE_MODEL_NAME
