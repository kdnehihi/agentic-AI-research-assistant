from app.services.chunk_indexing import PaperIndexMetadata, index_chunks


class FakeEmbedder:
    model_name = "fake-index"

    def __init__(self):
        self.document_batches = []
        self.query_calls = []

    def embed_documents(self, texts):
        self.document_batches.append(texts)
        return [[1.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, query):
        self.query_calls.append(query)
        return [1.0, 0.0, 0.0]


class MemoryVectorStore:
    def __init__(self):
        self.records = {}
        self.batches = []

    def upsert_records(self, records, *, batch_size=64):
        self.batches.append(list(records))
        for record in records:
            self.records[record.id] = record

        class Result:
            attempted = len(records)
            upserted = len(records)
            skipped = 0
            failed = 0

        return Result()


def test_existing_document_embedder_is_used_and_alignment_is_preserved():
    embedder = FakeEmbedder()
    store = MemoryVectorStore()
    chunks = [_chunk("c1", 0, "retrieval text"), _chunk("c2", 1, "generation text")]

    result = index_chunks(
        chunks=chunks,
        paper_metadata=_paper_metadata(),
        embedder=embedder,
        vector_store=store,
        batch_size=10,
    )

    assert result.upserted == 2
    assert len(embedder.document_batches) == 1
    assert embedder.query_calls == []
    assert list(store.records) == ["c1", "c2"]
    assert store.records["c1"].embedding == [1.0, 0.0, 0.0]
    assert store.records["c1"].metadata["paper_id"] == "p1"


def test_batch_ingestion_works_and_repeated_ingestion_does_not_duplicate():
    embedder = FakeEmbedder()
    store = MemoryVectorStore()
    chunks = [_chunk("c1", 0, "one"), _chunk("c2", 1, "two"), _chunk("c3", 2, "three")]

    first = index_chunks(chunks, _paper_metadata(), embedder, store, batch_size=2)
    second = index_chunks(chunks, _paper_metadata(), embedder, store, batch_size=2)

    assert first.upserted == 3
    assert second.upserted == 3
    assert len(store.records) == 3
    assert len(store.batches) == 4


def test_stored_document_is_raw_chunk_text_not_enriched_embedding_text():
    embedder = FakeEmbedder()
    store = MemoryVectorStore()

    index_chunks(
        [_chunk("c1", 0, "raw chunk content")],
        _paper_metadata(title="Semantic Context Title"),
        embedder,
        store,
    )

    embedding_text = embedder.document_batches[0][0]
    assert "Title: Semantic Context Title" not in embedding_text
    assert "Section: Introduction" in embedding_text
    assert "Content:\nraw chunk content" in embedding_text
    assert store.records["c1"].document == "raw chunk content"


def test_empty_ingestion_input_is_successful_noop():
    result = index_chunks([], _paper_metadata(), FakeEmbedder(), MemoryVectorStore())

    assert result.attempted == 0
    assert result.upserted == 0
    assert result.failed == 0


def _paper_metadata(title="Paper Title"):
    return PaperIndexMetadata(
        paper_id="p1",
        title=title,
        source="arxiv",
        knowledge_base_ids=("agentic_rag",),
        published_date="2025-01-02",
        topics=("Agentic RAG",),
    )


def _chunk(chunk_id, index, text):
    return {
        "chunk_id": chunk_id,
        "paper_id": "p1",
        "section": "Introduction",
        "section_index": 0,
        "chunk_index": index,
        "section_chunk_index": index,
        "start_word": 0,
        "end_word": 3,
        "section_word_count": 3,
        "word_count": 3,
        "text": text,
    }
