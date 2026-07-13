import pytest

from app.retrieval.models import RetrievalFilters
from app.vectorstores.chroma_store import ChromaVectorStore
from app.vectorstores.errors import EmbeddingDimensionMismatchError
from app.vectorstores.models import VectorRecord

chromadb = pytest.importorskip("chromadb")


def test_records_can_be_upserted_and_reupsert_is_idempotent(tmp_path):
    store = _store(tmp_path)

    result = store.upsert_records([_record("c1", "p1", [1.0, 0.0, 0.0])])
    second = store.upsert_records([_record("c1", "p1", [1.0, 0.0, 0.0])])

    assert result.upserted == 1
    assert second.upserted == 1
    assert store.count() == 1


def test_existing_records_are_updated(tmp_path):
    store = _store(tmp_path)

    store.upsert_records([_record("c1", "p1", [1.0, 0.0, 0.0], document="old")])
    store.upsert_records([_record("c1", "p1", [0.0, 1.0, 0.0], document="new")])
    records = store.get_by_ids(["c1"])

    assert len(records) == 1
    assert records[0].document == "new"
    assert records[0].embedding == [0.0, 1.0, 0.0]


def test_persistence_works_after_reopening(tmp_path):
    store = _store(tmp_path)
    store.upsert_records([_record("c1", "p1", [1.0, 0.0, 0.0])])

    reopened = _store(tmp_path)

    assert reopened.count() == 1
    assert reopened.get_by_ids(["c1"])[0].metadata["paper_id"] == "p1"


def test_search_returns_typed_results_and_respects_filters(tmp_path):
    store = _store(tmp_path)
    store.upsert_records(
        [
            _record("c1", "p1", [1.0, 0.0, 0.0], kb=("agentic_rag",), section_group="method", published=20250101),
            _record("c2", "p2", [0.0, 1.0, 0.0], kb=("rlvr",), section_group="results", published=20260101),
        ]
    )

    by_paper = store.search(
        query_embedding=[1.0, 0.0, 0.0],
        top_k=5,
        filters=RetrievalFilters(paper_ids=("p1",)),
    )
    by_kb = store.search(
        query_embedding=[1.0, 0.0, 0.0],
        top_k=5,
        filters=RetrievalFilters(knowledge_base_ids=("agentic_rag",)),
    )
    by_section = store.search(
        query_embedding=[1.0, 0.0, 0.0],
        top_k=5,
        filters=RetrievalFilters(section_groups=("method",)),
    )
    by_date = store.search(
        query_embedding=[1.0, 0.0, 0.0],
        top_k=5,
        filters=RetrievalFilters(
            published_from_yyyymmdd=20250101,
            published_to_yyyymmdd=20251231,
        ),
    )

    assert [result.id for result in by_paper] == ["c1"]
    assert [result.id for result in by_kb] == ["c1"]
    assert [result.id for result in by_section] == ["c1"]
    assert [result.id for result in by_date] == ["c1"]
    assert by_paper[0].document
    assert by_paper[0].metadata["knowledge_base_ids"] == ["agentic_rag"]


def test_delete_by_paper_only_removes_that_paper(tmp_path):
    store = _store(tmp_path)
    store.upsert_records(
        [
            _record("c1", "p1", [1.0, 0.0, 0.0]),
            _record("c2", "p2", [0.0, 1.0, 0.0]),
        ]
    )

    removed = store.delete_by_paper("p1")

    assert removed == 1
    assert store.count() == 1
    assert store.get_by_ids(["c2"])[0].metadata["paper_id"] == "p2"


def test_embedding_dimension_mismatch_fails(tmp_path):
    store = _store(tmp_path)

    with pytest.raises(EmbeddingDimensionMismatchError):
        store.upsert_records([_record("c1", "p1", [1.0, 0.0])])


def test_empty_collection_search_returns_empty_list(tmp_path):
    store = _store(tmp_path)

    assert store.search(query_embedding=[1.0, 0.0, 0.0], top_k=5) == []


def _store(tmp_path):
    return ChromaVectorStore(
        persist_path=tmp_path / "chroma",
        collection_name="test_chunks",
        embedding_model_id="fake",
        embedding_dimension=3,
        metadata_schema_version=1,
    )


def _record(
    chunk_id,
    paper_id,
    embedding,
    document="retrieval augmented generation",
    kb=("agentic_rag",),
    section_group="introduction",
    published=20250101,
):
    return VectorRecord(
        id=chunk_id,
        document=document,
        embedding=embedding,
        metadata={
            "paper_id": paper_id,
            "knowledge_base_ids": list(kb),
            "source": "arxiv",
            "language": "en",
            "title": f"Paper {paper_id}",
            "published_year": published // 10000,
            "published_yyyymmdd": published,
            "section": "Introduction",
            "section_group": section_group,
            "chunk_type": "paper_section_chunk",
            "chunk_index": 0,
            "word_count": 10,
            "text_source": "clean_text",
            "metadata_schema_version": 1,
            "embedding_model_id": "fake",
            "embedding_dimension": 3,
        },
    )
