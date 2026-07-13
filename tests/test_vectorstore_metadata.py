import pytest

from app.retrieval.models import RetrievalFilters
from app.vectorstores.errors import InvalidChunkMetadataError
from app.vectorstores.metadata import build_chroma_where, normalize_metadata


def test_empty_filters_produce_none():
    assert build_chroma_where(RetrievalFilters()) is None


def test_one_paper_id_produces_equality():
    assert build_chroma_where(RetrievalFilters(paper_ids=("p1",))) == {"paper_id": "p1"}


def test_multiple_paper_ids_produce_in():
    assert build_chroma_where(RetrievalFilters(paper_ids=("p1", "p2"))) == {
        "paper_id": {"$in": ["p1", "p2"]}
    }


def test_knowledge_base_filters_use_contains_and_or():
    assert build_chroma_where(RetrievalFilters(knowledge_base_ids=("agentic_rag",))) == {
        "knowledge_base_ids": {"$contains": "agentic_rag"}
    }
    assert build_chroma_where(
        RetrievalFilters(knowledge_base_ids=("agentic_rag", "rlvr"))
    ) == {
        "$or": [
            {"knowledge_base_ids": {"$contains": "agentic_rag"}},
            {"knowledge_base_ids": {"$contains": "rlvr"}},
        ]
    }


def test_date_range_filters_are_inclusive_and_composable():
    assert build_chroma_where(
        RetrievalFilters(
            paper_ids=("p1",),
            published_from_yyyymmdd=20250101,
            published_to_yyyymmdd=20251231,
        )
    ) == {
        "$and": [
            {"paper_id": "p1"},
            {"published_yyyymmdd": {"$gte": 20250101}},
            {"published_yyyymmdd": {"$lte": 20251231}},
        ]
    }


def test_invalid_date_range_fails():
    with pytest.raises(ValueError):
        RetrievalFilters(
            published_from_yyyymmdd=20251231,
            published_to_yyyymmdd=20250101,
        )


def test_metadata_normalization_removes_empty_values_and_normalizes_tags():
    metadata = _required_metadata()
    metadata.update(
        {
            "authors": [],
            "topics": ["Agentic RAG", None, "Agentic RAG"],
            "methods": None,
        }
    )

    normalized = normalize_metadata(metadata)

    assert "authors" not in normalized
    assert "methods" not in normalized
    assert normalized["topics"] == ["agentic_rag"]


def test_required_metadata_validation_fails_clearly():
    metadata = _required_metadata()
    del metadata["paper_id"]

    with pytest.raises(InvalidChunkMetadataError, match="paper_id"):
        normalize_metadata(metadata)


def _required_metadata():
    return {
        "paper_id": "p1",
        "knowledge_base_ids": ["agentic_rag"],
        "source": "arxiv",
        "language": "en",
        "title": "Paper",
        "published_year": 2025,
        "published_yyyymmdd": 20250101,
        "section": "Introduction",
        "section_group": "introduction",
        "chunk_type": "paper_section_chunk",
        "chunk_index": 0,
        "word_count": 10,
        "text_source": "clean_text",
        "metadata_schema_version": 1,
        "embedding_model_id": "fake",
        "embedding_dimension": 3,
    }
