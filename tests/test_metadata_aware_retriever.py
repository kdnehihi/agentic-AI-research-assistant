from app.retrieval.models import (
    RetrievalFilters,
    RetrievalRequest,
    SemanticMetadataHints,
)
from app.retrieval.retriever import MetadataAwareRetriever
from app.vectorstores.models import VectorSearchResult


class SpyEmbedder:
    model_name = "spy"

    def __init__(self):
        self.query_calls = []
        self.document_calls = []

    def embed_query(self, query):
        self.query_calls.append(query)
        return [1.0, 0.0, 0.0]

    def embed_documents(self, texts):
        self.document_calls.append(texts)
        return [[1.0, 0.0, 0.0] for _ in texts]


class FakeVectorStore:
    def __init__(self, results):
        self.results = results
        self.last_filters = None
        self.last_top_k = None

    def search(self, *, query_embedding, top_k, filters=None, include_embeddings=False):
        self.last_filters = filters
        self.last_top_k = top_k
        results = self.results
        if filters and filters.paper_ids:
            results = [
                result
                for result in results
                if result.metadata["paper_id"] in filters.paper_ids
            ]
        return results[:top_k]


def test_query_uses_embed_query_not_document_embedding():
    embedder = SpyEmbedder()
    store = FakeVectorStore([_result("c1", "p1", 0.1)])

    MetadataAwareRetriever(embedder=embedder, vector_store=store).retrieve(
        RetrievalRequest(query="agentic rag")
    )

    assert embedder.query_calls == ["agentic rag"]
    assert embedder.document_calls == []


def test_semantic_ranking_works_without_metadata_hints():
    embedder = SpyEmbedder()
    store = FakeVectorStore(
        [
            _result("c2", "p1", 0.4),
            _result("c1", "p1", 0.1),
        ]
    )

    results = MetadataAwareRetriever(embedder=embedder, vector_store=store).retrieve(
        RetrievalRequest(query="rag", top_k=2)
    )

    assert [result.chunk_id for result in results] == ["c1", "c2"]
    assert results[0].metadata_score == 0.0
    assert results[0].final_score == results[0].semantic_score


def test_hard_filters_restrict_candidate_scope_and_candidate_k_is_respected():
    embedder = SpyEmbedder()
    store = FakeVectorStore(
        [
            _result("c1", "p1", 0.1),
            _result("c2", "p2", 0.1),
        ]
    )

    results = MetadataAwareRetriever(embedder=embedder, vector_store=store).retrieve(
        RetrievalRequest(
            query="rag",
            top_k=1,
            candidate_k=2,
            filters=RetrievalFilters(paper_ids=("p2",)),
        )
    )

    assert [result.paper_id for result in results] == ["p2"]
    assert store.last_filters == RetrievalFilters(paper_ids=("p2",))
    assert store.last_top_k == 2


def test_soft_metadata_hints_promote_but_do_not_exclude_candidates():
    embedder = SpyEmbedder()
    store = FakeVectorStore(
        [
            _result("c1", "p1", 0.2, topics=["general"]),
            _result("c2", "p1", 0.21, topics=["agentic_rag"]),
        ]
    )

    results = MetadataAwareRetriever(embedder=embedder, vector_store=store).retrieve(
        RetrievalRequest(
            query="rag",
            top_k=2,
            candidate_k=2,
            metadata_hints=SemanticMetadataHints(topics=("agentic rag",)),
            metadata_weight=0.5,
        )
    )

    assert [result.chunk_id for result in results] == ["c2", "c1"]
    assert len(results) == 2
    assert results[0].metadata_score == 1.0
    assert results[1].metadata_score == 0.0


def test_metadata_weight_zero_produces_semantic_only_ranking():
    embedder = SpyEmbedder()
    store = FakeVectorStore(
        [
            _result("c1", "p1", 0.1, topics=["general"]),
            _result("c2", "p1", 0.2, topics=["agentic_rag"]),
        ]
    )

    results = MetadataAwareRetriever(embedder=embedder, vector_store=store).retrieve(
        RetrievalRequest(
            query="rag",
            top_k=2,
            candidate_k=2,
            metadata_hints=SemanticMetadataHints(topics=("agentic_rag",)),
            metadata_weight=0.0,
        )
    )

    assert [result.chunk_id for result in results] == ["c1", "c2"]


def test_sorting_is_deterministic_with_ties_and_scores_are_separate():
    embedder = SpyEmbedder()
    store = FakeVectorStore(
        [
            _result("c2", "p1", 0.1),
            _result("c1", "p1", 0.1),
        ]
    )

    results = MetadataAwareRetriever(embedder=embedder, vector_store=store).retrieve(
        RetrievalRequest(query="rag", top_k=2, candidate_k=2)
    )

    assert [result.chunk_id for result in results] == ["c1", "c2"]
    assert results[0].rank == 1
    assert results[0].distance == 0.1
    assert results[0].semantic_score == 0.95
    assert results[0].final_score == 0.95


def _result(chunk_id, paper_id, distance, topics=None):
    return VectorSearchResult(
        id=chunk_id,
        document=f"document {chunk_id}",
        distance=distance,
        metadata={
            "paper_id": paper_id,
            "topics": topics or [],
        },
    )
