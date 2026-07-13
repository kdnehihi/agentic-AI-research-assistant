from app.retrieval.hybrid_retriever import HybridRetriever, HybridScoreWeights
from app.retrieval.models import RetrievalFilters, RetrievalRequest
from app.vectorstores.models import VectorSearchResult


class FakeEmbedder:
    model_name = "fake"

    def embed_query(self, query):
        return [1.0, 0.0]

    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]


class FakeVectorStore:
    def __init__(self, results):
        self.results = results
        self.last_filters = None
        self.last_top_k = None

    def search(self, *, query_embedding, top_k, filters=None, include_embeddings=False):
        self.last_filters = filters
        self.last_top_k = top_k
        return self.results[:top_k]


def test_hybrid_retriever_uses_bm25_and_query_section_intent():
    store = FakeVectorStore(
        [
            _result(
                chunk_id="conclusion",
                section="Conclusion",
                section_group="conclusion",
                distance=0.05,
                document="This paper concludes with future work.",
            ),
            _result(
                chunk_id="limitations",
                section="Limitations",
                section_group="limitations",
                distance=0.20,
                document="The limitations include missed industry studies and narrow coverage.",
            ),
        ]
    )
    retriever = HybridRetriever(
        embedder=FakeEmbedder(),
        vector_store=store,
        weights=HybridScoreWeights(semantic=0.65, bm25=0.25, metadata=0.10),
    )

    results = retriever.retrieve(
        RetrievalRequest(
            query="What limitations are discussed?",
            top_k=2,
            candidate_k=2,
            filters=RetrievalFilters(paper_ids=("paper_1",)),
        )
    )

    assert [result.chunk_id for result in results] == ["limitations", "conclusion"]
    assert results[0].metadata_score == 1.0
    assert results[0].metadata["bm25_score"] > results[1].metadata["bm25_score"]
    assert store.last_filters == RetrievalFilters(paper_ids=("paper_1",))
    assert store.last_top_k == 2


def _result(chunk_id, section, section_group, distance, document):
    return VectorSearchResult(
        id=chunk_id,
        document=document,
        distance=distance,
        metadata={
            "paper_id": "paper_1",
            "section": section,
            "section_group": section_group,
        },
    )
