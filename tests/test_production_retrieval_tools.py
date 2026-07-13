from app.agent.state import AgentState
from app.tools.production.retrieval_tools import retrieve_evidence
from app.vectorstores.models import VectorSearchResult


class FakeEmbedder:
    model_name = "fake"

    def embed_query(self, query):
        return [1.0, 0.0]

    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]


class FakeVectorStore:
    def __init__(self, indexed=True):
        self.indexed = indexed
        self.last_filters = None

    def get_by_paper(self, paper_id):
        return [object()] if self.indexed else []

    def search(self, *, query_embedding, top_k, filters=None, include_embeddings=False):
        self.last_filters = filters
        return [
            VectorSearchResult(
                id="paper:1::chunk:0",
                document="The method uses retrieval augmented generation.",
                metadata={
                    "paper_id": "paper:1",
                    "title": "RAG Paper",
                    "section": "Method",
                    "section_group": "method",
                },
                distance=0.1,
            )
        ]


def test_retrieve_evidence_returns_prerequisite_error_for_unindexed_paper():
    observation = retrieve_evidence(
        AgentState(topic="rag", max_papers=1),
        query="What method is used?",
        paper_ids=["paper:1"],
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(indexed=False),
    )

    assert observation["status"] == "failed"
    assert observation["error_type"] == "paper_not_retrievable"
    assert observation["missing_paper_ids"] == ["paper:1"]


def test_retrieve_evidence_supports_paper_kb_and_section_filters():
    vector_store = FakeVectorStore(indexed=True)

    observation = retrieve_evidence(
        AgentState(topic="rag", max_papers=1),
        query="What method is used?",
        paper_ids=["paper:1"],
        knowledge_base_ids=["default"],
        section_groups=["method"],
        top_k=1,
        embedder=FakeEmbedder(),
        vector_store=vector_store,
    )

    assert observation["status"] == "success"
    assert observation["evidence"][0]["chunk_id"] == "paper:1::chunk:0"
    assert observation["evidence"][0]["lexical_score"] is not None
    assert vector_store.last_filters.paper_ids == ("paper:1",)
    assert vector_store.last_filters.knowledge_base_ids == ("default",)
    assert vector_store.last_filters.section_groups == ("method",)
