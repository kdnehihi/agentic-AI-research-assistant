from app.agent.state import AgentState, Paper
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.models import RetrievedChunk
from app.retrieval.retriever import MetadataAwareRetriever
from app.tools.rag_answer_tools import answer_question_with_retrieval, _build_default_retriever


class FakeRetriever:
    def __init__(self):
        self.requests = []

    def retrieve(self, request):
        self.requests.append(request)
        return [
            RetrievedChunk(
                chunk_id="paper_1::chunk:1",
                paper_id="paper_1",
                document="The system uses retrieved chunks as grounded evidence.",
                metadata={"section": "Method", "title": "Grounded RAG"},
                distance=0.1,
                semantic_score=0.95,
                metadata_score=0.0,
                final_score=0.95,
                rank=1,
            )
        ]


class FakeLLM:
    def generate(self, prompt, **kwargs):
        return "The system answers from retrieved chunks [E1]."


def test_answer_question_with_retrieval_returns_evidence_and_citations():
    state = AgentState(topic="rag answering", max_papers=1)
    state.set_selected_papers(
        [
            Paper(
                title="Grounded RAG",
                paper_id="paper_1",
                source="arxiv",
                url="https://arxiv.org/abs/1234.5678",
            )
        ]
    )
    retriever = FakeRetriever()

    observation = answer_question_with_retrieval(
        state=state,
        query="How does the system answer?",
        retriever=retriever,
        llm_client=FakeLLM(),
        top_k=3,
    )

    assert observation["status"] == "success"
    assert observation["answer"] == "The system answers from retrieved chunks [E1]."
    assert observation["cited_chunk_ids"] == ["paper_1::chunk:1"]
    assert observation["evidence_chunks"][0]["evidence_id"] == "E1"
    assert retriever.requests[0].filters.paper_ids == ("paper_1",)
    assert retriever.requests[0].top_k == 3


def test_default_rag_answer_retriever_uses_hybrid_mode():
    retriever = _build_default_retriever(
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
        model_name="fake",
        embedding_dimension=2,
        use_hybrid_retrieval=True,
        hybrid_weights=None,
    )

    assert isinstance(retriever, HybridRetriever)


def test_default_rag_answer_retriever_can_disable_hybrid_mode():
    retriever = _build_default_retriever(
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
        model_name="fake",
        embedding_dimension=2,
        use_hybrid_retrieval=False,
        hybrid_weights=None,
    )

    assert isinstance(retriever, MetadataAwareRetriever)


class FakeEmbedder:
    model_name = "fake"

    def embed_query(self, query):
        return [1.0, 0.0]

    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]


class FakeVectorStore:
    def search(self, *, query_embedding, top_k, filters=None, include_embeddings=False):
        return []
