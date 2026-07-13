import json

from app.agent.state import AgentState, Paper
from app.tools.retrieval_eval_tools import (
    _build_eval_cases_from_state,
    evaluate_retrieval_from_selected_chunks,
)
from app.retrieval.models import RetrievalFilters
from app.vectorstores.models import VectorSearchResult


class FakeEvalEmbedder:
    model_name = "fake"

    def embed_query(self, query):
        return [1.0, 0.0]

    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]


class SpyEvalVectorStore:
    def __init__(self):
        self.filters = []

    def search(self, *, query_embedding, top_k, filters=None, include_embeddings=False):
        self.filters.append(filters)
        return [
            VectorSearchResult(
                id="paper_1::chunk:1",
                document="method chunk",
                distance=0.1,
                metadata={"paper_id": "paper_1", "section": "Method"},
            )
        ]


def test_build_eval_cases_from_real_chunk_file(tmp_path):
    paper_id = "paper_1"
    chunks_path = tmp_path / "chunks.jsonl"
    chunks = [
        {
            "chunk_id": f"{paper_id}::chunk:0",
            "paper_id": paper_id,
            "section": "Introduction",
            "section_index": 0,
            "chunk_index": 0,
            "section_chunk_index": 0,
            "start_word": 0,
            "end_word": 10,
            "word_count": 10,
            "text": "This paper studies retrieval augmented generation.",
        },
        {
            "chunk_id": f"{paper_id}::chunk:1",
            "paper_id": paper_id,
            "section": "Method",
            "section_index": 1,
            "chunk_index": 1,
            "section_chunk_index": 0,
            "start_word": 0,
            "end_word": 12,
            "word_count": 12,
            "text": "The method uses query aware diverse retrieval.",
        },
        {
            "chunk_id": f"{paper_id}::chunk:2",
            "paper_id": paper_id,
            "section": "Method",
            "section_index": 1,
            "chunk_index": 2,
            "section_chunk_index": 1,
            "start_word": 8,
            "end_word": 20,
            "word_count": 12,
            "text": "The method also reranks chunks by diversity.",
        },
    ]
    chunks_path.write_text(
        "\n".join(json.dumps(chunk) for chunk in chunks),
        encoding="utf-8",
    )

    state = AgentState(topic="rag", max_papers=1)
    state.set_selected_papers(
        [
            Paper(
                title="DF-RAG",
                paper_id=paper_id,
                source="arxiv",
                url="https://arxiv.org/abs/2601.17212",
            )
        ]
    )
    state.set_paper_chunk_paths({paper_id: str(chunks_path)})

    cases = _build_eval_cases_from_state(state=state, max_cases=2)

    assert len(cases) == 2
    assert cases[0].relevant_chunk_ids == (
        f"{paper_id}::chunk:1",
        f"{paper_id}::chunk:2",
    )
    assert cases[0].gold_section == "Method"
    assert cases[0].section_groups == ("method",)
    assert cases[0].case_id == f"{paper_id}::method"
    assert cases[0].query == "What method or approach is proposed?"


def test_retrieval_eval_filters_only_by_paper_id(tmp_path):
    paper_id = "paper_1"
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_text(
        json.dumps(
            {
                "chunk_id": f"{paper_id}::chunk:1",
                "paper_id": paper_id,
                "section": "Method",
                "section_index": 1,
                "chunk_index": 1,
                "section_chunk_index": 0,
                "start_word": 0,
                "end_word": 12,
                "word_count": 12,
                "text": "The method uses query aware diverse retrieval.",
            }
        ),
        encoding="utf-8",
    )
    state = AgentState(topic="rag", max_papers=1)
    state.set_selected_papers(
        [
            Paper(
                title="DF-RAG",
                paper_id=paper_id,
                source="arxiv",
                url="https://arxiv.org/abs/2601.17212",
            )
        ]
    )
    state.set_paper_chunk_paths({paper_id: str(chunks_path)})
    vector_store = SpyEvalVectorStore()

    observation = evaluate_retrieval_from_selected_chunks(
        state=state,
        embedder=FakeEvalEmbedder(),
        vector_store=vector_store,
        top_k=5,
        max_cases=1,
    )

    assert observation["status"] == "success"
    assert vector_store.filters == [RetrievalFilters(paper_ids=(paper_id,))]
    assert state.eval_results["retrieval_filter_mode"] == "paper_id_only"
