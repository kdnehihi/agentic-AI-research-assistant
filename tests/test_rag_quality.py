import pytest

from app.agent.state import AgentState
from app.retrieval.answering import (
    RetrievalAugmentedAnswerer,
    build_evidence_chunks,
    build_grounded_answer_prompt,
)
from app.retrieval.evaluation import RetrievalEvalCase, evaluate_retrieval_results
from app.retrieval.hybrid_retriever import HybridRetriever, HybridScoreWeights
from app.retrieval.models import RetrievedChunk, RetrievalFilters, RetrievalRequest
from app.retrieval.query_intent import (
    infer_explicit_section_groups_from_query,
    infer_section_groups_from_query,
)
from app.tools.production.retrieval_tools import retrieve_evidence
from app.vectorstores.models import VectorSearchResult


class KeywordEmbedder:
    model_name = "quality-test"

    def embed_query(self, query):
        del query
        return [1.0, 0.0]

    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]


class FilteringVectorStore:
    def __init__(self, results):
        self.results = list(results)
        self.last_filters = None
        self.last_top_k = None

    def get_by_paper(self, paper_id):
        return [
            result
            for result in self.results
            if result.metadata.get("paper_id") == paper_id
        ]

    def search(self, *, query_embedding, top_k, filters=None, include_embeddings=False):
        del query_embedding, include_embeddings
        self.last_filters = filters
        self.last_top_k = top_k
        results = self.results
        if filters and filters.paper_ids:
            results = [
                result
                for result in results
                if result.metadata.get("paper_id") in filters.paper_ids
            ]
        if filters and filters.section_groups:
            results = [
                result
                for result in results
                if result.metadata.get("section_group") in filters.section_groups
            ]
        return results[:top_k]


class EmptyRetriever:
    def retrieve(self, request):
        del request
        return []


class StaticRetriever:
    def __init__(self, chunks):
        self.chunks = chunks

    def retrieve(self, request):
        del request
        return self.chunks


class RefusalWhenNoEvidenceLLM:
    def generate(self, prompt, **kwargs):
        del kwargs
        if "No retrieved evidence was available." in prompt:
            return "I do not have enough evidence from the retrieved chunks to answer that."
        return "The answer is supported [E1]."


class CiteSecondEvidenceLLM:
    def generate(self, prompt, **kwargs):
        del prompt, kwargs
        return "Only the limitations chunk supports the answer [E2]."


def test_scoped_section_retrieval_isolates_active_paper_from_stale_noise():
    vector_store = FilteringVectorStore(
        [
            _search_result(
                "other_intro",
                paper_id="paper:stale",
                section="Introduction",
                section_group="introduction",
                document="This unrelated paper discusses tool synthesis.",
                distance=0.01,
            ),
            _search_result(
                "sok_method",
                paper_id="paper:sok",
                section="Method",
                section_group="method",
                document="The method formalizes agentic RAG as a decision process.",
                distance=0.05,
            ),
            _search_result(
                "sok_intro",
                paper_id="paper:sok",
                section="Introduction",
                section_group="introduction",
                document=(
                    "The introduction motivates agentic RAG by describing why "
                    "static retrieval pipelines are insufficient."
                ),
                distance=0.20,
            ),
        ]
    )

    observation = retrieve_evidence(
        AgentState(topic="agentic RAG"),
        query="Give me the introduction of the SoK paper.",
        paper_ids=["paper:sok"],
        section_groups=["introduction"],
        top_k=3,
        embedder=KeywordEmbedder(),
        vector_store=vector_store,
    )

    assert observation["status"] == "success"
    assert [item["chunk_id"] for item in observation["evidence"]] == ["sok_intro"]
    assert vector_store.last_filters.paper_ids == ("paper:sok",)
    assert vector_store.last_filters.section_groups == ("introduction",)


@pytest.mark.parametrize(
    ("query", "expected_chunk_id"),
    [
        ("What problem motivates this paper?", "intro"),
        ("What method or approach is proposed?", "method"),
        ("What limitations are discussed?", "limitations"),
    ],
)
def test_hybrid_retrieval_quality_promotes_relevant_sections(query, expected_chunk_id):
    retriever = HybridRetriever(
        embedder=KeywordEmbedder(),
        vector_store=FilteringVectorStore(
            [
                _search_result(
                    "intro",
                    section="Introduction",
                    section_group="introduction",
                    document=(
                        "The research problem motivates adaptive retrieval for "
                        "agentic RAG assistants."
                    ),
                ),
                _search_result(
                    "method",
                    section="Method",
                    section_group="method",
                    document=(
                        "The method proposes a planner that retrieves evidence "
                        "and then writes grounded answers."
                    ),
                ),
                _search_result(
                    "limitations",
                    section="Limitations",
                    section_group="limitations",
                    document=(
                        "The limitations include retrieval noise, stale memory, "
                        "and incomplete evidence coverage."
                    ),
                ),
            ]
        ),
        weights=HybridScoreWeights(semantic=0.45, bm25=0.35, metadata=0.20),
    )

    results = retriever.retrieve(
        RetrievalRequest(
            query=query,
            top_k=3,
            candidate_k=3,
            filters=RetrievalFilters(paper_ids=("paper:sok",)),
        )
    )

    assert results[0].chunk_id == expected_chunk_id
    assert results[0].metadata_score == 1.0


def test_broad_finding_query_gets_soft_section_intent_without_hard_filter():
    assert infer_section_groups_from_query("What are the main findings?") == ("results",)
    assert infer_explicit_section_groups_from_query("What are the main findings?") == ()
    assert infer_explicit_section_groups_from_query("Extract the introduction.") == (
        "introduction",
    )


def test_evidence_context_budget_uses_contiguous_ids_after_skipping_empty_chunks():
    chunks = [
        _retrieved_chunk("empty", ""),
        _retrieved_chunk("long_intro", " ".join(f"intro{i}" for i in range(30))),
        _retrieved_chunk("method", "method evidence should still fit"),
    ]

    evidence = build_evidence_chunks(
        retrieved_chunks=chunks,
        max_context_chars=90,
        max_chunk_chars=60,
    )

    assert [chunk.evidence_id for chunk in evidence] == ["E1", "E2"]
    assert [chunk.chunk_id for chunk in evidence] == ["long_intro", "method"]
    assert sum(len(chunk.text) for chunk in evidence) <= 90


def test_grounded_answer_refuses_when_retrieval_returns_no_evidence():
    answerer = RetrievalAugmentedAnswerer(
        retriever=EmptyRetriever(),
        llm_client=RefusalWhenNoEvidenceLLM(),
    )

    answer = answerer.answer(RetrievalRequest(query="What GPU was used?"))

    assert answer.answer == (
        "I do not have enough evidence from the retrieved chunks to answer that."
    )
    assert answer.evidence_chunks == []
    assert answer.cited_evidence_ids == []
    assert answer.cited_chunk_ids == []


def test_grounded_answer_maps_citations_to_the_actual_cited_chunks_only():
    answerer = RetrievalAugmentedAnswerer(
        retriever=StaticRetriever(
            [
                _retrieved_chunk(
                    "intro",
                    "The introduction motivates the problem.",
                    section="Introduction",
                ),
                _retrieved_chunk(
                    "limitations",
                    "The limitations include stale retrieval evidence.",
                    section="Limitations",
                    rank=2,
                ),
            ]
        ),
        llm_client=CiteSecondEvidenceLLM(),
    )

    answer = answerer.answer(RetrievalRequest(query="What limitations are discussed?"))

    assert answer.cited_evidence_ids == ["E2"]
    assert answer.cited_chunk_ids == ["limitations"]


def test_retrieval_quality_gate_metrics_cover_multiple_question_types():
    cases = [
        RetrievalEvalCase(
            query="What problem motivates this paper?",
            case_id="intro",
            relevant_chunk_ids=("intro",),
            relevance_by_chunk_id={"intro": 3},
        ),
        RetrievalEvalCase(
            query="What method is proposed?",
            case_id="method",
            relevant_chunk_ids=("method",),
            relevance_by_chunk_id={"method": 3},
        ),
        RetrievalEvalCase(
            query="What limitations are discussed?",
            case_id="limitations",
            relevant_chunk_ids=("limitations",),
            relevance_by_chunk_id={"limitations": 3},
        ),
    ]
    summary = evaluate_retrieval_results(
        cases=cases,
        results_by_query={
            "intro": ["intro", "method", "noise"],
            "method": ["method", "intro", "noise"],
            "limitations": ["limitations", "noise", "method"],
        },
        top_k=3,
    )

    assert summary.num_cases == 3
    assert summary.hit_rate_at_k == 1.0
    assert summary.mrr == 1.0
    assert summary.mean_ndcg_at_k == 1.0


def test_grounded_prompt_names_evidence_and_refusal_rule():
    prompt = build_grounded_answer_prompt(query="q", evidence_chunks=[])

    assert "No retrieved evidence was available." in prompt
    assert "output exactly this sentence" in prompt
    assert "I do not have enough evidence" in prompt


def _search_result(
    chunk_id,
    *,
    paper_id="paper:sok",
    section,
    section_group,
    document,
    distance=0.20,
):
    return VectorSearchResult(
        id=chunk_id,
        document=document,
        distance=distance,
        metadata={
            "paper_id": paper_id,
            "title": "SoK RAG",
            "section": section,
            "section_group": section_group,
        },
    )


def _retrieved_chunk(chunk_id, document, *, section="Introduction", rank=1):
    return RetrievedChunk(
        chunk_id=chunk_id,
        paper_id="paper:sok",
        document=document,
        metadata={
            "title": "SoK RAG",
            "section": section,
            "section_group": section.lower(),
        },
        distance=0.1,
        semantic_score=0.95,
        metadata_score=1.0,
        final_score=0.96,
        rank=rank,
    )
