from app.retrieval.answering import (
    RetrievalAugmentedAnswerer,
    build_grounded_answer_prompt,
    cited_ids_from_answer,
)
from app.retrieval.models import RetrievedChunk, RetrievalRequest


class FakeRetriever:
    def retrieve(self, request):
        return [
            RetrievedChunk(
                chunk_id="paper_1::chunk:1",
                paper_id="paper_1",
                document="The paper proposes a retrieval augmented generation method.",
                metadata={"section": "Method", "title": "RAG Paper"},
                distance=0.1,
                semantic_score=0.95,
                metadata_score=0.0,
                final_score=0.95,
                rank=1,
            ),
            RetrievedChunk(
                chunk_id="paper_1::chunk:2",
                paper_id="paper_1",
                document="The main limitation is reliance on external evidence quality.",
                metadata={"section": "Limitations", "title": "RAG Paper"},
                distance=0.2,
                semantic_score=0.90,
                metadata_score=0.0,
                final_score=0.90,
                rank=2,
            ),
        ]


class CitingFakeLLM:
    def __init__(self):
        self.prompts = []

    def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return (
            "The paper proposes a retrieval augmented generation method [E1]. "
            "Its limitation is reliance on external evidence quality [E2]."
        )


def test_answerer_returns_answer_with_cited_evidence_chunks():
    llm = CitingFakeLLM()
    answerer = RetrievalAugmentedAnswerer(
        retriever=FakeRetriever(),
        llm_client=llm,
    )

    answer = answerer.answer(RetrievalRequest(query="What is proposed and limited?"))

    assert answer.cited_evidence_ids == ["E1", "E2"]
    assert answer.cited_chunk_ids == ["paper_1::chunk:1", "paper_1::chunk:2"]
    assert len(answer.evidence_chunks) == 2
    assert "[E1]" in answer.answer
    assert "Every factual sentence" in llm.prompts[0]


def test_prompt_contains_evidence_ids_and_grounding_rules():
    evidence_chunks = RetrievalAugmentedAnswerer(
        retriever=FakeRetriever(),
        llm_client=CitingFakeLLM(),
    ).answer(RetrievalRequest(query="q")).evidence_chunks

    prompt = build_grounded_answer_prompt(
        query="What does the paper do?",
        evidence_chunks=evidence_chunks,
    )

    assert "[E1]" in prompt
    assert "paper_1::chunk:1" in prompt
    assert "Do not use outside knowledge." in prompt
    assert "Only answer claims that are directly supported" in prompt


def test_cited_ids_from_answer_deduplicates_in_order():
    assert cited_ids_from_answer("A [E2]. B [E1]. Again [E2].") == ["E2", "E1"]
