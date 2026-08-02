from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.retrieval.evaluation import RetrievalEvalCase


Difficulty = Literal["easy", "medium", "hard"]


class RetrievalExpectation(BaseModel):
    """Ground-truth retrieval labels for one evaluation query."""

    model_config = ConfigDict(extra="forbid")

    relevant_chunk_ids: list[str] = Field(default_factory=list)
    relevance_by_chunk_id: dict[str, int] = Field(default_factory=dict)
    required_paper_ids: list[str] = Field(default_factory=list)
    required_sections: list[str] = Field(default_factory=list)
    section_groups: list[str] = Field(default_factory=list)

    def to_retrieval_case(self, *, query: str, case_id: str) -> RetrievalEvalCase:
        """Convert this expectation to the existing retrieval metric schema."""

        return RetrievalEvalCase(
            query=query,
            relevant_chunk_ids=tuple(self.relevant_chunk_ids),
            relevance_by_chunk_id=dict(self.relevance_by_chunk_id),
            case_id=case_id,
            paper_id=self.required_paper_ids[0] if self.required_paper_ids else None,
            gold_section=(
                self.required_sections[0] if self.required_sections else None
            ),
            section_groups=tuple(self.section_groups),
        )


class AnswerExpectation(BaseModel):
    """Answer-level checks that can run without a judge model."""

    model_config = ConfigDict(extra="forbid")

    ground_truth: str = ""
    required_substrings: list[str] = Field(default_factory=list)
    forbidden_substrings: list[str] = Field(default_factory=list)
    expect_refusal: bool = False
    min_citations: int = 0


class EvaluationCase(BaseModel):
    """One full RAG/agent evaluation item."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    question: str
    difficulty: Difficulty = "medium"
    retrieval: RetrievalExpectation
    answer: AnswerExpectation = Field(default_factory=AnswerExpectation)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationDataset(BaseModel):
    """Named collection of agent evaluation cases."""

    model_config = ConfigDict(extra="forbid")

    name: str
    cases: list[EvaluationCase] = Field(default_factory=list)


def default_evaluation_dataset() -> EvaluationDataset:
    """Return a tiny deterministic dataset for CI and evaluator smoke tests."""

    return EvaluationDataset(
        name="default_agent_quality_smoke",
        cases=[
            EvaluationCase(
                case_id="mainrag_filtering",
                question="What does MAIN-RAG use to filter noisy evidence?",
                difficulty="easy",
                retrieval=RetrievalExpectation(
                    relevant_chunk_ids=["mainrag_filtering"],
                    relevance_by_chunk_id={"mainrag_filtering": 3},
                    required_paper_ids=["p-mainrag"],
                    required_sections=["method"],
                    section_groups=["method"],
                ),
                answer=AnswerExpectation(
                    ground_truth=(
                        "MAIN-RAG uses multi-agent filtering to rank and remove "
                        "noisy retrieved evidence."
                    ),
                    required_substrings=["multi-agent filtering", "noisy"],
                    min_citations=1,
                ),
            ),
            EvaluationCase(
                case_id="training_budget_refusal",
                question="What GPU model and training budget were used?",
                difficulty="hard",
                retrieval=RetrievalExpectation(
                    relevant_chunk_ids=["method_filtering"],
                    relevance_by_chunk_id={"method_filtering": 1},
                    required_paper_ids=["p-mainrag"],
                    required_sections=["method"],
                    section_groups=["method"],
                ),
                answer=AnswerExpectation(
                    ground_truth="The retrieved evidence does not state this.",
                    expect_refusal=True,
                    forbidden_substrings=["A100", "H100", "GPU-hours"],
                ),
            ),
        ],
    )
