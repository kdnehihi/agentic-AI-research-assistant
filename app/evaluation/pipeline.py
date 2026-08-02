from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from app.evaluation.dataset import EvaluationCase, EvaluationDataset
from app.evaluation.metrics import (
    AnswerQualityScore,
    evaluate_answer_quality,
    evaluate_context_quality,
)
from app.retrieval.evaluation import RetrievalMetricResult


@dataclass(frozen=True)
class AgentRunOutput:
    """Normalized output from one agent run for evaluation."""

    answer: dict[str, Any]
    retrieved_chunk_ids: list[str]
    tool_sequence: list[str]
    status: str = "success"
    latency_ms: float | None = None
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "retrieved_chunk_ids": self.retrieved_chunk_ids,
            "tool_sequence": self.tool_sequence,
            "status": self.status,
            "latency_ms": self.latency_ms,
            "metadata": self.metadata or {},
        }


@dataclass(frozen=True)
class AgentEvaluationResult:
    """Full evaluation result for one dataset case."""

    case_id: str
    question: str
    status: str
    latency_ms: float
    tool_sequence: list[str]
    retrieval: RetrievalMetricResult
    answer_quality: AnswerQualityScore
    failures: list[str]

    @property
    def passed(self) -> bool:
        return not self.failures

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "question": self.question,
            "status": self.status,
            "latency_ms": self.latency_ms,
            "tool_sequence": self.tool_sequence,
            "retrieval": asdict(self.retrieval),
            "answer_quality": self.answer_quality.to_dict(),
            "failures": self.failures,
            "passed": self.passed,
        }


AgentRunFunction = Callable[[EvaluationCase], AgentRunOutput]


def evaluate_agent_outputs(
    dataset: EvaluationDataset,
    *,
    run_case: AgentRunFunction,
    top_k: int = 5,
) -> list[AgentEvaluationResult]:
    """Run an agent over a dataset and evaluate retrieval plus answer quality."""

    results = []
    for case in dataset.cases:
        started_at = time.perf_counter()
        output = run_case(case)
        measured_latency_ms = (time.perf_counter() - started_at) * 1000
        latency_ms = (
            output.latency_ms
            if output.latency_ms is not None
            else measured_latency_ms
        )
        retrieval_score = evaluate_context_quality(
            case,
            output.retrieved_chunk_ids,
            top_k=top_k,
        )
        answer_score = evaluate_answer_quality(case, output.answer)
        failures = _failures_for(
            output=output,
            retrieval_score=retrieval_score,
            answer_score=answer_score,
        )
        results.append(
            AgentEvaluationResult(
                case_id=case.case_id,
                question=case.question,
                status=output.status,
                latency_ms=round(latency_ms, 3),
                tool_sequence=output.tool_sequence,
                retrieval=retrieval_score,
                answer_quality=answer_score,
                failures=failures,
            )
        )
    return results


def _failures_for(
    *,
    output: AgentRunOutput,
    retrieval_score: RetrievalMetricResult,
    answer_score: AnswerQualityScore,
) -> list[str]:
    failures = []
    if output.status != "success":
        failures.append(f"status={output.status}")
    if retrieval_score.hit_at_k < 1.0:
        failures.append("retrieval_miss")
    if retrieval_score.recall_at_k < 1.0:
        failures.append(f"context_recall={retrieval_score.recall_at_k:.3f}")
    if answer_score.overall < 0.75:
        failures.append(f"answer_overall={answer_score.overall:.3f}")
    failures.extend(answer_score.failures)
    return failures
