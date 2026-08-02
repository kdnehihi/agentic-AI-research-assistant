from __future__ import annotations

from typing import Any

from app.evaluation.pipeline import AgentEvaluationResult


def build_evaluation_report(results: list[AgentEvaluationResult]) -> dict[str, Any]:
    """Aggregate case-level evaluation results into a report dictionary."""

    passed = sum(1 for result in results if result.passed)
    latencies = [result.latency_ms for result in results]
    return {
        "passed": passed,
        "failed": len(results) - passed,
        "total": len(results),
        "pass_rate": passed / len(results) if results else 0.0,
        "latency_ms": {
            "avg": _avg(latencies),
            "max": max(latencies) if latencies else 0.0,
        },
        "retrieval": {
            "context_precision": _avg(
                [result.retrieval.precision_at_k for result in results]
            ),
            "context_recall": _avg(
                [result.retrieval.recall_at_k for result in results]
            ),
            "mrr": _avg([result.retrieval.reciprocal_rank for result in results]),
            "ndcg": _avg([result.retrieval.ndcg_at_k for result in results]),
        },
        "answer": {
            "faithfulness": _avg(
                [result.answer_quality.faithfulness for result in results]
            ),
            "answer_relevancy": _avg(
                [result.answer_quality.answer_relevancy for result in results]
            ),
            "answer_score": _avg(
                [result.answer_quality.overall for result in results]
            ),
        },
        "failures_by_case": {
            result.case_id: result.failures
            for result in results
            if result.failures
        },
        "results": [result.to_dict() for result in results],
    }


def format_evaluation_report(report: dict[str, Any]) -> str:
    """Format a compact human-readable report."""

    retrieval = report["retrieval"]
    answer = report["answer"]
    latency = report["latency_ms"]
    lines = [
        (
            "agent_eval "
            f"passed={report['passed']} "
            f"failed={report['failed']} "
            f"total={report['total']} "
            f"pass_rate={report['pass_rate']:.3f}"
        ),
        f"Latency Avg      {latency['avg']:.2f} ms",
        f"Latency Max      {latency['max']:.2f} ms",
        f"Context Precision {retrieval['context_precision']:.3f}",
        f"Context Recall    {retrieval['context_recall']:.3f}",
        f"MRR               {retrieval['mrr']:.3f}",
        f"nDCG              {retrieval['ndcg']:.3f}",
        f"Faithfulness      {answer['faithfulness']:.3f}",
        f"Answer Relevancy  {answer['answer_relevancy']:.3f}",
        f"Answer Score      {answer['answer_score']:.3f}",
    ]
    for case_id, failures in report["failures_by_case"].items():
        lines.append(f"FAIL {case_id}: {', '.join(failures)}")
    return "\n".join(lines)


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
