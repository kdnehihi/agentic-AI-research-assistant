from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.evaluation import (
    AgentRunOutput,
    EvaluationCase,
    build_evaluation_report,
    default_evaluation_dataset,
    evaluate_agent_outputs,
    format_evaluation_report,
)
from app.evaluation.metrics import NO_ANSWER_PHRASE


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the agent evaluation runner."""

    parser = argparse.ArgumentParser(
        description="Run agent latency, retrieval, and answer-quality evaluation."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full JSON report.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the JSON report.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-k cutoff for retrieval metrics.",
    )
    parser.add_argument(
        "--no-fail",
        action="store_true",
        help="Exit 0 even when evaluation fails.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the default evaluation suite."""

    args = parse_args()
    dataset = default_evaluation_dataset()
    results = evaluate_agent_outputs(
        dataset,
        run_case=_run_default_static_case,
        top_k=args.top_k,
    )
    report = build_evaluation_report(results)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(format_evaluation_report(report))

    if report["failed"] and not args.no_fail:
        return 1
    return 0


def _run_default_static_case(case: EvaluationCase) -> AgentRunOutput:
    """Return stable fake run outputs for evaluator smoke tests."""

    if case.case_id == "training_budget_refusal":
        return AgentRunOutput(
            answer={
                "answer": NO_ANSWER_PHRASE,
                "source": "retrieved_evidence",
                "cited_evidence_ids": [],
                "cited_chunk_ids": [],
            },
            retrieved_chunk_ids=["method_filtering"],
            tool_sequence=["retrieve_evidence"],
            latency_ms=12.0,
        )

    return AgentRunOutput(
        answer={
            "answer": (
                "MAIN-RAG uses multi-agent filtering to rank and remove noisy "
                "retrieved evidence [E1]."
            ),
            "source": "retrieved_evidence",
            "cited_evidence_ids": ["E1"],
            "cited_chunk_ids": ["mainrag_filtering"],
        },
        retrieved_chunk_ids=["mainrag_filtering", "noise_chunk"],
        tool_sequence=["retrieve_evidence"],
        latency_ms=10.0,
    )


if __name__ == "__main__":
    raise SystemExit(main())
