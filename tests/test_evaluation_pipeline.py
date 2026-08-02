from app.evaluation import (
    AgentRunOutput,
    build_evaluation_report,
    default_evaluation_dataset,
    evaluate_agent_outputs,
)
from app.evaluation.metrics import NO_ANSWER_PHRASE, evaluate_answer_quality
from app.evaluation.synthetic import synthetic_cases_from_chunks


def test_evaluation_pipeline_scores_retrieval_answer_and_latency():
    dataset = default_evaluation_dataset()

    results = evaluate_agent_outputs(
        dataset,
        run_case=_good_static_run,
        top_k=2,
    )
    report = build_evaluation_report(results)

    assert report["failed"] == 0
    assert report["retrieval"]["context_recall"] == 1.0
    assert report["answer"]["faithfulness"] == 1.0
    assert report["latency_ms"]["max"] == 12.0


def test_evaluation_pipeline_detects_retrieval_and_answer_failures():
    dataset = default_evaluation_dataset()

    results = evaluate_agent_outputs(
        dataset,
        run_case=_bad_static_run,
        top_k=2,
    )
    report = build_evaluation_report(results)

    assert report["failed"] == 2
    assert "mainrag_filtering" in report["failures_by_case"]
    assert "training_budget_refusal" in report["failures_by_case"]


def test_answer_quality_detects_required_forbidden_and_refusal_rules():
    case = default_evaluation_dataset().cases[1]

    score = evaluate_answer_quality(
        case,
        {
            "answer": "The paper used H100 GPUs for many GPU-hours.",
            "source": "retrieved_evidence",
            "cited_evidence_ids": ["E1"],
        },
    )

    assert score.overall < 1.0
    assert "expected_refusal" in score.failures
    assert any("forbidden_substrings_found" in item for item in score.failures)


def test_synthetic_cases_from_chunks_creates_grounded_dataset():
    dataset = synthetic_cases_from_chunks(
        [
            {
                "chunk_id": "chunk-1",
                "paper_id": "paper-1",
                "section_group": "limitations",
                "text": "The main limitation is stale retrieval evidence.",
            }
        ]
    )

    assert dataset.name == "synthetic_from_chunks"
    assert len(dataset.cases) == 1
    assert dataset.cases[0].retrieval.relevant_chunk_ids == ["chunk-1"]
    assert "limitations" in dataset.cases[0].question.lower()


def _good_static_run(case):
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
                "evidence [E1]."
            ),
            "source": "retrieved_evidence",
            "cited_evidence_ids": ["E1"],
            "cited_chunk_ids": ["mainrag_filtering"],
        },
        retrieved_chunk_ids=["mainrag_filtering"],
        tool_sequence=["retrieve_evidence"],
        latency_ms=10.0,
    )


def _bad_static_run(case):
    if case.case_id == "training_budget_refusal":
        return AgentRunOutput(
            answer={
                "answer": "The paper used H100 GPUs for many GPU-hours.",
                "source": "retrieved_evidence",
                "cited_evidence_ids": ["E1"],
            },
            retrieved_chunk_ids=["wrong"],
            tool_sequence=["retrieve_evidence"],
            latency_ms=9.0,
        )
    return AgentRunOutput(
        answer={
            "answer": "This does not mention the expected concept.",
            "source": "retrieved_evidence",
            "cited_evidence_ids": [],
        },
        retrieved_chunk_ids=["wrong"],
        tool_sequence=["retrieve_evidence"],
        latency_ms=8.0,
    )
