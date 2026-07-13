import pytest

from app.retrieval.evaluation import (
    RetrievalEvalCase,
    compute_retrieval_metrics,
    evaluate_retrieval_results,
)


def test_compute_retrieval_metrics_hit_recall_precision_mrr_ndcg():
    case = RetrievalEvalCase(
        query="How does the paper implement episodic memory?",
        relevant_chunk_ids=("chunk_method", "chunk_appendix"),
        relevance_by_chunk_id={
            "chunk_method": 3,
            "chunk_appendix": 1,
        },
        paper_id="paper_1",
        gold_section="3.2 Episodic Memory Module",
    )

    result = compute_retrieval_metrics(
        case=case,
        retrieved_chunk_ids=["chunk_intro", "chunk_method", "chunk_noise"],
        top_k=3,
    )

    assert result.hit_at_k == 1.0
    assert result.recall_at_k == 0.5
    assert result.precision_at_k == pytest.approx(1 / 3)
    assert result.reciprocal_rank == 0.5
    assert result.first_relevant_rank == 2
    assert 0.0 < result.ndcg_at_k < 1.0


def test_evaluate_retrieval_results_averages_cases():
    cases = [
        RetrievalEvalCase(query="q1", relevant_chunk_ids=("a",)),
        RetrievalEvalCase(query="q2", relevant_chunk_ids=("b",)),
    ]
    results_by_query = {
        "q1": ["a", "x"],
        "q2": ["x", "y"],
    }

    summary = evaluate_retrieval_results(
        cases=cases,
        results_by_query=results_by_query,
        top_k=2,
    )

    assert summary.num_cases == 2
    assert summary.hit_rate_at_k == 0.5
    assert summary.mean_recall_at_k == 0.5
    assert summary.mean_precision_at_k == 0.25
    assert summary.mrr == 0.5
    assert summary.mean_ndcg_at_k == 0.5


def test_compute_retrieval_metrics_requires_relevant_chunks():
    case = RetrievalEvalCase(query="q", relevant_chunk_ids=())

    with pytest.raises(ValueError):
        compute_retrieval_metrics(case=case, retrieved_chunk_ids=["x"])
