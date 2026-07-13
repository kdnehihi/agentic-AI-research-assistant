from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class RetrievalEvalCase:
    query: str
    relevant_chunk_ids: tuple[str, ...]
    relevance_by_chunk_id: dict[str, int] = field(default_factory=dict)
    case_id: str | None = None
    paper_id: str | None = None
    gold_section: str | None = None
    section_groups: tuple[str, ...] = ()


@dataclass(frozen=True)
class RetrievalMetricResult:
    query: str
    retrieved_chunk_ids: list[str]
    relevant_chunk_ids: list[str]
    top_k: int
    hit_at_k: float
    recall_at_k: float
    precision_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float
    first_relevant_rank: int | None
    relevant_retrieved: int
    case_id: str | None = None
    paper_id: str | None = None
    gold_section: str | None = None
    section_groups: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RetrievalEvalSummary:
    top_k: int
    num_cases: int
    hit_rate_at_k: float
    mean_recall_at_k: float
    mean_precision_at_k: float
    mrr: float
    mean_ndcg_at_k: float
    results: list[RetrievalMetricResult]

    def to_dict(self) -> dict:
        return {
            "top_k": self.top_k,
            "num_cases": self.num_cases,
            "hit_rate_at_k": self.hit_rate_at_k,
            "mean_recall_at_k": self.mean_recall_at_k,
            "mean_precision_at_k": self.mean_precision_at_k,
            "mrr": self.mrr,
            "mean_ndcg_at_k": self.mean_ndcg_at_k,
            "results": [asdict(result) for result in self.results],
        }


def evaluate_retrieval_results(
    cases: list[RetrievalEvalCase],
    results_by_query: dict[str, list[str]],
    top_k: int = 5,
) -> RetrievalEvalSummary:
    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    metric_results = [
        compute_retrieval_metrics(
            case=case,
            retrieved_chunk_ids=results_by_query.get(_case_result_key(case), []),
            top_k=top_k,
        )
        for case in cases
    ]

    if not metric_results:
        return RetrievalEvalSummary(
            top_k=top_k,
            num_cases=0,
            hit_rate_at_k=0.0,
            mean_recall_at_k=0.0,
            mean_precision_at_k=0.0,
            mrr=0.0,
            mean_ndcg_at_k=0.0,
            results=[],
        )

    return RetrievalEvalSummary(
        top_k=top_k,
        num_cases=len(metric_results),
        hit_rate_at_k=_mean(result.hit_at_k for result in metric_results),
        mean_recall_at_k=_mean(result.recall_at_k for result in metric_results),
        mean_precision_at_k=_mean(result.precision_at_k for result in metric_results),
        mrr=_mean(result.reciprocal_rank for result in metric_results),
        mean_ndcg_at_k=_mean(result.ndcg_at_k for result in metric_results),
        results=metric_results,
    )


def compute_retrieval_metrics(
    case: RetrievalEvalCase,
    retrieved_chunk_ids: list[str],
    top_k: int = 5,
) -> RetrievalMetricResult:
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    if not case.relevant_chunk_ids:
        raise ValueError("relevant_chunk_ids must not be empty.")

    retrieved_at_k = retrieved_chunk_ids[:top_k]
    relevant_ids = set(case.relevant_chunk_ids)
    relevant_retrieved = sum(1 for chunk_id in retrieved_at_k if chunk_id in relevant_ids)
    first_relevant_rank = _first_relevant_rank(retrieved_at_k, relevant_ids)

    return RetrievalMetricResult(
        query=case.query,
        retrieved_chunk_ids=retrieved_at_k,
        relevant_chunk_ids=list(case.relevant_chunk_ids),
        top_k=top_k,
        hit_at_k=1.0 if relevant_retrieved > 0 else 0.0,
        recall_at_k=relevant_retrieved / len(relevant_ids),
        precision_at_k=relevant_retrieved / top_k,
        reciprocal_rank=1.0 / first_relevant_rank if first_relevant_rank else 0.0,
        ndcg_at_k=_ndcg_at_k(
            retrieved_chunk_ids=retrieved_at_k,
            relevance_by_chunk_id=_resolve_relevance(case),
            top_k=top_k,
        ),
        first_relevant_rank=first_relevant_rank,
        relevant_retrieved=relevant_retrieved,
        case_id=case.case_id,
        paper_id=case.paper_id,
        gold_section=case.gold_section,
        section_groups=list(case.section_groups),
    )


def _resolve_relevance(case: RetrievalEvalCase) -> dict[str, int]:
    if case.relevance_by_chunk_id:
        return dict(case.relevance_by_chunk_id)

    return {chunk_id: 1 for chunk_id in case.relevant_chunk_ids}


def _case_result_key(case: RetrievalEvalCase) -> str:
    return case.case_id or case.query


def _first_relevant_rank(
    retrieved_chunk_ids: list[str],
    relevant_ids: set[str],
) -> int | None:
    for index, chunk_id in enumerate(retrieved_chunk_ids, start=1):
        if chunk_id in relevant_ids:
            return index
    return None


def _ndcg_at_k(
    retrieved_chunk_ids: list[str],
    relevance_by_chunk_id: dict[str, int],
    top_k: int,
) -> float:
    dcg = _dcg([relevance_by_chunk_id.get(chunk_id, 0) for chunk_id in retrieved_chunk_ids])
    ideal_relevances = sorted(relevance_by_chunk_id.values(), reverse=True)[:top_k]
    ideal_dcg = _dcg(ideal_relevances)
    if ideal_dcg == 0.0:
        return 0.0
    return dcg / ideal_dcg


def _dcg(relevances: list[int]) -> float:
    return sum(
        (2.0 ** relevance - 1.0) / math.log2(rank + 1)
        for rank, relevance in enumerate(relevances, start=1)
    )


def _mean(values) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)
