from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from app.evaluation.dataset import AnswerExpectation, EvaluationCase
from app.retrieval.evaluation import RetrievalMetricResult, compute_retrieval_metrics


NO_ANSWER_PHRASE = "I do not have enough evidence from the retrieved chunks to answer that."


@dataclass(frozen=True)
class AnswerQualityScore:
    """Deterministic answer-quality score bundle."""

    faithfulness: float
    answer_relevancy: float
    citation_score: float
    refusal_score: float
    required_terms_score: float
    forbidden_terms_score: float
    overall: float
    failures: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_context_quality(
    case: EvaluationCase,
    retrieved_chunk_ids: list[str],
    *,
    top_k: int = 5,
) -> RetrievalMetricResult:
    """Evaluate context precision/recall using existing retrieval metrics."""

    return compute_retrieval_metrics(
        case.retrieval.to_retrieval_case(
            query=case.question,
            case_id=case.case_id,
        ),
        retrieved_chunk_ids=retrieved_chunk_ids,
        top_k=top_k,
    )


def evaluate_answer_quality(
    case: EvaluationCase,
    answer_payload: Any,
) -> AnswerQualityScore:
    """Evaluate answer quality with stable, no-LLM checks."""

    answer_text = _answer_text(answer_payload)
    expectation = case.answer
    cited_evidence_ids = _string_list(_payload_value(answer_payload, "cited_evidence_ids"))
    cited_chunk_ids = _string_list(_payload_value(answer_payload, "cited_chunk_ids"))
    citation_count = len(cited_evidence_ids or cited_chunk_ids)
    refused = _model_refused(answer_text)

    failures: list[str] = []
    refusal_score = _refusal_score(
        expect_refusal=expectation.expect_refusal,
        refused=refused,
        failures=failures,
    )
    citation_score = _citation_score(
        citation_count=citation_count,
        min_citations=expectation.min_citations,
        expect_refusal=expectation.expect_refusal,
        failures=failures,
    )
    required_terms_score = _required_terms_score(
        answer_text=answer_text,
        required_substrings=expectation.required_substrings,
        failures=failures,
    )
    forbidden_terms_score = _forbidden_terms_score(
        answer_text=answer_text,
        forbidden_substrings=expectation.forbidden_substrings,
        failures=failures,
    )
    answer_relevancy = _answer_relevancy(
        answer_text=answer_text,
        expectation=expectation,
    )
    faithfulness = _faithfulness_score(
        citation_score=citation_score,
        forbidden_terms_score=forbidden_terms_score,
        refusal_score=refusal_score,
        expect_refusal=expectation.expect_refusal,
    )
    overall = _mean(
        [
            faithfulness,
            answer_relevancy,
            citation_score,
            refusal_score,
            required_terms_score,
            forbidden_terms_score,
        ]
    )
    if not answer_text.strip():
        failures.append("missing_answer_text")

    return AnswerQualityScore(
        faithfulness=faithfulness,
        answer_relevancy=answer_relevancy,
        citation_score=citation_score,
        refusal_score=refusal_score,
        required_terms_score=required_terms_score,
        forbidden_terms_score=forbidden_terms_score,
        overall=overall,
        failures=failures,
    )


def _answer_text(answer_payload: Any) -> str:
    answer = _payload_value(answer_payload, "answer")
    if isinstance(answer, str):
        return answer
    if answer is None:
        return ""
    return str(answer)


def _payload_value(payload: Any, key: str) -> Any:
    if isinstance(payload, dict):
        return payload.get(key)
    return None


def _string_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [value for value in values if isinstance(value, str)]


def _model_refused(answer: str) -> bool:
    normalized_answer = " ".join(answer.lower().split())
    return (
        NO_ANSWER_PHRASE.lower() in normalized_answer
        or (
            "do not have enough evidence" in normalized_answer
            and "retrieved chunks" in normalized_answer
        )
    )


def _refusal_score(
    *,
    expect_refusal: bool,
    refused: bool,
    failures: list[str],
) -> float:
    if expect_refusal:
        if refused:
            return 1.0
        failures.append("expected_refusal")
        return 0.0
    if refused:
        failures.append("unexpected_refusal")
        return 0.0
    return 1.0


def _citation_score(
    *,
    citation_count: int,
    min_citations: int,
    expect_refusal: bool,
    failures: list[str],
) -> float:
    if expect_refusal:
        return 1.0 if citation_count == 0 else 0.5
    if min_citations <= 0:
        return 1.0
    if citation_count >= min_citations:
        return 1.0
    failures.append(f"citation_count={citation_count}, min={min_citations}")
    return citation_count / min_citations


def _required_terms_score(
    *,
    answer_text: str,
    required_substrings: list[str],
    failures: list[str],
) -> float:
    if not required_substrings:
        return 1.0
    normalized = answer_text.lower()
    found = [
        term
        for term in required_substrings
        if term.lower() in normalized
    ]
    missing = [
        term
        for term in required_substrings
        if term.lower() not in normalized
    ]
    if missing:
        failures.append(f"missing_required_substrings={missing!r}")
    return len(found) / len(required_substrings)


def _forbidden_terms_score(
    *,
    answer_text: str,
    forbidden_substrings: list[str],
    failures: list[str],
) -> float:
    if not forbidden_substrings:
        return 1.0
    normalized = answer_text.lower()
    found = [
        term
        for term in forbidden_substrings
        if term.lower() in normalized
    ]
    if found:
        failures.append(f"forbidden_substrings_found={found!r}")
    return 1.0 - len(found) / len(forbidden_substrings)


def _answer_relevancy(
    *,
    answer_text: str,
    expectation: AnswerExpectation,
) -> float:
    if expectation.expect_refusal:
        return 1.0 if _model_refused(answer_text) else 0.0
    expected_terms = _keywords(
        " ".join([expectation.ground_truth, *expectation.required_substrings])
    )
    if not expected_terms:
        return 1.0 if answer_text.strip() else 0.0
    answer_terms = set(_keywords(answer_text))
    return len(answer_terms.intersection(expected_terms)) / len(expected_terms)


def _faithfulness_score(
    *,
    citation_score: float,
    forbidden_terms_score: float,
    refusal_score: float,
    expect_refusal: bool,
) -> float:
    if expect_refusal:
        return refusal_score
    return _mean([citation_score, forbidden_terms_score])


def _keywords(text: str) -> list[str]:
    terms = [
        term
        for term in re.findall(r"[A-Za-z][A-Za-z0-9_\-]{2,}", text.lower())
        if term not in _STOPWORDS
    ]
    return sorted(set(terms))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


_STOPWORDS = {
    "and",
    "are",
    "for",
    "from",
    "into",
    "not",
    "that",
    "the",
    "this",
    "uses",
    "was",
    "were",
    "what",
    "with",
}
