from __future__ import annotations

from typing import Any

from app.evaluation.dataset import (
    AnswerExpectation,
    EvaluationCase,
    EvaluationDataset,
    RetrievalExpectation,
)


SECTION_QUESTION_TEMPLATES = {
    "introduction": "What problem does the paper introduce?",
    "method": "What method or approach does the paper use?",
    "limitations": "What limitations are discussed?",
    "results": "What results are reported?",
    "conclusion": "What does the paper conclude?",
}


def synthetic_cases_from_chunks(
    chunks: list[dict[str, Any]],
    *,
    dataset_name: str = "synthetic_from_chunks",
    max_cases: int = 20,
) -> EvaluationDataset:
    """Create deterministic section-grounded eval cases from indexed chunks.

    This is intentionally no-LLM so the evaluator remains stable in CI. A later
    LLM-backed generator can produce richer questions while keeping this schema.
    """

    cases: list[EvaluationCase] = []
    for chunk in chunks:
        if len(cases) >= max_cases:
            break
        chunk_id = str(chunk.get("chunk_id") or chunk.get("id") or "")
        paper_id = str(chunk.get("paper_id") or "")
        text = str(chunk.get("text") or chunk.get("document") or "")
        section_group = str(
            chunk.get("section_group")
            or chunk.get("section")
            or "method"
        ).lower()
        if not chunk_id or not paper_id or not text.strip():
            continue
        question = SECTION_QUESTION_TEMPLATES.get(
            section_group,
            "What evidence is relevant in this paper?",
        )
        cases.append(
            EvaluationCase(
                case_id=f"synthetic_{len(cases) + 1}_{chunk_id}",
                question=question,
                difficulty="medium",
                retrieval=RetrievalExpectation(
                    relevant_chunk_ids=[chunk_id],
                    relevance_by_chunk_id={chunk_id: 2},
                    required_paper_ids=[paper_id],
                    required_sections=[section_group],
                    section_groups=[section_group],
                ),
                answer=AnswerExpectation(
                    ground_truth=_first_sentence(text),
                    required_substrings=_salient_terms(text, limit=2),
                    min_citations=1,
                ),
                metadata={"synthetic": True},
            )
        )
    return EvaluationDataset(name=dataset_name, cases=cases)


def _first_sentence(text: str) -> str:
    for delimiter in (". ", "? ", "! "):
        if delimiter in text:
            return text.split(delimiter, 1)[0].strip() + delimiter.strip()
    return text.strip()[:300]


def _salient_terms(text: str, *, limit: int) -> list[str]:
    terms: list[str] = []
    for raw in text.replace("-", " ").split():
        term = raw.strip(".,:;()[]{}").lower()
        if len(term) < 5 or term in _STOPWORDS or term in terms:
            continue
        terms.append(term)
        if len(terms) >= limit:
            break
    return terms


_STOPWORDS = {
    "about",
    "after",
    "based",
    "between",
    "could",
    "their",
    "there",
    "these",
    "those",
    "using",
    "where",
    "which",
}
