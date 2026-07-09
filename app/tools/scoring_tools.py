from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.agent.state import AgentState

SCORE_SCALE = 10.0
LEXICAL_WEIGHT = 0.45
SEMANTIC_WEIGHT = 0.35
TITLE_EXACT_WEIGHT = 0.15
RECENCY_WEIGHT = 0.05
BM25_K1 = 1.5
BM25_B = 0.75

CORE_TERM_GROUPS = [
    [
        "rlhf",
        "reinforcement learning from human feedback",
        "human feedback",
    ],
    [
        "rlvr",
        "reinforcement learning with verifiable rewards",
        "verifiable rewards",
        "verifiable reward",
    ],
    [
        "reward model",
        "reward models",
    ],
    [
        "preference optimization",
        "direct preference optimization",
    ],
]


def rank_papers_by_similarity(
    state: AgentState,
    query: str | None = None,
    max_papers: int | None = None,
    lexical_weight: float = LEXICAL_WEIGHT,
    semantic_weight: float = SEMANTIC_WEIGHT,
    title_exact_weight: float = TITLE_EXACT_WEIGHT,
    recency_weight: float = RECENCY_WEIGHT,
) -> dict[str, Any]:
    user_query = query or state.topic
    normalized_query = _normalize_text(user_query)
    max_papers = max_papers or state.max_papers

    if not state.candidate_papers:
        state.set_selected_papers([])
        return {
            "status": "partial_success",
            "selected": 0,
            "summary": "No candidate papers available for ranking.",
        }

    paper_texts = [
        _paper_search_text(paper)
        for paper in state.candidate_papers
    ]
    paper_titles = [
        _normalize_text(paper.title)
        for paper in state.candidate_papers
    ]
    query_terms = _tokenize(normalized_query)
    core_terms = _core_terms_for_query(user_query, state)

    lexical_scores = _bm25_scores(query_terms, paper_texts)
    semantic_scores = _semantic_scores(normalized_query, paper_texts)
    title_exact_scores = [
        _title_exact_match_score(core_terms=core_terms, title=title)
        for title in paper_titles
    ]
    recency_scores = [
        _recency_score(paper.published_date)
        for paper in state.candidate_papers
    ]

    hard_gate_enabled = bool(core_terms)
    blocked_by_gate = 0

    for idx, paper in enumerate(state.candidate_papers):
        paper_text = paper_texts[idx]
        passes_hard_gate = (
            _matches_any_core_term(paper_text, core_terms)
            if hard_gate_enabled
            else True
        )

        if not passes_hard_gate:
            blocked_by_gate += 1
            paper.score = 0.0
            paper.relevant_reasons = [
                "Blocked by hard gate: no core topic signal in title/abstract"
            ]
            continue

        hybrid_score = (
            lexical_weight * lexical_scores[idx]
            + semantic_weight * semantic_scores[idx]
            + title_exact_weight * title_exact_scores[idx]
            + recency_weight * recency_scores[idx]
        )

        paper.score = float(hybrid_score * SCORE_SCALE)
        paper.relevant_reasons = _build_hybrid_reasons(
            core_terms=core_terms,
            paper_text=paper_text,
            lexical_score=lexical_scores[idx],
            semantic_score=semantic_scores[idx],
            title_exact_score=title_exact_scores[idx],
            recency_score=recency_scores[idx],
            hard_gate_enabled=hard_gate_enabled,
        )

    ranked = sorted(
        state.candidate_papers,
        key=lambda paper: paper.score,
        reverse=True,
    )

    selected = ranked[:max_papers]
    state.set_selected_papers(selected)

    return {
        "status": "success",
        "selected": len(selected),
        "hard_gate_enabled": hard_gate_enabled,
        "blocked_by_hard_gate": blocked_by_gate,
        "summary": (
            f"Selected top {len(selected)} papers using hybrid BM25, semantic, "
            f"title-match, and recency scoring scaled by {SCORE_SCALE:g}."
        ),
    }


def _paper_search_text(paper) -> str:
    return _normalize_text(f"{paper.title} {paper.abstract or ''}")


def _semantic_scores(query: str, docs: list[str]) -> list[float]:
    if not docs:
        return []
    if not any(docs) or not query:
        return [0.0 for _ in docs]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
    )

    try:
        doc_vectors = vectorizer.fit_transform(docs)
    except ValueError:
        return [0.0 for _ in docs]

    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, doc_vectors).flatten()
    return [float(score) for score in scores]


def _bm25_scores(query_terms: list[str], docs: list[str]) -> list[float]:
    if not docs or not query_terms:
        return [0.0 for _ in docs]

    tokenized_docs = [_tokenize(doc) for doc in docs]
    doc_count = len(tokenized_docs)
    doc_lengths = [len(doc) for doc in tokenized_docs]
    avg_doc_length = sum(doc_lengths) / doc_count if doc_count else 0.0

    doc_freqs: dict[str, int] = {}
    for doc in tokenized_docs:
        for term in set(doc):
            doc_freqs[term] = doc_freqs.get(term, 0) + 1

    raw_scores = []
    for doc, doc_length in zip(tokenized_docs, doc_lengths):
        term_counts: dict[str, int] = {}
        for term in doc:
            term_counts[term] = term_counts.get(term, 0) + 1

        score = 0.0
        for term in query_terms:
            term_frequency = term_counts.get(term, 0)
            if term_frequency == 0:
                continue

            doc_frequency = doc_freqs.get(term, 0)
            inverse_doc_frequency = math.log(
                1 + (doc_count - doc_frequency + 0.5) / (doc_frequency + 0.5)
            )
            denominator = term_frequency + BM25_K1 * (
                1 - BM25_B + BM25_B * doc_length / (avg_doc_length or 1)
            )
            score += inverse_doc_frequency * (
                term_frequency * (BM25_K1 + 1) / denominator
            )

        raw_scores.append(score)

    return _min_max_normalize(raw_scores)


def _title_exact_match_score(core_terms: list[str], title: str) -> float:
    if not core_terms:
        return 0.0

    return 1.0 if _matches_any_core_term(title, core_terms) else 0.0


def _recency_score(published_date: str | None) -> float:
    if not published_date:
        return 0.0

    try:
        published = datetime.fromisoformat(published_date).replace(tzinfo=timezone.utc)
    except ValueError:
        return 0.0

    age_days = max((datetime.now(timezone.utc) - published).days, 0)
    if age_days <= 180:
        return 1.0
    if age_days <= 365:
        return 0.7
    if age_days <= 730:
        return 0.4
    return 0.1


def _core_terms_for_query(query: str, state: AgentState) -> list[str]:
    planned_terms = []
    if state.search_plan:
        planned_terms.extend(state.search_plan.core_terms)

    query_lower = query.lower()
    for term_group in CORE_TERM_GROUPS:
        if any(term in query_lower for term in term_group):
            planned_terms.extend(term_group)

    return _dedupe_preserve_order(
        _normalize_text(term)
        for term in planned_terms
        if _normalize_text(term)
    )


def _matches_any_core_term(text: str, core_terms: list[str]) -> bool:
    return any(_contains_phrase(text, term) for term in core_terms)


def _contains_phrase(text: str, phrase: str) -> bool:
    if not phrase:
        return False

    if " " in phrase:
        return phrase in text

    return re.search(rf"\b{re.escape(phrase)}\b", text) is not None


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", _normalize_text(text))


def _min_max_normalize(scores: list[float]) -> list[float]:
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return [1.0 if score > 0 else 0.0 for score in scores]

    return [
        (score - min_score) / (max_score - min_score)
        for score in scores
    ]


def _dedupe_preserve_order(values) -> list[str]:
    seen = set()
    unique_values = []

    for value in values:
        if value in seen:
            continue

        seen.add(value)
        unique_values.append(value)

    return unique_values


def _build_hybrid_reasons(
    core_terms: list[str],
    paper_text: str,
    lexical_score: float,
    semantic_score: float,
    title_exact_score: float,
    recency_score: float,
    hard_gate_enabled: bool,
) -> list[str]:
    reasons = []

    if hard_gate_enabled:
        matched_core_terms = [
            term
            for term in core_terms
            if _contains_phrase(paper_text, term)
        ]
        reasons.append(
            "Passed hard gate: "
            + ", ".join(matched_core_terms[:3])
        )

    reasons.extend(
        [
            f"BM25 lexical score: {lexical_score:.3f}",
            f"Semantic score: {semantic_score:.3f}",
            f"Title exact match score: {title_exact_score:.3f}",
            f"Recency score: {recency_score:.3f}",
        ]
    )

    return reasons
