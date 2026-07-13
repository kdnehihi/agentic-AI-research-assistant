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
CONTEXT_MISS_MULTIPLIER = 0.6

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

RAG_CORE_TERMS = [
    "rag",
    "retrieval augmented generation",
    "retrieval-augmented generation",
]

RAG_CONTEXT_TERMS = [
    "agentic",
    "agent",
    "agents",
    "scientific",
    "science",
    "literature",
    "summarization",
    "summarisation",
    "summary",
    "question answering",
    "qa",
    "literature search",
    "paper summarization",
    "research paper summarization",
]

KNOWN_TITLE_PHRASES = [
    "agentic rag",
    "agentic retrieval augmented generation",
    "retrieval augmented generation",
    "retrieval-augmented generation",
    "scientific literature",
    "literature search",
    "research paper summarization",
    "paper summarization",
    "question answering",
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
    """Rank candidate papers with hard gates plus hybrid lexical/semantic scoring."""

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
    context_terms = _context_terms_for_query(user_query, state)
    title_key_terms = _title_key_terms_for_query(
        query=user_query,
        core_terms=core_terms,
        context_terms=context_terms,
    )

    lexical_scores = _bm25_scores(query_terms, paper_texts)
    semantic_scores = _semantic_scores(normalized_query, paper_texts)
    title_exact_scores = [
        _title_exact_match_score(title_key_terms=title_key_terms, title=title)
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
        context_match_score = _context_match_score(paper_text, context_terms)

        if not passes_hard_gate:
            blocked_by_gate += 1
            paper.score = 0.0
            paper.score_components = {
                "bm25_lexical": lexical_scores[idx],
                "semantic": semantic_scores[idx],
                "title_exact_match": title_exact_scores[idx],
                "recency": recency_scores[idx],
                "context_match": context_match_score,
            }
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
        if context_terms and context_match_score == 0.0:
            hybrid_score *= CONTEXT_MISS_MULTIPLIER

        paper.score = float(hybrid_score * SCORE_SCALE)
        paper.score_components = {
            "bm25_lexical": lexical_scores[idx],
            "semantic": semantic_scores[idx],
            "title_exact_match": title_exact_scores[idx],
            "recency": recency_scores[idx],
            "context_match": context_match_score,
        }
        paper.relevant_reasons = _build_hybrid_reasons(
            core_terms=core_terms,
            context_terms=context_terms,
            paper_text=paper_text,
            lexical_score=lexical_scores[idx],
            semantic_score=semantic_scores[idx],
            title_exact_score=title_exact_scores[idx],
            recency_score=recency_scores[idx],
            context_match_score=context_match_score,
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
    """Build the searchable title + abstract text for one paper."""

    return _normalize_text(f"{paper.title} {paper.abstract or ''}")


def _semantic_scores(query: str, docs: list[str]) -> list[float]:
    """Compute TF-IDF cosine similarity scores for query and documents."""

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
    """Compute normalized BM25 lexical scores for candidate paper text."""

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


def _title_exact_match_score(title_key_terms: list[str], title: str) -> float:
    """Score how many important query phrases appear exactly in the title."""

    if not title_key_terms:
        return 0.0

    matched = [
        term
        for term in title_key_terms
        if _contains_phrase(title, term)
    ]
    return len(matched) / len(title_key_terms)


def _recency_score(published_date: str | None) -> float:
    """Give newer papers a small score boost based on publication date."""

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
    """Extract core hard-gate terms from the query and optional search plan."""

    planned_terms = []
    if state.search_plan:
        planned_terms.extend(state.search_plan.core_terms)

    query_lower = query.lower()
    for term_group in CORE_TERM_GROUPS:
        if any(term in query_lower for term in term_group):
            planned_terms.extend(term_group)

    if any(term in query_lower for term in RAG_CORE_TERMS):
        planned_terms.extend(RAG_CORE_TERMS)

    return _dedupe_preserve_order(
        _normalize_text(term)
        for term in planned_terms
        if _normalize_text(term)
    )


def _context_terms_for_query(query: str, state: AgentState) -> list[str]:
    """Extract softer context terms that can penalize weakly related papers."""

    planned_terms = []
    if state.search_plan:
        planned_terms.extend(state.search_plan.context_terms)

    query_lower = query.lower()
    if any(term in query_lower for term in RAG_CORE_TERMS):
        planned_terms.extend(
            term
            for term in RAG_CONTEXT_TERMS
            if term in query_lower
        )

    return _dedupe_preserve_order(
        _normalize_text(term)
        for term in planned_terms
        if _normalize_text(term)
    )


def _title_key_terms_for_query(
    query: str,
    core_terms: list[str],
    context_terms: list[str],
) -> list[str]:
    """Choose phrase-level title terms instead of matching the full user query."""

    normalized_query = _normalize_text(query)
    title_terms = [
        phrase
        for phrase in KNOWN_TITLE_PHRASES
        if phrase in normalized_query
    ]
    title_terms.extend(core_terms)
    title_terms.extend(context_terms)

    return _dedupe_preserve_order(
        term
        for term in title_terms
        if len(term) > 2
    )


def _context_match_score(text: str, context_terms: list[str]) -> float:
    """Return a binary score for whether any context signal appears."""

    if not context_terms:
        return 0.0

    return 1.0 if _matches_any_core_term(text, context_terms) else 0.0


def _matches_any_core_term(text: str, core_terms: list[str]) -> bool:
    """Return whether text contains at least one core search term."""

    return any(_contains_phrase(text, term) for term in core_terms)


def _contains_phrase(text: str, phrase: str) -> bool:
    """Match multi-word phrases by substring and single terms by word boundary."""

    if not phrase:
        return False

    if " " in phrase:
        return phrase in text

    return re.search(rf"\b{re.escape(phrase)}\b", text) is not None


def _normalize_text(text: str) -> str:
    """Lowercase text and keep only searchable alphanumeric/hyphen characters."""

    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _tokenize(text: str) -> list[str]:
    """Tokenize normalized text for lexical scoring."""

    return re.findall(r"[a-z0-9]+", _normalize_text(text))


def _min_max_normalize(scores: list[float]) -> list[float]:
    """Normalize scores to 0-1 while preserving positive single-value cases."""

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
    """Remove duplicates while preserving first occurrence order."""

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
    context_terms: list[str],
    paper_text: str,
    lexical_score: float,
    semantic_score: float,
    title_exact_score: float,
    recency_score: float,
    context_match_score: float,
    hard_gate_enabled: bool,
) -> list[str]:
    """Build human-readable scoring reasons for the selected paper report."""

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

    if context_terms:
        if context_match_score > 0:
            reasons.append("Matched context signal")
        else:
            reasons.append("Missing context signal; score softly penalized")

    reasons.extend(
        [
            f"BM25 lexical score: {lexical_score:.3f}",
            f"Semantic score: {semantic_score:.3f}",
            f"Title exact match score: {title_exact_score:.3f}",
            f"Recency score: {recency_score:.3f}",
            f"Context match score: {context_match_score:.3f}",
        ]
    )

    return reasons
