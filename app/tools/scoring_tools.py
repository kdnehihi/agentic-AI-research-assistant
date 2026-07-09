from __future__ import annotations

import re
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.agent.state import AgentState

SCORE_SCALE = 10.0


def rank_papers_by_similarity(
    state: AgentState,
    query: str | None = None,
    max_papers: int | None = None,
    title_weight: float = 0.3,
    abstract_weight: float = 0.7,
) -> dict[str, Any]:
    query = _normalize_text(query or state.topic)
    max_papers = max_papers or state.max_papers

    if not state.candidate_papers:
        state.set_selected_papers([])
        return {
            "status": "partial_success",
            "selected": 0,
            "summary": "No candidate papers available for ranking.",
        }

    titles = [
        _normalize_text(paper.title)
        for paper in state.candidate_papers
    ]

    abstracts = [
        _normalize_text(paper.abstract or "")
        for paper in state.candidate_papers
    ]

    title_scores = _cosine_scores(query, titles)
    abstract_scores = _cosine_scores(query, abstracts)

    for paper, title_score, abstract_score in zip(
        state.candidate_papers,
        title_scores,
        abstract_scores,
    ):
        final_similarity = title_weight * title_score + abstract_weight * abstract_score

        paper.score = float(final_similarity * SCORE_SCALE)
        paper.relevant_reasons = _build_similarity_reasons(
            title_score=title_score,
            abstract_score=abstract_score,
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
        "summary": (
            f"Selected top {len(selected)} papers using title/abstract TF-IDF "
            f"similarity scaled by {SCORE_SCALE:g}."
        ),
    }


def _cosine_scores(query: str, docs: list[str]) -> list[float]:
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


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _build_similarity_reasons(
    title_score: float,
    abstract_score: float,
) -> list[str]:
    reasons = []

    if title_score > 0:
        reasons.append(f"Title similarity: {title_score:.3f}")

    if abstract_score > 0:
        reasons.append(f"Abstract similarity: {abstract_score:.3f}")

    if not reasons:
        reasons.append("Low similarity to query")

    return reasons
