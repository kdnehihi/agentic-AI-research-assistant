from __future__ import annotations

import re


TRANSFORMER_TRIGGERS = ("transformer", "transformers")
TRANSFORMER_NON_AI_HINTS = (
    "electric",
    "electrical",
    "power",
    "voltage",
    "uhv",
    "grid",
    "distribution",
    "substation",
    "winding",
    "insulation",
    "harmonic",
    "load",
    "fault",
    "oil",
    "magnetic",
    "current transformer",
    "power transformer",
    "voltage transformer",
)
TRANSFORMER_AI_CONTEXT_TERMS = (
    "attention",
    "self-attention",
    "self attention",
    "neural network",
    "deep learning",
    "machine learning",
    "language model",
    "large language model",
    "LLM",
    "encoder decoder",
    "sequence model",
)
RECENCY_REQUEST_TERMS = (
    "latest",
    "recent",
    "newest",
    "new",
    "state of the art",
    "sota",
)
PUBLICATION_YEAR_PATTERN = re.compile(r"\b(19\d{2}|20\d{2})\b")


def normalize_paper_source_query(query: str) -> str:
    """Return a source-search query with domain context for ambiguous AI topics."""

    clean_query = _normalize_spaces(query)
    if requires_ai_domain_disambiguation(clean_query):
        additions = [
            term
            for term in TRANSFORMER_AI_CONTEXT_TERMS
            if term.lower() not in clean_query.lower()
        ]
        return _normalize_spaces(" ".join([clean_query, *additions[:5]]))
    return clean_query


def requires_ai_domain_disambiguation(query: str) -> bool:
    """Return whether a broad topic should be interpreted in the AI/ML sense."""

    lowered = query.lower()
    return (
        any(_contains_word(lowered, trigger) for trigger in TRANSFORMER_TRIGGERS)
        and not any(hint in lowered for hint in TRANSFORMER_NON_AI_HINTS)
    )


def ai_domain_terms_for_query(query: str) -> list[str]:
    """Return required AI-domain signals for disambiguating broad paper topics."""

    if not requires_ai_domain_disambiguation(query):
        return []
    return list(TRANSFORMER_AI_CONTEXT_TERMS)


def ambiguous_core_terms_for_query(query: str) -> list[str]:
    """Return the literal ambiguous core topic terms present in the query."""

    if not requires_ai_domain_disambiguation(query):
        return []
    lowered = query.lower()
    return [
        trigger
        for trigger in TRANSFORMER_TRIGGERS
        if _contains_word(lowered, trigger)
    ]


def prefers_recent_results(query: str) -> bool:
    """Return whether the user explicitly asks for newer papers."""

    lowered = query.lower()
    return any(term in lowered for term in RECENCY_REQUEST_TERMS)


def requested_publication_years(query: str) -> list[int]:
    """Return explicit publication years mentioned in the user query."""

    years: list[int] = []
    for match in PUBLICATION_YEAR_PATTERN.finditer(query):
        year = int(match.group(1))
        if year not in years:
            years.append(year)
    return years


def _normalize_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _contains_word(text: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word)}\b", text) is not None
