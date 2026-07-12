# ArXiv API Tools
# The file contains tools for interacting with the arXiv API, including searching for papers and retrieving paper details.
from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlencode
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

from app.agent.state import AgentState, Paper

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ARXIV_TIMEOUT_SECONDS = 40
ARXIV_USER_AGENT = "agentic-ai-research-assistant/0.1"
DEFAULT_CANDIDATE_MULTIPLIER = 10
MIN_CANDIDATE_RESULTS = 20
MAX_PRIMARY_TERMS = 8
MAX_CONTEXT_TERMS = 5
AI_CATEGORY_CLAUSE = "(cat:cs.CL OR cat:cs.AI OR cat:cs.LG OR cat:stat.ML)"

QUERY_STOPWORDS = {
    "about",
    "after",
    "also",
    "and",
    "are",
    "can",
    "could",
    "find",
    "for",
    "from",
    "give",
    "have",
    "help",
    "include",
    "includes",
    "including",
    "into",
    "latest",
    "like",
    "looking",
    "model",
    "models",
    "need",
    "paper",
    "papers",
    "please",
    "reduce",
    "reduces",
    "research",
    "show",
    "some",
    "system",
    "systems",
    "that",
    "the",
    "their",
    "them",
    "this",
    "use",
    "using",
    "want",
    "what",
    "with",
}

QUERY_TERM_EXPANSIONS = [
    (
        ("rlhf", "human feedback"),
        [
            "RLHF",
            "reinforcement learning from human feedback",
            "human feedback",
            "verifiable rewards",
            "preference optimization",
            "reward model",
        ],
    ),
    (
        ("rlvr", "verifiable reward", "verifiable rewards"),
        [
            "RLVR",
            "verifiable rewards",
            "reinforcement learning with verifiable rewards",
        ],
    ),
    (
        ("rag", "retrieval augmented generation", "retrieval-augmented generation"),
        [
            "RAG",
            "retrieval augmented generation",
            "retrieval-augmented generation",
        ],
    ),
    (
        ("large language model", "large language models", "llm", "llms"),
        [
            "large language model",
            "language model",
            "LLM",
        ],
    ),
]

KNOWN_SEARCH_PHRASES = [
    "chain of thought",
    "direct preference optimization",
    "in-context learning",
    "large language model",
    "language model",
    "mathematical reasoning",
    "preference optimization",
    "reasoning models",
    "retrieval augmented generation",
    "retrieval-augmented generation",
    "reward model",
    "verifiable rewards",
]


def search_arxiv_papers(
    state: AgentState,
    query: str | None = None,
    max_results: int | None = None,
) -> dict[str, Any]:
    """
    Search papers from arXiv and store them in state.candidate_papers.

    This tool only retrieves metadata:
    - title
    - authors
    - abstract
    - published date
    - source
    - url
    - paper_id

    It does not download or parse PDFs.
    """
    user_query = query or state.topic
    max_results = max_results or max(
        state.max_papers * DEFAULT_CANDIDATE_MULTIPLIER,
        MIN_CANDIDATE_RESULTS,
    )
    arxiv_query = _arxiv_query_from_state(state=state, user_query=user_query)

    params = {
        "search_query": arxiv_query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }

    url = f"{ARXIV_API_URL}?{urlencode(params)}"

    request = Request(
        url,
        headers={"User-Agent": ARXIV_USER_AGENT},
    )

    try:
        with urlopen(request, timeout=ARXIV_TIMEOUT_SECONDS) as response:
            xml_data = response.read()
    except HTTPError as exc:
        return {
            "status": "failed",
            "num_results": 0,
            "summary": _arxiv_http_error_summary(exc),
            "error": str(exc),
            "search_query": arxiv_query,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "num_results": 0,
            "summary": "Failed to fetch papers from arXiv.",
            "error": str(exc),
            "search_query": arxiv_query,
        }

    try:
        papers = _parse_arxiv_response(xml_data)
    except Exception as exc:
        return {
            "status": "failed",
            "num_results": 0,
            "summary": "Failed to parse arXiv response.",
            "error": str(exc),
            "search_query": arxiv_query,
        }

    state.set_candidate_papers(papers)

    return {
        "status": "success",
        "num_results": len(papers),
        "summary": f"Found {len(papers)} papers from arXiv for query: {user_query}",
        "search_query": arxiv_query,
    }


def _arxiv_http_error_summary(exc: HTTPError) -> str:
    if exc.code == 429:
        return (
            "arXiv rate-limited the request. Wait a bit before retrying, "
            "or reduce repeated runs."
        )

    return f"arXiv returned HTTP {exc.code}."


def _build_arxiv_search_query(user_query: str) -> str:
    """
    Build a compact arXiv query from a user topic or longer natural prompt.

    Retrieval is intentionally stricter than scoring: it searches title and
    abstract fields first, adds broad AI/ML/NLP categories, and avoids all-field
    matches for terms like "reasoning" that pull in noisy candidates.
    """
    normalized_query = _normalize_query_text(user_query)
    core_terms = _expanded_search_terms(normalized_query)

    if _is_rl_reward_topic(normalized_query):
        context_terms = _rl_context_terms(normalized_query)
        return _join_required_clauses(
            [
                _build_title_abstract_clause(core_terms),
                _build_title_abstract_clause(context_terms),
                AI_CATEGORY_CLAUSE,
            ]
        )

    if not core_terms:
        core_terms = _known_search_phrases(normalized_query)
    if not core_terms:
        core_terms = _informative_tokens(normalized_query, limit=MAX_PRIMARY_TERMS)

    context_terms = _context_terms(
        normalized_query=normalized_query,
        core_terms=core_terms,
        limit=MAX_CONTEXT_TERMS,
    )

    core_clause = _build_title_abstract_clause(core_terms)
    context_clause = _build_title_abstract_clause(context_terms)

    if core_clause and context_clause:
        return _join_required_clauses([core_clause, context_clause, AI_CATEGORY_CLAUSE])

    if core_clause:
        return _join_required_clauses([core_clause, AI_CATEGORY_CLAUSE])

    fallback_query = normalized_query or user_query
    return _join_required_clauses(
        [
            _build_title_abstract_clause([fallback_query]),
            AI_CATEGORY_CLAUSE,
        ]
    )


def build_arxiv_query_from_terms(
    core_terms: list[str],
    context_terms: list[str] | None = None,
    categories: list[str] | None = None,
) -> str:
    """
    Build an arXiv query from already-planned search terms.

    The planner may come from an LLM, but this function still owns the arXiv
    syntax and keeps retrieval limited to title/abstract plus known categories.
    """
    clean_core_terms = _sanitize_terms(core_terms, limit=MAX_PRIMARY_TERMS)
    clean_context_terms = _sanitize_terms(context_terms or [], limit=MAX_CONTEXT_TERMS)
    category_clause = _build_category_clause(categories or [])

    return _join_required_clauses(
        [
            _build_title_abstract_clause(clean_core_terms),
            _build_title_abstract_clause(clean_context_terms),
            category_clause or AI_CATEGORY_CLAUSE,
        ]
    )


def _arxiv_query_from_state(state: AgentState, user_query: str) -> str:
    if (
        state.search_plan
        and state.search_plan.original_query == user_query
        and state.search_plan.arxiv_query
    ):
        return state.search_plan.arxiv_query

    return _build_arxiv_search_query(user_query)


def _normalize_query_text(query: str) -> str:
    query = query.strip()
    query = re.sub(r"[“”]", '"', query)
    query = re.sub(r"[’']", "'", query)
    query = re.sub(r"\s+", " ", query)
    return query


def _expanded_search_terms(query: str) -> list[str]:
    query_lower = query.lower()
    terms: list[str] = []

    for triggers, expanded_terms in QUERY_TERM_EXPANSIONS:
        if any(trigger in query_lower for trigger in triggers):
            terms.extend(expanded_terms)

    return _dedupe_preserve_order(terms)[:MAX_PRIMARY_TERMS]


def _is_rl_reward_topic(query: str) -> bool:
    query_lower = query.lower()
    return any(
        term in query_lower
        for term in [
            "rlhf",
            "rlvr",
            "human feedback",
            "preference optimization",
            "reward model",
            "verifiable reward",
            "verifiable rewards",
        ]
    )


def _rl_context_terms(query: str) -> list[str]:
    context_terms = [
        "reasoning",
        "reasoning models",
        "language model",
        "large language model",
        "LLM",
    ]

    query_lower = query.lower()
    if "mathematical" in query_lower or "math" in query_lower:
        context_terms.append("mathematical reasoning")

    return _dedupe_preserve_order(context_terms)


def _known_search_phrases(query: str) -> list[str]:
    query_lower = query.lower()
    return [
        phrase
        for phrase in KNOWN_SEARCH_PHRASES
        if phrase in query_lower
    ][:MAX_PRIMARY_TERMS]


def _context_terms(
    normalized_query: str,
    core_terms: list[str],
    limit: int,
) -> list[str]:
    core_words = {
        word
        for term in core_terms
        for word in re.findall(r"[a-z0-9]+", term.lower())
    }

    phrases = [
        phrase
        for phrase in _known_search_phrases(normalized_query)
        if not set(re.findall(r"[a-z0-9]+", phrase)).issubset(core_words)
    ]

    tokens = [
        token
        for token in _informative_tokens(normalized_query, limit=limit * 2)
        if token.lower() not in core_words
    ]

    return _dedupe_preserve_order([*phrases, *tokens])[:limit]


def _informative_tokens(query: str, limit: int) -> list[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", query)
    filtered = [
        token
        for token in tokens
        if token.lower() not in QUERY_STOPWORDS
    ]
    return _dedupe_preserve_order(filtered)[:limit]


def _sanitize_terms(terms: list[str], limit: int) -> list[str]:
    clean_terms = [
        _normalize_query_text(term)
        for term in terms
        if term and _normalize_query_text(term)
    ]
    return _dedupe_preserve_order(clean_terms)[:limit]


def _build_title_abstract_clause(terms: list[str]) -> str:
    formatted_terms = [
        f"ti:{_format_arxiv_value(term)} OR abs:{_format_arxiv_value(term)}"
        for term in terms
        if term.strip()
    ]
    if not formatted_terms:
        return ""

    return f"({' OR '.join(formatted_terms)})"


def _join_required_clauses(clauses: list[str]) -> str:
    return " AND ".join(
        clause
        for clause in clauses
        if clause
    )


def _build_category_clause(categories: list[str]) -> str:
    valid_categories = {
        "cs.AI",
        "cs.CL",
        "cs.LG",
        "cs.IR",
        "stat.ML",
    }
    clean_categories = [
        category
        for category in _dedupe_preserve_order(categories)
        if category in valid_categories
    ]

    if not clean_categories:
        return ""

    return "(" + " OR ".join(f"cat:{category}" for category in clean_categories) + ")"


def _format_arxiv_value(value: str) -> str:
    value = value.strip()
    value = value.replace('"', "")

    if re.fullmatch(r"[A-Za-z0-9_\-]+", value):
        return value

    return f'"{value}"'


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    unique_values = []

    for value in values:
        normalized_value = value.lower()
        if normalized_value in seen:
            continue

        seen.add(normalized_value)
        unique_values.append(value)

    return unique_values


def _parse_arxiv_response(xml_data: bytes) -> list[Paper]:
    """
    Parse arXiv Atom XML response into Paper objects.
    """
    root = ET.fromstring(xml_data)

    ns = {
        "atom": "http://www.w3.org/2005/Atom",
    }

    papers: list[Paper] = []

    for entry in root.findall("atom:entry", ns):
        title = _clean_text(entry.findtext("atom:title", default="", namespaces=ns))
        abstract = _clean_text(entry.findtext("atom:summary", default="", namespaces=ns))
        published_date = entry.findtext("atom:published", default="", namespaces=ns)

        paper_url = entry.findtext("atom:id", default="", namespaces=ns)
        paper_id = paper_url.rstrip("/").split("/")[-1] if paper_url else ""

        authors = [
            _clean_text(author.findtext("atom:name", default="", namespaces=ns))
            for author in entry.findall("atom:author", ns)
        ]

        paper = Paper(
            title=title,
            paper_id=f"arxiv:{paper_id}",
            authors=authors,
            abstract=abstract,
            source="arxiv",
            url=paper_url,
            published_date=published_date[:10],
        )

        papers.append(paper)

    return papers


def _clean_text(text: str) -> str:
    """
    Normalize whitespace from arXiv XML fields.
    """
    return " ".join(text.split())
