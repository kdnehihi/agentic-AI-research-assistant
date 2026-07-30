from __future__ import annotations

import re
import os
from concurrent.futures import (
    ThreadPoolExecutor,
    TimeoutError as FuturesTimeoutError,
    as_completed,
)
from typing import Any

from app.paper_sources.arxiv import ArxivSource
from app.paper_sources.base import (
    PaperCandidate,
    PaperSource,
    PaperSourceProvenance,
    PaperSourceResult,
    PaperSourceSearchRequest,
)
from app.paper_sources.semantic_scholar import SemanticScholarSource
from app.paper_sources.query_policy import (
    prefers_recent_results,
    requested_publication_years,
)


DEFAULT_SOURCE_NAMES = ("arxiv", "semantic_scholar")
SUPPORTED_SOURCE_NAMES = (*DEFAULT_SOURCE_NAMES, "acl", "openreview", "europe_pmc")
NLP_HINTS = ("acl", "nlp", "natural language", "linguistic", "translation")
BIOMEDICAL_HINTS = ("biomedical", "clinical", "medicine", "protein", "genomic")
OPENREVIEW_HINTS = ("openreview", "iclr", "neurips", "conference review")
MIN_SOURCE_CANDIDATE_POOL = 20
SOURCE_CANDIDATE_MULTIPLIER = 4
MAX_SOURCE_CANDIDATE_POOL = 100
PAPER_SOURCE_TIMEOUT_SECONDS = 12.0


def search_paper_sources(
    *,
    query: str,
    max_results: int = 20,
    sources: list[str] | None = None,
    arxiv_query: str | None = None,
    adapters: list[PaperSource] | None = None,
) -> dict[str, Any]:
    """Search selected source adapters, then merge and deduplicate candidates."""

    source_names = select_source_names(query=query, requested_sources=sources)
    selected_adapters = adapters or build_default_paper_sources(source_names)
    final_max_results = max(max_results, 1)
    prefer_recent = prefers_recent_results(query)
    requested_years = requested_publication_years(query)
    request = PaperSourceSearchRequest(
        query=query,
        max_results=_source_candidate_pool_size(
            final_max_results,
            prefer_recent=prefer_recent,
        ),
        arxiv_query=arxiv_query,
        publication_years=requested_years,
    )
    results = _search_adapters(selected_adapters, request)
    candidates = merge_and_deduplicate_candidates(
        [
            candidate
            for result in results
            if result.status == "success"
            for candidate in result.candidates
        ]
    )
    candidates = _filter_by_publication_years(candidates, requested_years)
    ranked = sorted(
        candidates,
        key=lambda candidate: _candidate_rank_key(
            candidate,
            prefer_recent=prefer_recent,
        ),
    )[:final_max_results]

    return {
        "status": _merge_status(results),
        "query": query,
        "sources": [adapter.name for adapter in selected_adapters],
        "source_results": [result.model_dump(mode="json") for result in results],
        "candidates": [candidate.model_dump(mode="json") for candidate in ranked],
        "papers": [candidate.to_paper() for candidate in ranked],
        "candidate_count": len(ranked),
        "raw_candidate_count": sum(len(result.candidates) for result in results),
        "requested_publication_years": requested_years,
        "summary": (
            f"Searched {len(selected_adapters)} paper sources and returned "
            f"{len(ranked)} deduplicated candidates."
        ),
    }


def select_source_names(
    *,
    query: str,
    requested_sources: list[str] | None = None,
) -> list[str]:
    """Choose paper sources while keeping unsupported future adapters out."""

    if requested_sources:
        return _normalize_source_names(requested_sources)

    lowered = query.lower()
    selected = list(DEFAULT_SOURCE_NAMES)
    if any(hint in lowered for hint in NLP_HINTS):
        selected.append("acl")
    if any(hint in lowered for hint in BIOMEDICAL_HINTS):
        selected.append("europe_pmc")
    if any(hint in lowered for hint in OPENREVIEW_HINTS):
        selected.append("openreview")
    return _available_source_names(selected)


def build_default_paper_sources(source_names: list[str] | None = None) -> list[PaperSource]:
    """Build adapters for currently implemented sources."""

    names = _available_source_names(source_names or list(DEFAULT_SOURCE_NAMES))
    adapters: list[PaperSource] = []
    for name in names:
        if name == "arxiv":
            adapters.append(ArxivSource())
        elif name == "semantic_scholar":
            adapters.append(SemanticScholarSource())
    return adapters


def merge_and_deduplicate_candidates(
    candidates: list[PaperCandidate],
) -> list[PaperCandidate]:
    """Merge candidates by DOI, arXiv id, Semantic Scholar id, or title/authors."""

    merged_by_key: dict[str, PaperCandidate] = {}
    order: list[str] = []
    for candidate in candidates:
        keys = _dedupe_keys(candidate)
        existing_key = next((key for key in keys if key in merged_by_key), None)
        if existing_key is None:
            canonical_key = keys[0]
            for key in keys:
                merged_by_key[key] = candidate
            order.append(canonical_key)
            continue
        merged_candidate = _merge_candidates(merged_by_key[existing_key], candidate)
        for key in [*keys, *_dedupe_keys(merged_candidate)]:
            merged_by_key[key] = merged_candidate
    return [merged_by_key[key] for key in order]


def _search_adapters(
    adapters: list[PaperSource],
    request: PaperSourceSearchRequest,
) -> list[PaperSourceResult]:
    if not adapters:
        return []
    results_by_source: dict[str, PaperSourceResult] = {}
    pool = ThreadPoolExecutor(max_workers=min(len(adapters), 4))
    futures = {
        pool.submit(adapter.search, request): adapter.name
        for adapter in adapters
    }
    try:
        for future in as_completed(
            futures,
            timeout=_paper_source_timeout_seconds(),
        ):
            source_name = futures[future]
            try:
                results_by_source[source_name] = future.result()
            except Exception as exc:
                results_by_source[source_name] = PaperSourceResult(
                    source=source_name,
                    status="failed",
                    query=request.query,
                    error=str(exc),
                    summary=f"{source_name} search failed unexpectedly.",
                )
    except FuturesTimeoutError:
        pass
    finally:
        for future, source_name in futures.items():
            if source_name in results_by_source:
                continue
            future.cancel()
            results_by_source[source_name] = PaperSourceResult(
                source=source_name,
                status="failed",
                query=request.query,
                error=(
                    f"Timed out after {_paper_source_timeout_seconds():.1f} seconds."
                ),
                summary=f"{source_name} search timed out.",
            )
        pool.shutdown(wait=False, cancel_futures=True)
    return [
        results_by_source[adapter.name]
        for adapter in adapters
        if adapter.name in results_by_source
    ]


def _source_candidate_pool_size(max_results: int, *, prefer_recent: bool) -> int:
    if prefer_recent:
        expanded = max(max_results, MIN_SOURCE_CANDIDATE_POOL)
    else:
        expanded = max(
            max_results * SOURCE_CANDIDATE_MULTIPLIER,
            MIN_SOURCE_CANDIDATE_POOL,
        )
    return min(expanded, MAX_SOURCE_CANDIDATE_POOL)


def _paper_source_timeout_seconds() -> float:
    raw_timeout = os.getenv("PAPER_SOURCE_TIMEOUT_SECONDS")
    if raw_timeout is None:
        return PAPER_SOURCE_TIMEOUT_SECONDS
    try:
        return max(float(raw_timeout), 0.01)
    except ValueError:
        return PAPER_SOURCE_TIMEOUT_SECONDS


def _merge_status(results: list[PaperSourceResult]) -> str:
    if not results:
        return "failed"
    statuses = [result.status for result in results]
    if all(status == "success" for status in statuses):
        return "success"
    if any(status == "success" for status in statuses):
        return "partial_success"
    return "failed"


def _normalize_source_names(source_names: list[str]) -> list[str]:
    normalized = []
    for source_name in source_names:
        clean_name = source_name.strip().lower().replace("-", "_")
        if clean_name in SUPPORTED_SOURCE_NAMES and clean_name not in normalized:
            normalized.append(clean_name)
    return _available_source_names(normalized)


def _available_source_names(source_names: list[str]) -> list[str]:
    available = []
    for source_name in source_names:
        if source_name in {"arxiv", "semantic_scholar"} and source_name not in available:
            available.append(source_name)
    return available


def _dedupe_keys(candidate: PaperCandidate) -> list[str]:
    keys: list[str] = []
    doi = _normalized_doi(candidate.doi or candidate.external_ids.get("DOI"))
    if doi:
        keys.append(f"doi:{doi}")
    arxiv_id = _normalized_arxiv_id(
        candidate.arxiv_id or candidate.external_ids.get("ArXiv")
    )
    if arxiv_id:
        keys.append(f"arxiv:{arxiv_id}")
    if candidate.semantic_scholar_id:
        keys.append(f"semantic_scholar:{candidate.semantic_scholar_id.lower()}")
    keys.append(f"title:{_normalized_title_author_key(candidate)}")
    return keys


def _merge_candidates(
    left: PaperCandidate,
    right: PaperCandidate,
) -> PaperCandidate:
    primary = _choose_primary_candidate(left, right)
    secondary = right if primary is left else left
    external_ids = {**secondary.external_ids, **primary.external_ids}
    provenance = _merge_provenance(primary.provenance, secondary.provenance)
    authors = _merge_strings(primary.authors, secondary.authors)
    return primary.model_copy(
        update={
            "authors": authors,
            "abstract": primary.abstract or secondary.abstract,
            "doi": primary.doi or secondary.doi,
            "arxiv_id": primary.arxiv_id or secondary.arxiv_id,
            "semantic_scholar_id": (
                primary.semantic_scholar_id or secondary.semantic_scholar_id
            ),
            "external_ids": external_ids,
            "venue": primary.venue or secondary.venue,
            "citation_count": (
                primary.citation_count
                if primary.citation_count is not None
                else secondary.citation_count
            ),
            "open_access_pdf_url": (
                primary.open_access_pdf_url or secondary.open_access_pdf_url
            ),
            "provenance": provenance,
        }
    )


def _choose_primary_candidate(
    left: PaperCandidate,
    right: PaperCandidate,
) -> PaperCandidate:
    return min(left, right, key=_primary_candidate_key)


def _primary_candidate_key(candidate: PaperCandidate) -> tuple[int, int, int, int]:
    source_priority = 0 if candidate.source == "arxiv" else 1
    has_arxiv = 0 if _normalized_arxiv_id(candidate.arxiv_id) else 1
    has_doi = 0 if _normalized_doi(candidate.doi) else 1
    abstract_len = -(len(candidate.abstract or ""))
    return source_priority, has_arxiv, has_doi, abstract_len


def _candidate_rank_key(
    candidate: PaperCandidate,
    *,
    prefer_recent: bool = False,
) -> tuple[float, int, str]:
    if prefer_recent:
        return -_published_date_sort_value(candidate.published_date), 0, candidate.title.lower()
    score = float(candidate.citation_count or 0)
    return -score, -len(candidate.abstract or ""), candidate.title.lower()


def _filter_by_publication_years(
    candidates: list[PaperCandidate],
    requested_years: list[int],
) -> list[PaperCandidate]:
    if not requested_years:
        return candidates
    allowed_years = set(requested_years)
    return [
        candidate
        for candidate in candidates
        if _publication_year(candidate.published_date) in allowed_years
    ]


def _publication_year(value: str | None) -> int | None:
    if not value:
        return None
    try:
        return int(value[:4])
    except ValueError:
        return None


def _published_date_sort_value(value: str | None) -> float:
    if not value:
        return 0.0
    try:
        parts = [int(part) for part in value.split("-")[:3]]
    except ValueError:
        return 0.0
    while len(parts) < 3:
        parts.append(1)
    year, month, day = parts[:3]
    return float(year * 10000 + month * 100 + day)


def _merge_strings(left: list[str], right: list[str]) -> list[str]:
    merged: list[str] = []
    for value in [*left, *right]:
        if value and value not in merged:
            merged.append(value)
    return merged


def _merge_provenance(
    left: list[PaperSourceProvenance],
    right: list[PaperSourceProvenance],
) -> list[PaperSourceProvenance]:
    merged: list[PaperSourceProvenance] = []
    seen = set()
    for provenance in [*left, *right]:
        key = (provenance.source, provenance.source_paper_id, provenance.url)
        if key in seen:
            continue
        seen.add(key)
        merged.append(provenance)
    return merged


def _normalized_doi(value: str | None) -> str | None:
    if not value:
        return None
    return value.strip().lower().removeprefix("doi:")


def _normalized_arxiv_id(value: str | None) -> str | None:
    if not value:
        return None
    clean = value.strip().lower().removeprefix("arxiv:")
    return re.sub(r"v\d+$", "", clean)


def _normalized_title_author_key(candidate: PaperCandidate) -> str:
    title = re.sub(r"[^a-z0-9]+", " ", candidate.title.lower()).strip()
    first_author = candidate.authors[0].lower() if candidate.authors else ""
    first_author = re.sub(r"[^a-z0-9]+", " ", first_author).strip()
    return f"{title}|{first_author}"
