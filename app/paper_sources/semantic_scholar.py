from __future__ import annotations

import json
import os
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from dotenv import load_dotenv

from app.paper_sources.base import (
    PaperCandidate,
    PaperSourceProvenance,
    PaperSourceResult,
    PaperSourceSearchRequest,
)


SEMANTIC_SCHOLAR_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
SEMANTIC_SCHOLAR_FIELDS = ",".join(
    [
        "title",
        "abstract",
        "authors",
        "year",
        "publicationDate",
        "url",
        "venue",
        "citationCount",
        "externalIds",
        "openAccessPdf",
    ]
)
SEMANTIC_SCHOLAR_TIMEOUT_SECONDS = 8
SEMANTIC_SCHOLAR_USER_AGENT = "agentic-ai-research-assistant/0.1"


class SemanticScholarSource:
    """Paper source adapter for the Semantic Scholar Graph API."""

    name = "semantic_scholar"

    def __init__(self, api_key: str | None = None, *, load_env: bool = True) -> None:
        if load_env:
            load_dotenv()
        self.api_key = api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY")

    def search(self, request: PaperSourceSearchRequest) -> PaperSourceResult:
        params = {
            "query": request.query.replace("-", " "),
            "limit": request.max_results,
            "fields": SEMANTIC_SCHOLAR_FIELDS,
        }
        year_filter = _semantic_scholar_year_filter(request.publication_years)
        if year_filter:
            params["year"] = year_filter
        url = f"{SEMANTIC_SCHOLAR_SEARCH_URL}?{urlencode(params)}"
        headers = {"User-Agent": SEMANTIC_SCHOLAR_USER_AGENT}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        http_request = Request(url, headers=headers)

        try:
            with urlopen(
                http_request,
                timeout=_semantic_scholar_timeout_seconds(),
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            return PaperSourceResult(
                source=self.name,
                status="failed",
                query=request.query,
                error=str(exc),
                summary=_semantic_scholar_http_error_summary(exc),
            )
        except Exception as exc:
            return PaperSourceResult(
                source=self.name,
                status="failed",
                query=request.query,
                error=str(exc),
                summary="Failed to fetch papers from Semantic Scholar.",
            )

        try:
            candidates = _parse_semantic_scholar_response(payload)
        except Exception as exc:
            return PaperSourceResult(
                source=self.name,
                status="failed",
                query=request.query,
                error=str(exc),
                summary="Failed to parse Semantic Scholar response.",
            )

        return PaperSourceResult(
            source=self.name,
            status="success",
            candidates=candidates,
            query=request.query,
            summary=(
                f"Found {len(candidates)} papers from Semantic Scholar for "
                f"query: {request.query}"
            ),
        )


def _parse_semantic_scholar_response(payload: dict[str, Any]) -> list[PaperCandidate]:
    papers = payload.get("data") or []
    candidates: list[PaperCandidate] = []
    for paper in papers:
        if not isinstance(paper, dict):
            continue
        title = _clean_text(str(paper.get("title") or ""))
        paper_id = str(paper.get("paperId") or "")
        if not title or not paper_id:
            continue

        external_ids = _clean_external_ids(paper.get("externalIds") or {})
        arxiv_id = external_ids.get("ArXiv")
        doi = external_ids.get("DOI")
        open_access_pdf = paper.get("openAccessPdf") or {}
        open_access_pdf_url = (
            open_access_pdf.get("url")
            if isinstance(open_access_pdf, dict)
            else None
        )
        url = (
            _semantic_scholar_url(paper)
            or _arxiv_abs_url(arxiv_id)
            or open_access_pdf_url
            or ""
        )

        candidates.append(
            PaperCandidate(
                title=title,
                paper_id=(
                    f"arxiv:{arxiv_id}"
                    if arxiv_id
                    else f"semantic_scholar:{paper_id}"
                ),
                authors=_authors_from_semantic_scholar(paper),
                source="semantic_scholar",
                url=url,
                abstract=_clean_optional_text(paper.get("abstract")),
                published_date=_published_date(paper),
                doi=doi,
                arxiv_id=arxiv_id,
                semantic_scholar_id=paper_id,
                external_ids=external_ids,
                venue=_clean_optional_text(paper.get("venue")),
                citation_count=_int_or_none(paper.get("citationCount")),
                open_access_pdf_url=open_access_pdf_url,
                provenance=[
                    PaperSourceProvenance(
                        source="semantic_scholar",
                        source_paper_id=paper_id,
                        url=url or None,
                        raw={
                            "externalIds": external_ids,
                            "venue": paper.get("venue"),
                            "citationCount": paper.get("citationCount"),
                        },
                    )
                ],
            )
        )
    return candidates


def _semantic_scholar_http_error_summary(exc: HTTPError) -> str:
    if exc.code == 429:
        return (
            "Semantic Scholar rate-limited the request. Set "
            "SEMANTIC_SCHOLAR_API_KEY for higher limits or retry later."
        )
    return f"Semantic Scholar returned HTTP {exc.code}."


def _semantic_scholar_year_filter(years: list[int]) -> str | None:
    """Return a Semantic Scholar year filter when the request has exact years."""

    unique_years = sorted({year for year in years if 1900 <= year <= 2100})
    if not unique_years:
        return None
    if len(unique_years) == 1:
        return str(unique_years[0])
    if unique_years == list(range(unique_years[0], unique_years[-1] + 1)):
        return f"{unique_years[0]}-{unique_years[-1]}"
    return None


def _semantic_scholar_timeout_seconds() -> float:
    raw_timeout = os.getenv("SEMANTIC_SCHOLAR_TIMEOUT_SECONDS")
    if not raw_timeout:
        return SEMANTIC_SCHOLAR_TIMEOUT_SECONDS
    try:
        return max(float(raw_timeout), 1.0)
    except ValueError:
        return SEMANTIC_SCHOLAR_TIMEOUT_SECONDS


def _semantic_scholar_url(paper: dict[str, Any]) -> str | None:
    url = paper.get("url")
    return str(url) if url else None


def _arxiv_abs_url(arxiv_id: str | None) -> str | None:
    if not arxiv_id:
        return None
    return f"https://arxiv.org/abs/{arxiv_id}"


def _authors_from_semantic_scholar(paper: dict[str, Any]) -> list[str]:
    authors = []
    for author in paper.get("authors") or []:
        if isinstance(author, dict) and author.get("name"):
            authors.append(_clean_text(str(author["name"])))
    return authors


def _published_date(paper: dict[str, Any]) -> str | None:
    publication_date = paper.get("publicationDate")
    if publication_date:
        return str(publication_date)
    year = paper.get("year")
    if year:
        return str(year)
    return None


def _clean_external_ids(value: dict[str, Any]) -> dict[str, str]:
    return {
        str(key): str(item)
        for key, item in value.items()
        if item is not None and str(item).strip()
    }


def _clean_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = _clean_text(str(value))
    return text or None


def _clean_text(text: str) -> str:
    return " ".join(text.split())


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
