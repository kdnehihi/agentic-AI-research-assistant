from __future__ import annotations

from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request

from app.paper_sources.base import (
    PaperCandidate,
    PaperSourceProvenance,
    PaperSourceResult,
    PaperSourceSearchRequest,
)
from app.tools import arxiv_tools


class ArxivSource:
    """Paper source adapter for the arXiv Atom API."""

    name = "arxiv"

    def search(self, request: PaperSourceSearchRequest) -> PaperSourceResult:
        arxiv_query = request.arxiv_query or arxiv_tools._build_arxiv_search_query(
            request.query
        )
        params = {
            "search_query": arxiv_query,
            "start": 0,
            "max_results": request.max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        url = f"{arxiv_tools.ARXIV_API_URL}?{urlencode(params)}"
        http_request = Request(
            url,
            headers={"User-Agent": arxiv_tools.ARXIV_USER_AGENT},
        )

        try:
            with arxiv_tools.urlopen(
                http_request,
                timeout=arxiv_tools._arxiv_timeout_seconds(),
            ) as response:
                xml_data = response.read()
        except HTTPError as exc:
            return PaperSourceResult(
                source=self.name,
                status="failed",
                query=arxiv_query,
                error=str(exc),
                summary=arxiv_tools._arxiv_http_error_summary(exc),
            )
        except Exception as exc:
            return PaperSourceResult(
                source=self.name,
                status="failed",
                query=arxiv_query,
                error=str(exc),
                summary="Failed to fetch papers from arXiv.",
            )

        try:
            papers = arxiv_tools._parse_arxiv_response(xml_data)
        except Exception as exc:
            return PaperSourceResult(
                source=self.name,
                status="failed",
                query=arxiv_query,
                error=str(exc),
                summary="Failed to parse arXiv response.",
            )

        candidates = [_candidate_from_paper(paper) for paper in papers]

        return PaperSourceResult(
            source=self.name,
            status="success",
            candidates=candidates,
            query=arxiv_query,
            summary=(
                f"Found {len(candidates)} papers from arXiv for query: "
                f"{request.query}"
            ),
        )


def _arxiv_id_from_paper_id(paper_id: str | None) -> str | None:
    if not paper_id:
        return None
    return paper_id.removeprefix("arxiv:")


def _candidate_from_paper(paper) -> PaperCandidate:
    arxiv_id = _arxiv_id_from_paper_id(paper.paper_id)
    return PaperCandidate(
        title=paper.title,
        paper_id=paper.paper_id,
        authors=paper.authors,
        source="arxiv",
        url=paper.url,
        abstract=paper.abstract,
        published_date=paper.published_date,
        arxiv_id=arxiv_id,
        external_ids={"ArXiv": arxiv_id} if arxiv_id else {},
        provenance=[
            PaperSourceProvenance(
                source="arxiv",
                source_paper_id=arxiv_id,
                url=paper.url,
            )
        ],
    )
