# ArXiv API Tools
# The file contains tools for interacting with the arXiv API, including searching for papers and retrieving paper details.
from __future__ import annotations
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen
import xml.etree.ElementTree as ET

from app.agent.state import AgentState, Paper

ARXIV_API_URL = "http://export.arxiv.org/api/query"
DEFAULT_CANDIDATE_MULTIPLIER = 10
MIN_CANDIDATE_RESULTS = 20


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
    query = query or state.topic
    max_results = max_results or max(
        state.max_papers * DEFAULT_CANDIDATE_MULTIPLIER,
        MIN_CANDIDATE_RESULTS,
    )
    search_query = _build_arxiv_search_query(query)

    params = {
        "search_query": search_query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }

    url = f"{ARXIV_API_URL}?{urlencode(params)}"

    try: #mở kết nối tới arXiv, nếu quá 20 giây không có response thì timeout, dùng with để tự đóng connection sau khi xong
        with urlopen(url, timeout=20) as response:
            xml_data = response.read()
    except Exception as exc:
        return {
            "status": "error",
            "num_results": 0,
            "summary": "Failed to fetch papers from arXiv.",
            "error": str(exc),
        }

    try:
        papers = _parse_arxiv_response(xml_data)
    except Exception as exc:
        return {
            "status": "error",
            "num_results": 0,
            "summary": "Failed to parse arXiv response.",
            "error": str(exc),
        }

    state.set_candidate_papers(papers)

    return {
        "status": "success",
        "num_results": len(papers),
        "summary": f"Found {len(papers)} papers from arXiv for query: {query}",
    }


def _build_arxiv_search_query(query: str) -> str:
    """
    Build an arXiv query string.

    For RLHF/RLVR topics, expand shorthand terms so candidate retrieval is not
    limited to exact acronym matches or broad recent "reasoning" papers.
    """
    query_lower = query.lower()
    is_rl_topic = any(
        term in query_lower
        for term in [
            "rlhf",
            "rlvr",
            "human feedback",
            "verifiable reward",
        ]
    )

    if not is_rl_topic:
        return f"all:{query}"

    return (
        "("
        "all:RLHF OR "
        "all:RLVR OR "
        'all:"reinforcement learning from human feedback" OR '
        'all:"verifiable rewards" OR '
        'all:"preference optimization" OR '
        'all:"reward model"'
        ") AND ("
        "all:reasoning OR "
        'all:"language model" OR '
        'all:"large language model"'
        ")"
    )


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
