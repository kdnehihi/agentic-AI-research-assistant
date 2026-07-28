from __future__ import annotations

from typing import Any

from app.agent.state import AgentState
from app.paper_sources import search_paper_sources
from app.paper_sources.query_policy import normalize_paper_source_query


def search_papers(
    state: AgentState,
    *,
    query: str | None = None,
    max_results: int | None = None,
    sources: list[str] | None = None,
) -> dict[str, Any]:
    """Search all configured paper sources behind one workflow-level tool."""

    user_query = query or state.topic
    source_query = normalize_paper_source_query(user_query)
    resolved_max_results = max_results or max(state.max_papers * 10, 20)
    result = search_paper_sources(
        query=source_query,
        max_results=resolved_max_results,
        sources=sources,
        arxiv_query=state.search_plan.arxiv_query if state.search_plan else None,
    )

    papers = result["papers"]
    state.set_candidate_papers(papers)
    for source in result["sources"]:
        state.add_searched_source(source)

    source_results = result["source_results"]
    failed_source_results = [
        source_result
        for source_result in source_results
        if source_result.get("status") != "success"
    ]
    summary = result["summary"]
    source_error_summary = _source_error_summary(failed_source_results)
    if source_error_summary:
        summary = f"{summary} {source_error_summary}"
    return {
        "status": result["status"],
        "num_results": len(papers),
        "summary": summary,
        "search_query": user_query,
        "source_query": source_query,
        "sources": result["sources"],
        "source_results": source_results,
        "source_errors": failed_source_results,
        "candidate_paper_ids": [
            paper.paper_id
            for paper in papers
            if paper.paper_id
        ],
        "candidate_count": len(papers),
        "raw_candidate_count": result["raw_candidate_count"],
    }


def _source_error_summary(source_results: list[dict[str, Any]]) -> str:
    details = []
    for source_result in source_results:
        source = source_result.get("source") or "source"
        summary = source_result.get("summary") or source_result.get("error")
        if not summary:
            continue
        details.append(f"{source}: {summary}")
    if not details:
        return ""
    return "Source errors: " + " ".join(details)
