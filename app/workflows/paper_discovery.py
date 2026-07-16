from __future__ import annotations

from collections.abc import Callable
from typing import Any

from app.agent.state import AgentState
from app.tools.arxiv_tools import search_arxiv_papers
from app.tools.fake_paper_tools import deduplicate_papers
from app.tools.filter_relevant_papers import filter_relevant_papers
from app.tools.knowledge_base_tools import filter_seen_papers
from app.tools.llm_query_planner_tools import plan_arxiv_search_query_with_llm
from app.tools.scoring_tools import rank_papers_by_similarity

WorkflowStep = Callable[..., dict[str, Any]]


def discover_papers_workflow(
    state: AgentState,
    *,
    user_query: str,
    max_results: int | None = None,
    max_selected: int | None = None,
    exclude_seen: bool = True,
    plan_step: WorkflowStep = plan_arxiv_search_query_with_llm,
    search_step: WorkflowStep = search_arxiv_papers,
    filter_seen_step: WorkflowStep = filter_seen_papers,
    dedupe_step: WorkflowStep = deduplicate_papers,
    rank_step: WorkflowStep = rank_papers_by_similarity,
    relevance_step: WorkflowStep = filter_relevant_papers,
) -> dict[str, Any]:
    """Run deterministic paper discovery behind one planner-facing capability."""

    original_topic = state.topic
    original_max_papers = state.max_papers
    state.topic = user_query
    if max_selected is not None:
        state.max_papers = max_selected

    observations: dict[str, dict[str, Any]] = {}
    search_failed = False
    try:
        observations["plan"] = plan_step(state)
        observations["search"] = search_step(
            state,
            query=user_query,
            max_results=max_results or max(state.max_papers * 10, 20),
        )
        if _should_retry_with_rule_based_query(observations):
            state.search_plan = None
            observations["search_fallback"] = search_step(
                state,
                query=user_query,
                max_results=max_results or max(state.max_papers * 10, 20),
            )
            if _search_is_better(
                candidate=observations["search_fallback"],
                original=observations["search"],
            ):
                observations["search"] = observations["search_fallback"]

        if observations["search"].get("status") == "failed":
            search_failed = True
            state.set_candidate_papers([])
            state.set_selected_papers([])
        else:
            if exclude_seen:
                observations["filter_seen"] = filter_seen_step(state)
            observations["deduplicate"] = dedupe_step(state)
            observations["rank"] = rank_step(
                state,
                query=user_query,
                max_papers=state.max_papers,
            )
            observations["relevance"] = relevance_step(state)
    finally:
        state.topic = original_topic
        state.max_papers = original_max_papers if max_selected is not None else state.max_papers

    return _build_discovery_observation(
        state=state,
        observations=observations,
        exclude_seen=exclude_seen,
        search_failed=search_failed,
    )


def _build_discovery_observation(
    *,
    state: AgentState,
    observations: dict[str, dict[str, Any]],
    exclude_seen: bool,
    search_failed: bool,
) -> dict[str, Any]:
    candidate_ids = [
        paper.paper_id
        for paper in state.candidate_papers
        if paper.paper_id
    ]
    selected_ids = [
        paper.paper_id
        for paper in state.selected_papers
        if paper.paper_id
    ]
    excluded_seen_count = (
        observations.get("filter_seen", {}).get("removed_seen", 0)
        if exclude_seen
        else 0
    )

    status = _merge_status(observations.values())
    search_observation = observations.get("search", {})
    summary = (
        "Discovery stopped because arXiv search failed: "
        f"{search_observation.get('summary', 'search failed')}"
        if search_failed
        else (
            f"Discovered {len(candidate_ids)} candidate papers and selected "
            f"{len(selected_ids)} papers."
        )
    )
    return {
        "status": status,
        "planned_query": (
            observations.get("plan", {}).get("search_query")
            or (state.search_plan.arxiv_query if state.search_plan else None)
        ),
        "search_query": search_observation.get("search_query"),
        "failed_step": "search" if search_failed else None,
        "error": search_observation.get("error") if search_failed else None,
        "candidate_paper_ids": candidate_ids,
        "selected_paper_ids": selected_ids,
        "candidate_count": len(candidate_ids),
        "selected_count": len(selected_ids),
        "excluded_seen_count": excluded_seen_count,
        "steps": observations,
        "summary": summary,
    }


def _merge_status(observations) -> str:
    """Merge low-level statuses into one workflow status."""

    statuses = [observation.get("status") for observation in observations]
    if any(status == "failed" for status in statuses):
        return "failed"
    if any(status == "partial_success" for status in statuses):
        return "partial_success"
    if all(status in {"success", "skipped"} for status in statuses):
        return "success"
    return "partial_success"


def _should_retry_with_rule_based_query(observations: dict[str, dict[str, Any]]) -> bool:
    plan_observation = observations.get("plan", {})
    search_observation = observations.get("search", {})
    if plan_observation.get("planner") != "llm":
        return False
    return (
        search_observation.get("status") == "failed"
        or int(search_observation.get("num_results") or 0) == 0
    )


def _search_is_better(*, candidate: dict[str, Any], original: dict[str, Any]) -> bool:
    if candidate.get("status") == "success" and original.get("status") != "success":
        return True
    return int(candidate.get("num_results") or 0) > int(original.get("num_results") or 0)
