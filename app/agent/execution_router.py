from __future__ import annotations

from app.agent.execution_plan import ExecutionPlan, PlanStep
from app.agent.planner_state import PlannerState
from app.agent.request_intent import RequestIntent
from app.retrieval.query_intent import infer_explicit_section_groups_from_query


FAST_BRANCH_CONFIDENCE_THRESHOLD = 0.80


def build_fast_execution_plan(state: PlannerState) -> ExecutionPlan | None:
    """Build a deterministic execution plan for confident, low-ambiguity requests."""

    intent = state.request_intent
    if intent is None or intent.confidence < FAST_BRANCH_CONFIDENCE_THRESHOLD:
        return None

    if intent.task_type == "discovery_only":
        state.execution_branch = "fast_discovery"
        return _discovery_plan(state, intent)

    if intent.task_type == "metadata_lookup":
        state.execution_branch = "fast_metadata"
        return _metadata_plan(state)

    if intent.needs_retrieval:
        paper_source = _paper_id_source(state)
        if paper_source is None:
            return None
        state.execution_branch = "fast_scoped_retrieval"
        return _scoped_retrieval_plan(state, paper_source)

    return None


def _discovery_plan(state: PlannerState, intent: RequestIntent) -> ExecutionPlan:
    topic = intent.topic or state.user_request
    return ExecutionPlan(
        goal=state.user_request,
        strategy="Use the deterministic discovery branch for a metadata-only search.",
        steps=[
            PlanStep(
                step_id="discover",
                kind="tool",
                tool_name="discover_papers",
                arguments={"user_query": topic, "max_results": 5},
                success_condition="selected_paper_ids or candidate_paper_ids is not empty",
                rationale="The classified intent only needs paper metadata.",
            ),
            PlanStep(
                step_id="finish",
                kind="finish",
                answer_task=state.user_request,
                success_condition="paper metadata is available",
                rationale="Discovery produced the requested paper list.",
            ),
        ],
    )


def _metadata_plan(state: PlannerState) -> ExecutionPlan:
    paper_source = _paper_id_source(state)
    if paper_source is not None:
        steps = [
            PlanStep(
                step_id="get_metadata",
                kind="tool",
                tool_name="get_paper_metadata",
                argument_sources={"paper_ids": paper_source},
                success_condition="stored paper metadata is available",
                rationale="The request can be answered from scoped paper metadata.",
            )
        ]
        strategy = "Use scoped metadata from the current paper context."
    else:
        steps = [
            PlanStep(
                step_id="list_papers",
                kind="tool",
                tool_name="list_papers",
                arguments={"limit": 10},
                success_condition="stored paper metadata is available",
                rationale="The request can be answered from the stored paper list.",
            )
        ]
        strategy = "List stored paper metadata without invoking the planner."

    steps.append(
        PlanStep(
            step_id="finish",
            kind="finish",
            answer_task=state.user_request,
            success_condition="metadata artifacts are available",
            rationale="Metadata lookup produced the requested artifacts.",
        )
    )
    return ExecutionPlan(goal=state.user_request, strategy=strategy, steps=steps)


def _scoped_retrieval_plan(
    state: PlannerState,
    paper_source: str,
) -> ExecutionPlan:
    arguments: dict[str, object] = {"query": state.user_request, "top_k": 5}
    section_groups = infer_explicit_section_groups_from_query(state.user_request)
    if section_groups:
        arguments["section_groups"] = list(section_groups)

    return ExecutionPlan(
        goal=state.user_request,
        strategy=(
            "Use scoped retrieval from known runtime paper context before any "
            "open-ended planning."
        ),
        steps=[
            PlanStep(
                step_id="retrieve",
                kind="tool",
                tool_name="retrieve_evidence",
                arguments=arguments,
                argument_sources={"paper_ids": paper_source},
                success_condition="retrieved_evidence is not empty",
                rationale="The runtime state already identifies the relevant papers.",
            ),
            PlanStep(
                step_id="finish",
                kind="finish",
                answer_task=state.user_request,
                success_condition="grounded answer returned from retrieved evidence",
                rationale="Retrieved evidence is sufficient for grounded generation.",
            ),
        ],
    )


def _paper_id_source(state: PlannerState) -> str | None:
    if state.active_paper_ids:
        return "active_paper_ids"
    if state.retrievable_paper_ids:
        return "retrievable_paper_ids"
    if state.known_paper_ids:
        return "known_paper_ids"
    if state.saved_paper_ids:
        return "saved_paper_ids"
    return None
