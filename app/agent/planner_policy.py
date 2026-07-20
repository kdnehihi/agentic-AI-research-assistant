from __future__ import annotations

from app.agent.planner_models import CallToolAction, FinishAction, PlannerDecision
from app.agent.planner_state import PlannerState
from app.agent.request_intent import RequestIntent


def choose_policy_action(state: PlannerState) -> PlannerDecision | None:
    """Return a deterministic action from the LLM-classified request intent."""

    intent = state.request_intent
    if intent is None:
        return None

    finish = _finish_when_artifacts_are_enough(state, intent)
    if finish is not None:
        return finish

    plan_action = _next_plan_action(state)
    if plan_action is not None:
        return plan_action

    if state.step_count > 0:
        return None
    if _should_probe_existing_kb(intent):
        return CallToolAction(
            tool_name="retrieve_evidence",
            arguments={"query": state.user_request, "top_k": 5},
            decision_summary=(
                "The request intent requires paper-content evidence, so probe "
                "indexed knowledge-base evidence before discovering new papers."
            ),
        )
    return None


def _finish_when_artifacts_are_enough(
    state: PlannerState,
    intent: RequestIntent,
) -> FinishAction | None:
    """Stop once the intent's declared finish condition has been satisfied."""

    if not _finish_condition_satisfied(state, intent):
        return None
    return FinishAction(
        answer_task=state.user_request,
        decision_summary=(
            "The classified request intent has enough artifacts to finish."
        ),
    )


def _finish_condition_satisfied(state: PlannerState, intent: RequestIntent) -> bool:
    if intent.finish_condition == "paper_metadata":
        return bool(
            state.known_paper_ids
            or state.saved_paper_ids
            or state.retrievable_paper_ids
        )
    if intent.finish_condition == "stored_metadata":
        return bool(state.saved_paper_ids or state.known_paper_ids)
    if intent.finish_condition == "retrieved_evidence":
        return bool(
            state.retrieved_evidence_ids
            or state.report_available
            or state.summary_paper_ids
            or state.runtime_state.report
            or state.runtime_state.paper_summaries
        )
    if intent.finish_condition == "paper_summary":
        return bool(state.summary_paper_ids or state.runtime_state.paper_summaries)
    if intent.finish_condition == "report":
        return bool(state.report_available or state.runtime_state.report)
    return False


def _should_probe_existing_kb(intent: RequestIntent) -> bool:
    if not intent.needs_retrieval:
        return False
    return intent.probe_existing_kb_first


def _next_plan_action(state: PlannerState) -> PlannerDecision | None:
    plan = state.execution_plan
    if plan is None:
        return None
    for step in plan.steps:
        if step.status != "pending":
            continue
        if step.kind == "finish":
            state.current_plan_step_id = step.step_id
            return FinishAction(
                answer_task=step.answer_task or state.user_request,
                decision_summary=(
                    f"Finish from execution plan step '{step.step_id}'."
                ),
            )
        if step.kind != "tool" or not step.tool_name:
            continue
        arguments = _resolved_tool_arguments(
            state,
            tool_name=step.tool_name,
            arguments=step.arguments,
            argument_sources=step.argument_sources,
        )
        if arguments is None:
            return None
        state.current_plan_step_id = step.step_id
        return CallToolAction(
            tool_name=step.tool_name,
            arguments=arguments,
            decision_summary=f"Execute planned step '{step.step_id}'.",
        )
    return None


def _resolved_tool_arguments(
    state: PlannerState,
    *,
    tool_name: str,
    arguments: dict,
    argument_sources: dict,
) -> dict | None:
    resolved = dict(arguments)
    for argument_name, source_name in argument_sources.items():
        value = _source_value(state, source_name)
        if not value:
            return None
        resolved[argument_name] = value

    if tool_name == "discover_papers":
        resolved.setdefault("user_query", _topic_or_request(state))
    elif tool_name == "retrieve_evidence":
        resolved.setdefault("query", state.user_request)
    elif tool_name == "ensure_papers_retrievable":
        resolved.setdefault(
            "paper_ids",
            state.known_paper_ids
            or state.saved_paper_ids
            or state.retrievable_paper_ids
            or state.active_paper_ids,
        )
        if not resolved.get("paper_ids"):
            return None
    return resolved


def _source_value(state: PlannerState, source_name: str) -> list[str]:
    return list(getattr(state, source_name, []) or [])


def _topic_or_request(state: PlannerState) -> str:
    if state.request_intent is not None and state.request_intent.topic:
        return state.request_intent.topic
    return state.user_request
