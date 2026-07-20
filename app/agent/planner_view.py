from __future__ import annotations

from typing import Any

from app.agent.planner_state import PlannerState


def build_planner_view(state: PlannerState) -> dict[str, Any]:
    """Build a compact, LLM-safe state view for the planner."""

    runtime = state.runtime_state
    latest = (
        state.latest_observation.model_dump(mode="json")
        if state.latest_observation
        else None
    )
    recent_history = [
        {
            "step": record.step,
            "tool_name": record.decision.tool_name,
            "arguments": record.decision.arguments,
            "decision_summary": record.decision.decision_summary,
            "status": record.observation.status,
            "summary": record.observation.summary,
            "result_counts": _result_counts(record.observation.result),
            "state_changes": record.observation.state_changes,
            "latency_ms": record.latency_ms,
        }
        for record in state.tool_history[-3:]
    ]
    return {
        "user_request": state.user_request,
        "conversation": {
            "thread_id": state.thread_id,
            "run_id": state.run_id,
            "current_user_message_id": state.current_user_message_id,
            "summary": state.conversation_summary,
            "recent_messages": state.recent_messages,
            "active_paper_ids": state.active_paper_ids,
        },
        "request_intent": (
            state.request_intent.model_dump(mode="json")
            if state.request_intent is not None
            else None
        ),
        "execution_plan": (
            state.execution_plan.model_dump(mode="json")
            if state.execution_plan is not None
            else None
        ),
        "current_plan_step_id": state.current_plan_step_id,
        "known_paper_ids": state.known_paper_ids,
        "saved_paper_ids": state.saved_paper_ids,
        "retrievable_paper_ids": state.retrievable_paper_ids,
        "retrieved_evidence_ids": state.retrieved_evidence_ids,
        "retrieved_evidence_count": len(state.retrieved_evidence_ids),
        "kb_probe_attempted": any(
            record.decision.tool_name == "retrieve_evidence"
            and not record.decision.arguments.get("paper_ids")
            for record in state.tool_history
        ),
        "last_retrieval_count": _last_retrieval_count(state),
        "summary_paper_ids": state.summary_paper_ids,
        "report_available": state.report_available or runtime.report is not None,
        "selected_paper_ids": [
            paper.paper_id for paper in runtime.selected_papers if paper.paper_id
        ],
        "candidate_paper_count": len(runtime.candidate_papers),
        "selected_paper_count": len(runtime.selected_papers),
        "latest_observation": latest,
        "recent_history": recent_history,
        "step_count": state.step_count,
        "max_steps": state.max_steps,
        "steps_remaining": max(state.max_steps - state.step_count, 0),
    }


def _result_counts(result: dict[str, Any]) -> dict[str, Any]:
    counts = {}
    for key in (
        "retrieved",
        "candidate_count",
        "selected_count",
        "count",
        "processed",
        "failed",
    ):
        if key in result:
            counts[key] = result[key]
    return counts


def _last_retrieval_count(state: PlannerState) -> int | None:
    for record in reversed(state.tool_history):
        if record.decision.tool_name != "retrieve_evidence":
            continue
        retrieved = record.observation.result.get("retrieved")
        return int(retrieved) if retrieved is not None else None
    return None
