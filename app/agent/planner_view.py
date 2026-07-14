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
            "status": record.observation.status,
            "summary": record.observation.summary,
            "state_changes": record.observation.state_changes,
        }
        for record in state.tool_history[-3:]
    ]
    return {
        "user_request": state.user_request,
        "known_paper_ids": state.known_paper_ids,
        "saved_paper_ids": state.saved_paper_ids,
        "retrievable_paper_ids": state.retrievable_paper_ids,
        "retrieved_evidence_ids": state.retrieved_evidence_ids,
        "retrieved_evidence_count": len(state.retrieved_evidence_ids),
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

