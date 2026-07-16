import pytest
from pydantic import ValidationError

from app.agent.planner_models import (
    CallToolAction,
    FinishAction,
    PlannerDecisionAdapter,
    ToolObservation,
)
from app.agent.planner_state import PlannerState, ToolExecutionRecord
from app.agent.planner_view import build_planner_view
from app.agent.state import AgentState


def test_call_tool_action_validates_shape():
    action = CallToolAction(
        tool_name="retrieve_evidence",
        arguments={"query": "q"},
        decision_summary="Need evidence.",
    )

    assert action.action == "call_tool"
    assert action.arguments["query"] == "q"


def test_finish_action_validates_shape():
    action = FinishAction(
        answer_task="Answer from evidence.",
        decision_summary="Enough evidence exists.",
    )

    assert action.action == "finish"


def test_unknown_action_is_rejected():
    with pytest.raises(ValidationError):
        PlannerDecisionAdapter.validate_python(
            {"action": "run_code", "decision_summary": "bad"}
        )


def test_extra_fields_and_empty_tool_name_are_rejected():
    with pytest.raises(ValidationError):
        CallToolAction(
            tool_name="",
            arguments={},
            decision_summary="bad",
            extra_field=True,
        )


def test_planner_view_excludes_runtime_state_object():
    state = PlannerState(
        user_request="List papers",
        runtime_state=AgentState(topic="papers"),
        known_paper_ids=["p1"],
    )

    view = build_planner_view(state)

    assert "runtime_state" not in view
    assert view["known_paper_ids"] == ["p1"]
    assert view["steps_remaining"] == 8


def test_planner_view_includes_recent_tool_metadata_for_planning():
    state = PlannerState(
        user_request="What is missing?",
        runtime_state=AgentState(topic="papers"),
        tool_history=[
            ToolExecutionRecord(
                step=1,
                decision=CallToolAction(
                    tool_name="retrieve_evidence",
                    arguments={"query": "What is missing?", "top_k": 5},
                    decision_summary="Probe KB first.",
                ),
                observation=ToolObservation(
                    tool_name="retrieve_evidence",
                    status="success",
                    summary="Retrieved 0 evidence chunks.",
                    result={"retrieved": 0},
                ),
                call_fingerprint="fp",
                latency_ms=12.3,
            )
        ],
    )

    view = build_planner_view(state)

    assert view["kb_probe_attempted"] is True
    assert view["last_retrieval_count"] == 0
    assert view["recent_history"][0]["arguments"]["top_k"] == 5
    assert view["recent_history"][0]["result_counts"] == {"retrieved": 0}
    assert view["recent_history"][0]["latency_ms"] == 12.3
