import pytest
from pydantic import ValidationError

from app.agent.planner_models import (
    CallToolAction,
    FinishAction,
    PlannerDecisionAdapter,
)
from app.agent.planner_state import PlannerState
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

