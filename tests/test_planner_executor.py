import pytest

from app.agent.executor import ToolExecutor
from app.agent.planner_models import CallToolAction
from app.agent.planner_state import PlannerState
from app.agent.state import AgentState
from app.agent.tool_spec import (
    EmptyArgs,
    RetrieveEvidenceArgs,
    ToolSpec,
)


class FakeRegistry:
    def __init__(self):
        self.calls = []
        self.responses = {}
        self.specs = {
            "retrieve_evidence": ToolSpec(
                name="retrieve_evidence",
                description="Retrieve.",
                args_schema=RetrieveEvidenceArgs,
                read_only=True,
                category="production",
            ),
            "dev_tool": ToolSpec(
                name="dev_tool",
                description="Dev.",
                args_schema=EmptyArgs,
                read_only=True,
                category="development",
            ),
            "admin_tool": ToolSpec(
                name="admin_tool",
                description="Admin.",
                args_schema=EmptyArgs,
                read_only=False,
                destructive=True,
                requires_confirmation=True,
                category="admin",
            ),
        }

    def list_tools(self, category=None):
        return [
            name
            for name, spec in self.specs.items()
            if category is None or spec.category == category
        ]

    def get_tool_spec(self, name):
        return self.specs[name]

    def execute(self, tool_name, state, **kwargs):
        self.calls.append((tool_name, kwargs))
        response = self.responses.get(tool_name)
        if isinstance(response, Exception):
            raise response
        return response or {
            "status": "success",
            "query": kwargs["query"],
            "retrieved": 1,
            "evidence": [{"chunk_id": "c1", "paper_id": "p1", "text": "Evidence"}],
            "summary": "Retrieved 1 evidence chunks.",
        }


def _state():
    return PlannerState(
        user_request="answer",
        runtime_state=AgentState(topic="answer"),
    )


def test_executor_resolves_validates_runs_records_and_updates_state():
    registry = FakeRegistry()
    executor = ToolExecutor(registry=registry)
    state = _state()

    observation = executor.execute(
        state=state,
        decision=CallToolAction(
            tool_name="retrieve_evidence",
            arguments={"query": "q", "top_k": 1},
            decision_summary="Need evidence.",
        ),
    )

    assert observation.status == "success"
    assert registry.calls == [("retrieve_evidence", {"query": "q", "top_k": 1})]
    assert state.step_count == 1
    assert state.tool_history[0].step == 1
    assert state.retrieved_evidence_ids == ["c1"]
    assert state.retrieved_evidence[0]["text"] == "Evidence"


def test_executor_rejects_development_and_admin_tools():
    executor = ToolExecutor(registry=FakeRegistry())
    state = _state()

    dev_obs = executor.execute(
        state=state,
        decision=CallToolAction(
            tool_name="dev_tool",
            arguments={},
            decision_summary="bad",
        ),
    )
    admin_obs = executor.execute(
        state=state,
        decision=CallToolAction(
            tool_name="admin_tool",
            arguments={},
            decision_summary="bad",
        ),
    )

    assert dev_obs.status == "tool_error"
    assert admin_obs.status == "tool_error"
    assert dev_obs.error_type == "tool_not_allowed"
    assert admin_obs.error_type == "tool_not_allowed"


def test_executor_invalid_arguments_do_not_execute_tool():
    registry = FakeRegistry()
    executor = ToolExecutor(registry=registry)
    state = _state()

    observation = executor.execute(
        state=state,
        decision=CallToolAction(
            tool_name="retrieve_evidence",
            arguments={"query": "", "top_k": 0},
            decision_summary="bad args",
        ),
    )

    assert observation.status == "invalid_arguments"
    assert registry.calls == []


def test_executor_catches_unexpected_errors_as_retryable():
    registry = FakeRegistry()
    registry.responses["retrieve_evidence"] = RuntimeError("boom")
    executor = ToolExecutor(registry=registry)
    state = _state()

    observation = executor.execute(
        state=state,
        decision=CallToolAction(
            tool_name="retrieve_evidence",
            arguments={"query": "q"},
            decision_summary="try",
        ),
    )

    assert observation.status == "tool_error"
    assert observation.retryable is True


def test_executor_blocks_duplicate_successful_call_but_allows_retry_after_failure():
    registry = FakeRegistry()
    executor = ToolExecutor(registry=registry)
    state = _state()
    decision = CallToolAction(
        tool_name="retrieve_evidence",
        arguments={"query": "q"},
        decision_summary="Need evidence.",
    )

    first = executor.execute(state=state, decision=decision)
    second = executor.execute(state=state, decision=decision)

    assert first.status == "success"
    assert second.status == "no_progress"

    retry_state = _state()
    registry.responses["retrieve_evidence"] = {
        "status": "failed",
        "error_type": "paper_not_retrievable",
        "missing_paper_ids": ["p1"],
        "summary": "missing",
    }
    failed = executor.execute(state=retry_state, decision=decision)
    registry.responses["retrieve_evidence"] = None
    retried = executor.execute(state=retry_state, decision=decision)

    assert failed.status == "prerequisite_missing"
    assert retried.status == "success"

