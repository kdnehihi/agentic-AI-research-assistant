from app.agent.dynamic_runner import DynamicAgentRunner
from app.agent.grounded_answer import GroundedAnswerService
from app.agent.planner_models import CallToolAction, FinishAction
from app.agent.state import AgentState
from tests.test_planner_executor import FakeRegistry
from app.agent.executor import ToolExecutor


class ScriptedPlanner:
    def __init__(self, decisions):
        self.decisions = list(decisions)

    def decide(self, state, tool_specs):
        return self.decisions.pop(0)


class FakeAnswerService(GroundedAnswerService):
    def __init__(self):
        pass

    def generate(self, *, state, answer_task):
        return {"answer": "Grounded answer [E1].", "answer_task": answer_task}


def test_runner_retrieve_then_finish_generates_answer_without_ingestion():
    registry = FakeRegistry()
    runner = DynamicAgentRunner(
        planner=ScriptedPlanner(
            [
                CallToolAction(
                    tool_name="retrieve_evidence",
                    arguments={"query": "q"},
                    decision_summary="Need evidence.",
                ),
                FinishAction(
                    answer_task="Answer q.",
                    decision_summary="Evidence exists.",
                ),
            ]
        ),
        executor=ToolExecutor(registry=registry),
        answer_service=FakeAnswerService(),
    )

    state = runner.run(user_request="What does p1 say?", runtime_state=AgentState(topic="q"))

    assert state.status == "success"
    assert state.final_answer["answer"] == "Grounded answer [E1]."
    assert [call[0] for call in registry.calls] == ["retrieve_evidence"]


def test_runner_prerequisite_recovery_flow():
    registry = FakeRegistry()
    registry.specs["ensure_papers_retrievable"] = registry.specs["retrieve_evidence"].model_copy(
        update={"name": "ensure_papers_retrievable"}
    )
    responses = [
        {
            "status": "failed",
            "error_type": "paper_not_retrievable",
            "missing_paper_ids": ["p1"],
            "summary": "missing",
        },
        {
            "status": "success",
            "ready_paper_ids": ["p1"],
            "summary": "ready",
        },
        {
            "status": "success",
            "query": "q",
            "retrieved": 1,
            "evidence": [{"chunk_id": "c1", "paper_id": "p1", "text": "Evidence"}],
            "summary": "retrieved",
        },
    ]

    def execute(tool_name, state, **kwargs):
        registry.calls.append((tool_name, kwargs))
        return responses.pop(0)

    registry.execute = execute
    runner = DynamicAgentRunner(
        planner=ScriptedPlanner(
            [
                CallToolAction(
                    tool_name="retrieve_evidence",
                    arguments={"query": "q", "paper_ids": ["p1"]},
                    decision_summary="Try retrieve.",
                ),
                CallToolAction(
                    tool_name="ensure_papers_retrievable",
                    arguments={"query": "ignored"},
                    decision_summary="Prepare paper.",
                ),
                CallToolAction(
                    tool_name="retrieve_evidence",
                    arguments={"query": "q", "paper_ids": ["p1"]},
                    decision_summary="Retry retrieve.",
                ),
                FinishAction(answer_task="Answer q.", decision_summary="done"),
            ]
        ),
        executor=ToolExecutor(registry=registry),
        answer_service=FakeAnswerService(),
    )

    state = runner.run(user_request="What does p1 say?")

    assert state.status == "success"
    assert state.tool_history[0].observation.status == "prerequisite_missing"
    assert state.retrievable_paper_ids == ["p1"]
    assert state.retrieved_evidence_ids == ["c1"]


def test_runner_discovery_only_can_finish_without_retrieval():
    registry = FakeRegistry()
    registry.specs["discover_papers"] = registry.specs["retrieve_evidence"].model_copy(
        update={"name": "discover_papers"}
    )
    registry.responses["discover_papers"] = {
        "status": "success",
        "selected_paper_ids": ["p1"],
        "candidate_paper_ids": ["p1"],
        "summary": "found",
    }
    runner = DynamicAgentRunner(
        planner=ScriptedPlanner(
            [
                CallToolAction(
                    tool_name="discover_papers",
                    arguments={"query": "ignored"},
                    decision_summary="Find papers.",
                ),
                FinishAction(
                    answer_task="List discovered papers.",
                    decision_summary="Papers found.",
                ),
            ]
        ),
        executor=ToolExecutor(registry=registry),
        answer_service=FakeAnswerService(),
    )

    state = runner.run(user_request="Find papers about agent memory")

    assert state.status == "success"
    assert state.known_paper_ids == ["p1"]
    assert [call[0] for call in registry.calls] == ["discover_papers"]


def test_runner_max_steps_fails_gracefully():
    runner = DynamicAgentRunner(
        planner=ScriptedPlanner(
            [
                CallToolAction(
                    tool_name="retrieve_evidence",
                    arguments={"query": "q1"},
                    decision_summary="again",
                ),
                CallToolAction(
                    tool_name="retrieve_evidence",
                    arguments={"query": "q2"},
                    decision_summary="again",
                ),
            ]
        ),
        executor=ToolExecutor(registry=FakeRegistry()),
        answer_service=FakeAnswerService(),
    )

    state = runner.run(user_request="Loop", max_steps=2)

    assert state.status == "failed"
    assert state.last_error == "Maximum planner steps reached."


def test_runner_rejects_finish_too_early_for_factual_task():
    runner = DynamicAgentRunner(
        planner=ScriptedPlanner(
            [
                FinishAction(
                    answer_task="Answer what the paper says.",
                    decision_summary="Too early.",
                )
            ]
        ),
        executor=ToolExecutor(registry=FakeRegistry()),
        answer_service=FakeAnswerService(),
    )

    state = runner.run(user_request="What does the paper say?")

    assert state.status == "failed"
    assert "Finish requires" in state.last_error

