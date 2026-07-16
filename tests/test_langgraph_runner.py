import pytest

from app.agent.grounded_answer import GroundedAnswerService
from app.agent.langgraph_runner import LangGraphAgentRunner
from app.agent.planner_models import CallToolAction, FinishAction
from app.agent.state import AgentState
from app.agent.tool_spec import EnsurePapersRetrievableArgs
from app.agent.executor import ToolExecutor
from tests.test_dynamic_runner import ScriptedPlanner
from tests.test_planner_executor import FakeRegistry


pytest.importorskip("langgraph")


class FakeAnswerService(GroundedAnswerService):
    def __init__(self):
        pass

    def generate(self, *, state, answer_task):
        return {"answer": "Graph answer [E1].", "answer_task": answer_task}


def test_langgraph_runner_retrieve_then_finish():
    registry = FakeRegistry()
    runner = LangGraphAgentRunner(
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
        policy_enabled=False,
    )

    state = runner.run(user_request="What does p1 say?", runtime_state=AgentState(topic="q"))

    assert state.status == "success"
    assert state.final_answer["answer"] == "Graph answer [E1]."
    assert [call[0] for call in registry.calls] == ["retrieve_evidence"]


def test_langgraph_runner_auto_recovers_unindexed_retrieval():
    registry = FakeRegistry()
    registry.specs["ensure_papers_retrievable"] = registry.specs[
        "retrieve_evidence"
    ].model_copy(
        update={
            "name": "ensure_papers_retrievable",
            "args_schema": EnsurePapersRetrievableArgs,
        }
    )
    responses = [
        {
            "status": "failed",
            "error_type": "paper_not_retrievable",
            "missing_paper_ids": ["p1"],
            "evidence": [],
            "summary": "Retrieval prerequisite failed because papers are not indexed.",
        },
        {
            "status": "success",
            "ready_paper_ids": ["p1"],
            "summary": "Prepared 1 papers for semantic retrieval; failed 0.",
        },
        {
            "status": "success",
            "query": "q",
            "retrieved": 1,
            "evidence": [{"chunk_id": "c1", "paper_id": "p1", "text": "Evidence"}],
            "summary": "Retrieved 1 evidence chunks.",
        },
    ]

    def execute(tool_name, state, **kwargs):
        registry.calls.append((tool_name, kwargs))
        return responses.pop(0)

    registry.execute = execute
    runner = LangGraphAgentRunner(
        planner=ScriptedPlanner(
            [
                CallToolAction(
                    tool_name="retrieve_evidence",
                    arguments={"query": "q", "paper_ids": ["p1"]},
                    decision_summary="Try retrieval.",
                ),
                FinishAction(
                    answer_task="Answer q.",
                    decision_summary="Recovered evidence exists.",
                ),
            ]
        ),
        executor=ToolExecutor(registry=registry),
        answer_service=FakeAnswerService(),
        policy_enabled=False,
    )

    state = runner.run(user_request="What does p1 say?", runtime_state=AgentState(topic="q"))

    assert state.status == "success"
    assert [call[0] for call in registry.calls] == [
        "retrieve_evidence",
        "ensure_papers_retrievable",
        "retrieve_evidence",
    ]
    assert registry.calls[1][1] == {"paper_ids": ["p1"]}
    assert state.retrieved_evidence_ids == ["c1"]
