import pytest

from app.agent.executor import ToolExecutor
from app.agent.langgraph_runner import LangGraphAgentRunner
from app.agent.planner_eval import EvalAnswerService, EvalRegistry, ScriptedEvalPlanner
from app.agent.planner_models import CallToolAction, FinishAction
from app.conversations.context_builder import ConversationContextBuilder
from app.conversations.service import ConversationAgentService
from app.conversations.sqlite_repository import SQLiteConversationRepository


pytest.importorskip("langgraph")


def _service(tmp_path, planner, registry, *, summary_trigger_messages=100):
    repo = SQLiteConversationRepository(tmp_path / "conversations.sqlite3")
    runner = LangGraphAgentRunner(
        planner=planner,
        executor=ToolExecutor(registry=registry),
        answer_service=EvalAnswerService(),
    )
    return ConversationAgentService(
        conversation_repository=repo,
        run_repository=repo,
        runner=runner,
        context_builder=ConversationContextBuilder(repo, recent_message_limit=4),
        summary_trigger_messages=summary_trigger_messages,
    ), repo


def test_multi_turn_conversation_persists_context_and_traces(tmp_path):
    planner = ScriptedEvalPlanner(
        [
            CallToolAction(
                tool_name="discover_papers",
                arguments={"user_query": "Agentic RAG", "max_results": 2, "max_selected": 2},
                decision_summary="Find papers.",
            ),
            FinishAction(answer_task="Return papers.", decision_summary="found"),
            FinishAction(answer_task="Compare previous papers.", decision_summary="retrieved"),
        ]
    )
    registry = EvalRegistry(
        {
            "discover_papers": [
                {
                    "status": "success",
                    "candidate_paper_ids": ["p1", "p2"],
                    "selected_paper_ids": ["p1", "p2"],
                    "summary": "found",
                }
            ],
            "retrieve_evidence": [
                {
                    "status": "success",
                    "query": "Compare the first one with the second one.",
                    "retrieved": 2,
                    "evidence": [
                        {"chunk_id": "c1", "paper_id": "p1", "text": "E1"},
                        {"chunk_id": "c2", "paper_id": "p2", "text": "E2"},
                    ],
                    "summary": "retrieved",
                }
            ],
        }
    )
    service, repo = _service(tmp_path, planner, registry)

    first = service.run_turn(user_content="Find papers about Agentic RAG.")
    second = service.run_turn(
        thread_id=first.thread.thread_id,
        user_content="Compare the first one with the second one.",
    )

    messages = repo.list_messages(first.thread.thread_id)
    assert [message.role for message in messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert second.planner_state.active_paper_ids == ["p1", "p2"]
    assert second.assistant_message.metadata_json["agent_run_id"] == second.run_id
    assert second.assistant_message.metadata_json["active_paper_ids"] == ["p1", "p2"]
    assert repo.list_steps(second.run_id)[0].tool_name == "retrieve_evidence"


def test_different_threads_do_not_leak_context(tmp_path):
    planner = ScriptedEvalPlanner(
        [
            CallToolAction(
                tool_name="discover_papers",
                arguments={"user_query": "A", "max_results": 1},
                decision_summary="A",
            ),
            FinishAction(answer_task="A", decision_summary="done"),
            FinishAction(answer_task="B", decision_summary="done"),
        ]
    )
    registry = EvalRegistry(
        {
            "discover_papers": [
                {
                    "status": "success",
                    "candidate_paper_ids": ["pA"],
                    "selected_paper_ids": ["pA"],
                    "summary": "found",
                }
            ],
            "retrieve_evidence": [
                {"status": "success", "retrieved": 0, "evidence": [], "summary": "none"}
            ],
        }
    )
    service, _ = _service(tmp_path, planner, registry)

    first = service.run_turn(user_content="Find papers about A.")
    second = service.run_turn(user_content="What did we discuss?")

    assert first.thread.thread_id != second.thread.thread_id
    assert second.planner_state.active_paper_ids == []


def test_failed_run_does_not_create_fake_assistant_message(tmp_path):
    planner = ScriptedEvalPlanner(
        [
            FinishAction(
                answer_task="Answer too early.",
                decision_summary="No evidence.",
            )
        ]
    )
    registry = EvalRegistry({})
    service, repo = _service(tmp_path, planner, registry)

    result = service.run_turn(user_content="What does the paper say?")

    messages = repo.list_messages(result.thread.thread_id)
    stored_run = repo.get_run(result.run_id)
    steps = repo.list_steps(result.run_id)
    assert result.assistant_message is None
    assert [message.role for message in messages] == ["user"]
    assert stored_run.status == "failed"
    assert steps[-1].node_name == "finish"
    assert steps[-1].observation_status == "failed"
