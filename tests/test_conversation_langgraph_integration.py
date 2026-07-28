import pytest

from app.agent.executor import ToolExecutor
from app.agent.langgraph_runner import LangGraphAgentRunner
from app.agent.planner_eval import EvalAnswerService, EvalRegistry, ScriptedEvalPlanner
from app.agent.planner_models import CallToolAction, FinishAction
from app.agent.request_intent import RequestIntent
from app.conversations.context_builder import ConversationContextBuilder
from app.conversations.service import ConversationAgentService
from app.conversations.sqlite_repository import SQLiteConversationRepository


pytest.importorskip("langgraph")


class QueueIntentClassifier:
    def __init__(self, intents):
        self.intents = list(intents)

    def classify(self, user_request):
        del user_request
        return self.intents.pop(0)


def _discovery_only_intent(topic):
    return RequestIntent(
        task_type="discovery_only",
        topic=topic,
        needs_retrieval=False,
        needs_ingestion=False,
        probe_existing_kb_first=False,
        finish_condition="paper_metadata",
        confidence=0.95,
        rationale="Find papers only.",
    )


def _factual_intent(topic):
    return RequestIntent(
        task_type="comparison",
        topic=topic,
        needs_retrieval=True,
        needs_ingestion=False,
        probe_existing_kb_first=True,
        finish_condition="retrieved_evidence",
        confidence=0.95,
        rationale="Answer from paper evidence.",
    )


def _service(
    tmp_path,
    planner,
    registry,
    *,
    intent_classifier=None,
    summary_trigger_messages=100,
    policy_enabled=True,
):
    repo = SQLiteConversationRepository(tmp_path / "conversations.sqlite3")
    runner = LangGraphAgentRunner(
        planner=planner,
        executor=ToolExecutor(registry=registry),
        answer_service=EvalAnswerService(),
        intent_classifier=intent_classifier,
        policy_enabled=policy_enabled,
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
    service, repo = _service(
        tmp_path,
        planner,
        registry,
        intent_classifier=QueueIntentClassifier(
            [
                _discovery_only_intent("Agentic RAG"),
                _factual_intent("Agentic RAG comparison"),
            ]
        ),
    )

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
    assert second.planner_state.active_paper_ids == []
    assert second.assistant_message.metadata_json["agent_run_id"] == second.run_id
    assert second.assistant_message.metadata_json["active_paper_ids"] == []
    steps = repo.list_steps(second.run_id)
    assert steps[0].node_name == "planner_setup"
    assert steps[1].tool_name == "retrieve_evidence"


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


def test_same_thread_turns_reset_planner_step_budget(tmp_path):
    planner = ScriptedEvalPlanner(
        [
            CallToolAction(
                tool_name="retrieve_evidence",
                arguments={"query": "first"},
                decision_summary="Retrieve first turn evidence.",
            ),
            FinishAction(answer_task="First answer.", decision_summary="done"),
            CallToolAction(
                tool_name="retrieve_evidence",
                arguments={"query": "second"},
                decision_summary="Retrieve second turn evidence.",
            ),
            FinishAction(answer_task="Second answer.", decision_summary="done"),
        ]
    )
    registry = EvalRegistry(
        {
            "retrieve_evidence": [
                {
                    "status": "success",
                    "query": "first",
                    "retrieved": 1,
                    "evidence": [{"chunk_id": "c1", "paper_id": "p1", "text": "E1"}],
                    "summary": "first retrieved",
                },
                {
                    "status": "success",
                    "query": "second",
                    "retrieved": 1,
                    "evidence": [{"chunk_id": "c2", "paper_id": "p1", "text": "E2"}],
                    "summary": "second retrieved",
                },
            ]
        }
    )
    service, repo = _service(
        tmp_path,
        planner,
        registry,
        intent_classifier=QueueIntentClassifier(
            [_factual_intent("first"), _factual_intent("second")]
        ),
        policy_enabled=False,
    )

    first = service.run_turn(user_content="What is the abstract?", max_steps=2)
    second = service.run_turn(
        thread_id=first.thread.thread_id,
        user_content="What is the introduction?",
        max_steps=2,
    )

    first_run = repo.get_run(first.run_id)
    second_run = repo.get_run(second.run_id)
    assert first.thread.thread_id == second.thread.thread_id
    assert first.run_id != second.run_id
    assert first.planner_state.step_count == 1
    assert second.planner_state.step_count == 1
    assert first_run.graph_thread_id != second_run.graph_thread_id
    assert first_run.graph_thread_id.endswith(first.user_message.message_id)
    assert second_run.graph_thread_id.endswith(second.user_message.message_id)


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
