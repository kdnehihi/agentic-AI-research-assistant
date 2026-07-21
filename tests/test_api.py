from fastapi.testclient import TestClient

from app.agent.executor import ToolExecutor
from app.agent.execution_plan import ExecutionPlan, PlanStep
from app.agent.langgraph_runner import LangGraphAgentRunner
from app.agent.planner_eval import EvalAnswerService, EvalRegistry, ScriptedEvalPlanner
from app.agent.planner_models import CallToolAction, FinishAction
from app.agent.request_intent import RequestIntent
from app.api import _api_final_answer, create_app
from app.conversations.context_builder import ConversationContextBuilder
from app.conversations.service import ConversationAgentService
from app.conversations.sqlite_repository import SQLiteConversationRepository


class StaticIntentClassifier:
    def classify(self, user_request):
        del user_request
        return RequestIntent(
            task_type="factual_answer",
            topic="agentic RAG research directions",
            needs_retrieval=True,
            needs_ingestion=False,
            probe_existing_kb_first=False,
            finish_condition="retrieved_evidence",
            confidence=1.0,
            rationale="Test intent.",
        )


class StaticPlanGenerator:
    def generate_plan(self, *, user_request, request_intent, tool_specs):
        del request_intent, tool_specs
        return ExecutionPlan(
            goal=user_request,
            strategy="Retrieve evidence then finish.",
            steps=[
                PlanStep(
                    step_id="retrieve",
                    kind="tool",
                    tool_name="retrieve_evidence",
                    arguments={"query": "agentic RAG research directions"},
                ),
                PlanStep(
                    step_id="finish",
                    kind="finish",
                    answer_task="Answer from retrieved evidence.",
                ),
            ],
        )


def _client(tmp_path):
    repo = SQLiteConversationRepository(tmp_path / "conversations.sqlite3")
    planner = ScriptedEvalPlanner(
        [
            CallToolAction(
                tool_name="retrieve_evidence",
                arguments={"query": "agentic RAG research directions"},
                decision_summary="Need evidence.",
            ),
            FinishAction(
                answer_task="Answer from retrieved evidence.",
                decision_summary="Evidence exists.",
            ),
        ]
    )
    registry = EvalRegistry(
        {
            "retrieve_evidence": [
                {
                    "status": "success",
                    "query": "agentic RAG research directions",
                    "retrieved": 1,
                    "evidence": [
                        {
                            "chunk_id": "c1",
                            "paper_id": "p1",
                            "text": "Agentic RAG research direction evidence.",
                        }
                    ],
                    "summary": "Retrieved 1 evidence chunk.",
                }
            ]
        }
    )
    runner = LangGraphAgentRunner(
        planner=planner,
        executor=ToolExecutor(registry=registry),
        answer_service=EvalAnswerService(),
        intent_classifier=StaticIntentClassifier(),
        plan_generator=StaticPlanGenerator(),
        policy_enabled=False,
    )
    service = ConversationAgentService(
        conversation_repository=repo,
        run_repository=repo,
        runner=runner,
        context_builder=ConversationContextBuilder(repo),
        summary_trigger_messages=100,
    )
    return TestClient(create_app(conversation_service=service, repository=repo))


def test_api_chat_persists_messages_and_steps(tmp_path):
    client = _client(tmp_path)

    response = client.post(
        "/chat",
        json={"message": "What are agentic RAG research directions?"},
    )

    assert response.status_code == 200
    payload = response.json()
    thread_id = payload["thread"]["thread_id"]
    run_id = payload["run_id"]
    assert payload["status"] == "success"
    assert payload["assistant_message"]["role"] == "assistant"
    assert payload["request_intent"]["task_type"] == "factual_answer"
    assert payload["execution_plan"]["steps"][0]["step_id"] == "retrieve"
    assert payload["tool_history"][0]["tool_name"] == "retrieve_evidence"

    messages = client.get(f"/threads/{thread_id}/messages").json()["messages"]
    run_payload = client.get(f"/runs/{run_id}").json()
    steps = client.get(f"/runs/{run_id}/steps").json()["steps"]

    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert run_payload["run_id"] == run_id
    assert steps[0]["node_name"] == "planner_setup"
    assert steps[1]["tool_name"] == "retrieve_evidence"
    assert steps[-1]["node_name"] == "finish"


def test_api_ready_reports_storage_checks(tmp_path):
    client = _client(tmp_path)

    response = client.get("/ready")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["checks"]["conversation_db"]["status"] == "ok"
    assert payload["checks"]["llm_provider"]["status"] == "ok"


def test_api_final_answer_compacts_evidence_text(monkeypatch):
    monkeypatch.setenv("API_EVIDENCE_TEXT_MAX_CHARS", "12")
    monkeypatch.setenv("API_INCLUDE_FULL_EVIDENCE_TEXT", "false")
    final_answer = {
        "answer": "Answer [E1].",
        "evidence_chunks": [
            {
                "evidence_id": "E1",
                "chunk_id": "c1",
                "text": "This evidence chunk is intentionally long.",
            }
        ],
    }

    compact = _api_final_answer(final_answer)

    assert compact["evidence_chunks"][0]["text"] == "This evidenc..."
    assert compact["evidence_chunks"][0]["text_truncated"] is True
    assert final_answer["evidence_chunks"][0]["text"].endswith("long.")


def test_api_final_answer_can_include_full_evidence_text(monkeypatch):
    monkeypatch.setenv("API_INCLUDE_FULL_EVIDENCE_TEXT", "true")
    final_answer = {
        "answer": "Answer [E1].",
        "evidence_chunks": [{"evidence_id": "E1", "text": "full evidence text"}],
    }

    assert _api_final_answer(final_answer) is final_answer


def test_api_returns_404_for_missing_thread(tmp_path):
    client = _client(tmp_path)

    response = client.get("/threads/missing")

    assert response.status_code == 404
