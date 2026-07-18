from fastapi.testclient import TestClient

from app.agent.executor import ToolExecutor
from app.agent.langgraph_runner import LangGraphAgentRunner
from app.agent.planner_eval import EvalAnswerService, EvalRegistry, ScriptedEvalPlanner
from app.agent.planner_models import CallToolAction, FinishAction
from app.api import create_app
from app.conversations.context_builder import ConversationContextBuilder
from app.conversations.service import ConversationAgentService
from app.conversations.sqlite_repository import SQLiteConversationRepository


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
    assert payload["tool_history"][0]["tool_name"] == "retrieve_evidence"

    messages = client.get(f"/threads/{thread_id}/messages").json()["messages"]
    steps = client.get(f"/runs/{run_id}/steps").json()["steps"]

    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert steps[0]["tool_name"] == "retrieve_evidence"
    assert steps[-1]["node_name"] == "finish"


def test_api_returns_404_for_missing_thread(tmp_path):
    client = _client(tmp_path)

    response = client.get("/threads/missing")

    assert response.status_code == 404
