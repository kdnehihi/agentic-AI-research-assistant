from datetime import datetime, timezone
import time
from collections import deque
from pathlib import Path

from fastapi.testclient import TestClient

from app.agent.executor import ToolExecutor
from app.agent.execution_plan import ExecutionPlan, PlanStep
from app.agent.langgraph_runner import LangGraphAgentRunner
from app.agent.planner_eval import EvalAnswerService, EvalRegistry, ScriptedEvalPlanner
from app.agent.planner_models import CallToolAction, FinishAction, ToolObservation
from app.agent.planner_state import PlannerState, ToolExecutionRecord
from app.agent.request_intent import RequestIntent
from app.agent.state import AgentState, Paper
from app.agent.tool_catalog import build_tool_specs
from app.api import _api_final_answer, _chat_response, create_app
from app.config import get_settings
from app.conversations.context_builder import ConversationContextBuilder
from app.conversations.models import ConversationMessage, ConversationThread
from app.conversations.service import ConversationAgentResult, ConversationAgentService
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


class QueueIntentClassifier:
    def __init__(self, intents):
        self.intents = deque(intents)

    def classify(self, user_request):
        del user_request
        return self.intents.popleft()


class ProductFlowRegistry:
    def __init__(self, responses):
        self.specs = build_tool_specs()
        self.responses = {
            tool_name: deque(tool_responses)
            for tool_name, tool_responses in responses.items()
        }
        self.calls = []

    def list_tools(self, category=None):
        return [
            name
            for name, spec in self.specs.items()
            if category is None or spec.category == category
        ]

    def get_tool_spec(self, tool_name):
        return self.specs[tool_name]

    def has_tool(self, tool_name):
        return tool_name in self.specs

    def execute(self, tool_name, state, **kwargs):
        self.calls.append((tool_name, kwargs))
        queue = self.responses.get(tool_name)
        if not queue:
            return {
                "status": "failed",
                "summary": f"No response configured for {tool_name}.",
            }

        response = dict(queue.popleft())
        runtime_papers = response.pop("_runtime_papers", None)
        if runtime_papers is not None:
            state.set_candidate_papers(runtime_papers)
            state.set_selected_papers(runtime_papers)
        return response


def _product_client(tmp_path, *, registry, intents):
    repo = SQLiteConversationRepository(tmp_path / "conversations.sqlite3")
    runner = LangGraphAgentRunner(
        planner=ScriptedEvalPlanner([]),
        executor=ToolExecutor(registry=registry),
        answer_service=EvalAnswerService(),
        intent_classifier=QueueIntentClassifier(intents),
    )
    service = ConversationAgentService(
        conversation_repository=repo,
        run_repository=repo,
        runner=runner,
        context_builder=ConversationContextBuilder(repo),
        summary_trigger_messages=100,
    )
    return TestClient(create_app(conversation_service=service, repository=repo))


def _discovery_intent(topic):
    return RequestIntent(
        task_type="discovery_only",
        topic=topic,
        needs_retrieval=False,
        needs_ingestion=False,
        probe_existing_kb_first=False,
        finish_condition="paper_metadata",
        confidence=0.95,
        rationale="Find papers for the user.",
    )


def _factual_intent(topic):
    return RequestIntent(
        task_type="factual_answer",
        topic=topic,
        needs_retrieval=True,
        needs_ingestion=False,
        probe_existing_kb_first=True,
        finish_condition="retrieved_evidence",
        confidence=0.95,
        rationale="Answer from selected paper evidence.",
    )


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
    assert payload["execution_strategy"] == "knowledge_only"
    assert payload["execution_branch"] == "strategy_knowledge_only"
    assert payload["knowledge_coverage"]["coverage"] == "sufficient"
    assert payload["execution_plan"]["steps"][0]["step_id"] == "retrieve_existing"
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


def test_api_serves_web_ui(tmp_path):
    client = _client(tmp_path)

    response = client.get("/")

    assert response.status_code == 200
    assert "Research Assistant" in response.text
    assert "/static/app.js" in response.text


def test_web_ui_disables_input_and_updates_loading_status():
    script = (
        Path(__file__).resolve().parents[1] / "app" / "static" / "app.js"
    ).read_text(encoding="utf-8")

    assert "inputEl.disabled = isBusy" in script
    assert "updateLoadingBubble(assistantBubble, label)" in script
    assert "Still working... ${data.elapsed_seconds}s" in script
    assert "bubble.innerHTML = \"\"" in script


def test_api_lists_papers_for_sidebar(tmp_path, monkeypatch):
    paper_db = tmp_path / "papers.sqlite3"
    papers_dir = tmp_path / "papers"
    monkeypatch.setenv("PAPER_DB_PATH", str(paper_db))
    monkeypatch.setenv("PAPERS_DIR", str(papers_dir))
    get_settings.cache_clear()
    from app.storage.paper_store import PaperStore

    store = PaperStore()
    store.save_paper(
        Paper(
            paper_id="arxiv:test",
            title="A Test Paper",
            authors=["Ada Lovelace"],
            source="arxiv",
            url="https://example.test/paper",
            abstract="Test abstract.",
            published_date="2026-01-01",
        ),
        topic="test",
    )
    client = _client(tmp_path)

    response = client.get("/papers")

    assert response.status_code == 200
    assert response.json()["papers"][0]["paper_id"] == "arxiv:test"


def test_api_chat_response_includes_discovered_papers():
    now = datetime.now(timezone.utc)
    thread = ConversationThread(
        thread_id="thread-1",
        user_id="local-user",
        title="Research chat",
        created_at=now,
        updated_at=now,
    )
    user_message = ConversationMessage(
        message_id="message-1",
        thread_id="thread-1",
        role="user",
        content="Find papers about agentic RAG.",
        created_at=now,
        sequence_number=1,
        metadata_json={"message_type": "user_request"},
    )
    planner_state = PlannerState(
        user_request="Find papers about agentic RAG.",
        runtime_state=AgentState(
            topic="agentic RAG",
            selected_papers=[
                Paper(
                    paper_id="arxiv:2603.07379",
                    title="SoK: Agentic Retrieval-Augmented Generation",
                    authors=["Saroj Mishra"],
                    source="arxiv",
                    url="https://arxiv.org/abs/2603.07379",
                    abstract="Survey abstract.",
                    semantic_scholar_id="S2ID",
                )
            ],
        ),
        status="success",
        final_answer={"answer": "Found one paper."},
        tool_history=[
            ToolExecutionRecord(
                step=1,
                decision=CallToolAction(
                    tool_name="discover_papers",
                    arguments={"user_query": "agentic RAG"},
                    decision_summary="Find papers.",
                ),
                observation=ToolObservation(
                    tool_name="discover_papers",
                    status="success",
                    summary="Found papers.",
                ),
                call_fingerprint="discover",
            )
        ],
    )
    result = ConversationAgentResult(
        thread=thread,
        user_message=user_message,
        assistant_message=None,
        planner_state=planner_state,
        run_id="run-1",
    )

    response = _chat_response(result)

    assert response.discovered_papers[0]["paper_id"] == "arxiv:2603.07379"
    assert response.discovered_papers[0]["semantic_scholar_id"] == "S2ID"


def test_api_saves_discovered_papers_for_later(tmp_path, monkeypatch):
    paper_db = tmp_path / "papers.sqlite3"
    papers_dir = tmp_path / "papers"
    monkeypatch.setenv("PAPER_DB_PATH", str(paper_db))
    monkeypatch.setenv("PAPERS_DIR", str(papers_dir))
    get_settings.cache_clear()
    client = _client(tmp_path)

    response = client.post(
        "/papers/save-discovered",
        json={
            "papers": [
                {
                    "paper_id": "arxiv:2603.07379",
                    "title": "SoK: Agentic Retrieval-Augmented Generation",
                    "authors": ["Saroj Mishra"],
                    "source": "arxiv",
                    "url": "https://arxiv.org/abs/2603.07379",
                    "abstract": "Survey abstract.",
                    "semantic_scholar_id": "S2ID",
                }
            ],
            "paper_ids": ["arxiv:2603.07379"],
            "knowledge_base_id": "default",
            "prepare_for_rag": False,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["saved"]["inserted_paper_ids"] == ["arxiv:2603.07379"]
    assert payload["papers"][0]["paper_id"] == "arxiv:2603.07379"


def test_api_save_discovered_can_prepare_for_rag(tmp_path, monkeypatch):
    import app.api as api_module

    paper_db = tmp_path / "papers.sqlite3"
    papers_dir = tmp_path / "papers"
    monkeypatch.setenv("PAPER_DB_PATH", str(paper_db))
    monkeypatch.setenv("PAPERS_DIR", str(papers_dir))
    get_settings.cache_clear()

    def fake_prepare(state, *, paper_ids):
        assert state.candidate_papers[0].paper_id == "arxiv:2603.07379"
        assert paper_ids == ["arxiv:2603.07379"]
        return {
            "status": "success",
            "ready_paper_ids": paper_ids,
            "summary": "Prepared 1 paper for semantic retrieval; failed 0.",
        }

    monkeypatch.setattr(api_module, "ensure_papers_retrievable", fake_prepare)
    client = _client(tmp_path)

    response = client.post(
        "/papers/save-discovered",
        json={
            "papers": [
                {
                    "paper_id": "arxiv:2603.07379",
                    "title": "SoK: Agentic Retrieval-Augmented Generation",
                    "source": "arxiv",
                    "url": "https://arxiv.org/abs/2603.07379",
                }
            ],
            "paper_ids": ["arxiv:2603.07379"],
            "prepare_for_rag": True,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["prepared"]["ready_paper_ids"] == ["arxiv:2603.07379"]


def test_api_chat_discovery_respects_requested_paper_count(tmp_path):
    papers = [
        Paper(
            paper_id=f"arxiv:2607.0000{index}",
            title=f"Large Language Model Paper {index}",
            authors=["Test Author"],
            source="arxiv",
            url=f"https://arxiv.org/abs/2607.0000{index}",
            abstract="A recent large language model paper.",
            published_date="2026-07-01",
        )
        for index in range(1, 4)
    ]
    registry = ProductFlowRegistry(
        {
            "discover_papers": [
                {
                    "status": "success",
                    "candidate_paper_ids": [paper.paper_id for paper in papers],
                    "selected_paper_ids": [paper.paper_id for paper in papers],
                    "candidate_count": 3,
                    "selected_count": 3,
                    "summary": "Discovered 3 papers.",
                    "_runtime_papers": papers,
                }
            ],
        }
    )
    client = _product_client(
        tmp_path,
        registry=registry,
        intents=[_discovery_intent("large language model")],
    )

    response = client.post(
        "/chat",
        json={
            "message": "tìm cho tôi 3 papers mới nhất về large language model",
            "title": "LLM papers",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert len(payload["discovered_papers"]) == 3
    assert registry.calls[0][1] == {
        "user_query": "large language model",
        "max_results": 30,
        "max_selected": 3,
    }


def test_api_chat_find_save_prepare_then_ask_saved_paper_same_thread(
    tmp_path,
    monkeypatch,
):
    import app.api as api_module

    paper_db = tmp_path / "papers.sqlite3"
    papers_dir = tmp_path / "papers"
    monkeypatch.setenv("PAPER_DB_PATH", str(paper_db))
    monkeypatch.setenv("PAPERS_DIR", str(papers_dir))
    get_settings.cache_clear()

    def fake_prepare(state, *, paper_ids):
        assert state.candidate_papers[0].paper_id == "arxiv:2601.00001"
        assert paper_ids == ["arxiv:2601.00001"]
        return {
            "status": "success",
            "ready_paper_ids": paper_ids,
            "summary": "Prepared selected papers.",
        }

    monkeypatch.setattr(api_module, "ensure_papers_retrievable", fake_prepare)

    discovered_paper = Paper(
        paper_id="arxiv:2601.00001",
        title="Efficient Transformer Language Models with Linear Attention",
        authors=["Ada Researcher"],
        source="arxiv",
        url="https://arxiv.org/abs/2601.00001",
        abstract="A transformer language model paper with an introduction section.",
        published_date="2026-01-10",
    )
    registry = ProductFlowRegistry(
        {
            "discover_papers": [
                {
                    "status": "success",
                    "candidate_paper_ids": [discovered_paper.paper_id],
                    "selected_paper_ids": [discovered_paper.paper_id],
                    "candidate_count": 1,
                    "selected_count": 1,
                    "summary": "Discovered 1 transformer paper.",
                    "_runtime_papers": [discovered_paper],
                }
            ],
            "retrieve_evidence": [
                {
                    "status": "success",
                    "query": "Give me the introduction of this paper.",
                    "retrieved": 1,
                    "evidence": [
                        {
                            "chunk_id": "intro-1",
                            "paper_id": discovered_paper.paper_id,
                            "section": "Introduction",
                            "text": "The introduction motivates efficient transformer language models.",
                            "final_score": 0.92,
                        }
                    ],
                    "summary": "Retrieved 1 introduction chunk.",
                }
            ],
        }
    )
    client = _product_client(
        tmp_path,
        registry=registry,
        intents=[
            _discovery_intent("latest transformer language model papers"),
            _factual_intent("selected transformer paper introduction"),
        ],
    )

    find_response = client.post(
        "/chat",
        json={
            "message": "Find the latest paper about transformer language models.",
            "title": "Transformer workspace",
        },
    )
    assert find_response.status_code == 200
    find_payload = find_response.json()
    thread_id = find_payload["thread"]["thread_id"]
    assert find_payload["status"] == "success"
    assert find_payload["execution_branch"] == "strategy_discovery_only"
    assert find_payload["discovered_papers"][0]["paper_id"] == discovered_paper.paper_id

    save_response = client.post(
        "/papers/save-discovered",
        json={
            "papers": find_payload["discovered_papers"],
            "paper_ids": [discovered_paper.paper_id],
            "knowledge_base_id": "default",
            "prepare_for_rag": True,
        },
    )
    assert save_response.status_code == 200
    save_payload = save_response.json()
    assert save_payload["saved"]["inserted_paper_ids"] == [discovered_paper.paper_id]
    assert save_payload["prepared"]["ready_paper_ids"] == [discovered_paper.paper_id]

    paper_list = client.get("/papers").json()["papers"]
    assert paper_list[0]["paper_id"] == discovered_paper.paper_id

    ask_response = client.post(
        "/chat",
        json={
            "thread_id": thread_id,
            "message": "Give me the introduction of this paper.",
            "active_paper_ids": [discovered_paper.paper_id],
        },
    )
    assert ask_response.status_code == 200
    ask_payload = ask_response.json()
    assert ask_payload["status"] == "success"
    assert ask_payload["execution_branch"] == "strategy_knowledge_only"
    assert ask_payload["knowledge_coverage"]["coverage"] == "sufficient"
    assert ask_payload["tool_history"][0]["tool_name"] == "retrieve_evidence"
    assert ask_payload["tool_history"][0]["arguments"]["paper_ids"] == [
        discovered_paper.paper_id
    ]
    assert ask_payload["tool_history"][0]["arguments"]["section_groups"] == [
        "introduction"
    ]
    assert ask_payload["final_answer"]["retrieved_evidence_ids"] == ["intro-1"]


def test_api_chat_can_ask_existing_paper_then_find_new_papers_same_thread(tmp_path):
    existing_paper_id = "arxiv:2603.07379"
    new_paper = Paper(
        paper_id="arxiv:2605.00002",
        title="New Agentic RAG Filtering Methods",
        authors=["Grace Hopper"],
        source="arxiv",
        url="https://arxiv.org/abs/2605.00002",
        abstract="A recent paper about agentic RAG filtering.",
        published_date="2026-05-01",
    )
    registry = ProductFlowRegistry(
        {
            "retrieve_evidence": [
                {
                    "status": "success",
                    "query": "What is the introduction of the selected paper?",
                    "retrieved": 1,
                    "evidence": [
                        {
                            "chunk_id": "existing-intro",
                            "paper_id": existing_paper_id,
                            "section": "Introduction",
                            "text": "The existing paper introduces agentic RAG.",
                            "final_score": 0.9,
                        }
                    ],
                    "summary": "Retrieved selected paper introduction.",
                }
            ],
            "discover_papers": [
                {
                    "status": "success",
                    "candidate_paper_ids": [new_paper.paper_id],
                    "selected_paper_ids": [new_paper.paper_id],
                    "candidate_count": 1,
                    "selected_count": 1,
                    "summary": "Discovered a new paper.",
                    "_runtime_papers": [new_paper],
                }
            ],
        }
    )
    client = _product_client(
        tmp_path,
        registry=registry,
        intents=[
            _factual_intent("selected SoK paper introduction"),
            _discovery_intent("latest agentic RAG filtering"),
        ],
    )

    first = client.post(
        "/chat",
        json={
            "message": "What is the introduction of the selected paper?",
            "active_paper_ids": [existing_paper_id],
        },
    )
    assert first.status_code == 200
    first_payload = first.json()
    thread_id = first_payload["thread"]["thread_id"]
    assert first_payload["status"] == "success"
    assert first_payload["tool_history"][0]["arguments"]["paper_ids"] == [
        existing_paper_id
    ]

    second = client.post(
        "/chat",
        json={
            "thread_id": thread_id,
            "message": "Now find the latest paper about agentic RAG filtering.",
        },
    )
    assert second.status_code == 200
    second_payload = second.json()
    assert second_payload["status"] == "success"
    assert second_payload["execution_branch"] == "strategy_discovery_only"
    assert [record[0] for record in registry.calls] == [
        "retrieve_evidence",
        "discover_papers",
    ]
    assert second_payload["discovered_papers"][0]["paper_id"] == new_paper.paper_id

    messages = client.get(f"/threads/{thread_id}/messages").json()["messages"]
    assert [message["role"] for message in messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]


def test_api_stream_chat_returns_sse_events(tmp_path, monkeypatch):
    import app.api as api_module

    def fake_streaming_service(repository, *, on_token):
        del repository

        class FakeStreamingService:
            def run_turn(
                self,
                *,
                user_content,
                thread_id=None,
                title=None,
                user_id=None,
                active_paper_ids=None,
                max_steps=8,
            ):
                del thread_id, title, user_id, active_paper_ids, max_steps
                on_token("hello ")
                on_token("world")
                now = datetime.now(timezone.utc)
                thread = ConversationThread(
                    thread_id="thread-1",
                    user_id="local-user",
                    title="Research chat",
                    created_at=now,
                    updated_at=now,
                )
                user_message = ConversationMessage(
                    message_id="message-1",
                    thread_id="thread-1",
                    role="user",
                    content=user_content,
                    created_at=now,
                    sequence_number=1,
                    metadata_json={"message_type": "user_request"},
                )
                assistant_message = ConversationMessage(
                    message_id="message-2",
                    thread_id="thread-1",
                    role="assistant",
                    content="hello world",
                    created_at=now,
                    sequence_number=2,
                    metadata_json={},
                )
                planner_state = PlannerState(
                    user_request=user_content,
                    runtime_state=AgentState(topic="test"),
                    status="success",
                    final_answer={"answer": "hello world"},
                )
                return ConversationAgentResult(
                    thread=thread,
                    user_message=user_message,
                    assistant_message=assistant_message,
                    planner_state=planner_state,
                    run_id="run-1",
                )

        return FakeStreamingService()

    monkeypatch.setattr(
        api_module,
        "_build_streaming_conversation_service",
        fake_streaming_service,
    )
    client = _client(tmp_path)

    response = client.post("/chat/stream", json={"message": "Say hi"})

    assert response.status_code == 200
    assert "event: status" in response.text
    assert 'data: {"text": "hello "}' in response.text
    assert 'data: {"text": "world"}' in response.text
    assert "event: final" in response.text
    assert "event: done" in response.text


def test_api_stream_chat_times_out_slow_runs(tmp_path, monkeypatch):
    import app.api as api_module

    monkeypatch.setenv("CHAT_STREAM_TIMEOUT_SECONDS", "0.05")

    def fake_streaming_service(repository, *, on_token):
        del repository, on_token

        class SlowStreamingService:
            def run_turn(
                self,
                *,
                user_content,
                thread_id=None,
                title=None,
                user_id=None,
                active_paper_ids=None,
                max_steps=8,
            ):
                del user_content, thread_id, title, user_id, active_paper_ids, max_steps
                time.sleep(0.2)
                raise AssertionError("The stream should time out before finalizing.")

        return SlowStreamingService()

    monkeypatch.setattr(
        api_module,
        "_build_streaming_conversation_service",
        fake_streaming_service,
    )
    client = _client(tmp_path)

    response = client.post("/chat/stream", json={"message": "Slow question"})

    assert response.status_code == 200
    assert "event: error" in response.text
    assert "took too long" in response.text
    assert "event: done" in response.text


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
