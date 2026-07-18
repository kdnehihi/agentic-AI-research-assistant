from __future__ import annotations

from functools import lru_cache
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from app.agent.executor import ToolExecutor
from app.agent.grounded_answer import GroundedAnswerService
from app.agent.langgraph_runner import LangGraphAgentRunner
from app.agent.planner import Planner
from app.conversations.context_builder import ConversationContextBuilder
from app.conversations.models import (
    AgentStep,
    ConversationMessage,
    ConversationThread,
)
from app.conversations.service import ConversationAgentResult, ConversationAgentService
from app.conversations.sqlite_repository import SQLiteConversationRepository
from app.llm.client import create_default_llm_client


class ChatRequest(BaseModel):
    thread_id: str | None = None
    message: str = Field(min_length=1)
    title: str | None = None
    user_id: str | None = None
    max_steps: int = Field(default=8, ge=1, le=20)


class ChatResponse(BaseModel):
    thread: dict[str, Any]
    user_message: dict[str, Any]
    assistant_message: dict[str, Any] | None
    run_id: str
    status: str
    final_answer: Any | None = None
    last_error: str | None = None
    tool_history: list[dict[str, Any]]


class ThreadListResponse(BaseModel):
    threads: list[dict[str, Any]]


class MessageListResponse(BaseModel):
    messages: list[dict[str, Any]]


class StepListResponse(BaseModel):
    steps: list[dict[str, Any]]


def create_app(
    *,
    conversation_service: ConversationAgentService | None = None,
    repository: SQLiteConversationRepository | None = None,
) -> FastAPI:
    """Create the FastAPI app for local research-assistant serving."""

    app = FastAPI(
        title="Agentic AI Research Assistant",
        version="0.1.0",
        description="LangGraph research assistant API with persistent conversations.",
    )

    repo = repository
    service = conversation_service

    def get_repository() -> SQLiteConversationRepository:
        nonlocal repo
        if repo is None:
            repo = SQLiteConversationRepository()
        return repo

    def get_conversation_service() -> ConversationAgentService:
        nonlocal service
        if service is None:
            service = _build_conversation_service(get_repository())
        return service

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/chat", response_model=ChatResponse)
    def chat(
        request: ChatRequest,
        conversation_agent: ConversationAgentService = Depends(get_conversation_service),
    ) -> ChatResponse:
        try:
            result = conversation_agent.run_turn(
                user_content=request.message,
                thread_id=request.thread_id,
                title=request.title,
                user_id=request.user_id,
                max_steps=request.max_steps,
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return _chat_response(result)

    @app.get("/threads", response_model=ThreadListResponse)
    def list_threads(
        repo_dep: SQLiteConversationRepository = Depends(get_repository),
        user_id: str | None = None,
        limit: int = Query(default=50, ge=1, le=100),
    ) -> ThreadListResponse:
        return ThreadListResponse(
            threads=[
                _model_dict(thread)
                for thread in repo_dep.list_threads(user_id=user_id, limit=limit)
            ]
        )

    @app.get("/threads/{thread_id}", response_model=dict[str, Any])
    def get_thread(
        thread_id: str,
        repo_dep: SQLiteConversationRepository = Depends(get_repository),
    ) -> dict[str, Any]:
        thread = repo_dep.get_thread(thread_id)
        if thread is None:
            raise HTTPException(status_code=404, detail="Thread not found.")
        return _model_dict(thread)

    @app.get("/threads/{thread_id}/messages", response_model=MessageListResponse)
    def list_messages(
        thread_id: str,
        repo_dep: SQLiteConversationRepository = Depends(get_repository),
        limit: int = Query(default=50, ge=1, le=100),
    ) -> MessageListResponse:
        if repo_dep.get_thread(thread_id) is None:
            raise HTTPException(status_code=404, detail="Thread not found.")
        messages = repo_dep.list_messages(thread_id, limit=limit)
        return MessageListResponse(messages=[_message_dict(message) for message in messages])

    @app.get("/runs/{run_id}/steps", response_model=StepListResponse)
    def list_run_steps(
        run_id: str,
        repo_dep: SQLiteConversationRepository = Depends(get_repository),
    ) -> StepListResponse:
        if repo_dep.get_run(run_id) is None:
            raise HTTPException(status_code=404, detail="Run not found.")
        return StepListResponse(
            steps=[_step_dict(step) for step in repo_dep.list_steps(run_id)]
        )

    return app


@lru_cache(maxsize=1)
def _default_app() -> FastAPI:
    return create_app()


app = _default_app()


def _build_conversation_service(
    repository: SQLiteConversationRepository,
) -> ConversationAgentService:
    llm_client = create_default_llm_client()
    runner = LangGraphAgentRunner(
        planner=Planner(llm_client),
        executor=ToolExecutor(),
        answer_service=GroundedAnswerService(llm_client=llm_client),
    )
    return ConversationAgentService(
        conversation_repository=repository,
        run_repository=repository,
        runner=runner,
        context_builder=ConversationContextBuilder(repository),
    )


def _chat_response(result: ConversationAgentResult) -> ChatResponse:
    return ChatResponse(
        thread=_model_dict(result.thread),
        user_message=_message_dict(result.user_message),
        assistant_message=(
            _message_dict(result.assistant_message)
            if result.assistant_message is not None
            else None
        ),
        run_id=result.run_id,
        status=result.planner_state.status,
        final_answer=result.planner_state.final_answer,
        last_error=result.planner_state.last_error,
        tool_history=[
            {
                "step": record.step,
                "tool_name": record.decision.tool_name,
                "arguments": record.decision.arguments,
                "decision_summary": record.decision.decision_summary,
                "status": record.observation.status,
                "summary": record.observation.summary,
                "latency_ms": record.latency_ms,
            }
            for record in result.planner_state.tool_history
        ],
    )


def _model_dict(model: ConversationThread) -> dict[str, Any]:
    return model.model_dump(mode="json")


def _message_dict(message: ConversationMessage) -> dict[str, Any]:
    return message.model_dump(mode="json")


def _step_dict(step: AgentStep) -> dict[str, Any]:
    return step.model_dump(mode="json")
