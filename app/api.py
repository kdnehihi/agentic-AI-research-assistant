from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
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
from app.conversations.repository import ConversationRunRepository
from app.conversations.service import ConversationAgentResult, ConversationAgentService
from app.config import get_settings
from app.llm.client import create_default_llm_client
from app.observability import configure_logging, request_logging_middleware
from app.storage.factory import (
    create_conversation_repository,
    create_vector_store,
    storage_backend_summary,
)


logger = logging.getLogger(__name__)


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
    request_intent: dict[str, Any] | None = None
    execution_plan: dict[str, Any] | None = None
    tool_history: list[dict[str, Any]]


class ThreadListResponse(BaseModel):
    threads: list[dict[str, Any]]


class MessageListResponse(BaseModel):
    messages: list[dict[str, Any]]


class StepListResponse(BaseModel):
    steps: list[dict[str, Any]]


class ReadinessResponse(BaseModel):
    status: str
    checks: dict[str, Any]


def create_app(
    *,
    conversation_service: ConversationAgentService | None = None,
    repository: ConversationRunRepository | None = None,
) -> FastAPI:
    """Create the FastAPI app for local research-assistant serving."""

    configure_logging()

    @asynccontextmanager
    async def lifespan(app_instance: FastAPI):
        del app_instance
        _configure_runtime_cache_paths()
        _warmup_runtime_models()
        yield

    app = FastAPI(
        title="Agentic AI Research Assistant",
        version="0.1.0",
        description="LangGraph research assistant API with persistent conversations.",
        lifespan=lifespan,
    )
    app.middleware("http")(request_logging_middleware)

    repo = repository
    service = conversation_service

    def get_repository() -> ConversationRunRepository:
        nonlocal repo
        if repo is None:
            repo = create_conversation_repository()
        return repo

    def get_conversation_service() -> ConversationAgentService:
        nonlocal service
        if service is None:
            service = _build_conversation_service(get_repository())
        return service

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready", response_model=ReadinessResponse)
    def ready(
        repo_dep: ConversationRunRepository = Depends(get_repository),
    ) -> ReadinessResponse:
        checks = _readiness_checks(repo_dep)
        status = "ok" if all(
            check.get("status") == "ok" for check in checks.values()
        ) else "degraded"
        return ReadinessResponse(status=status, checks=checks)

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
        repo_dep: ConversationRunRepository = Depends(get_repository),
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
        repo_dep: ConversationRunRepository = Depends(get_repository),
    ) -> dict[str, Any]:
        thread = repo_dep.get_thread(thread_id)
        if thread is None:
            raise HTTPException(status_code=404, detail="Thread not found.")
        return _model_dict(thread)

    @app.get("/threads/{thread_id}/messages", response_model=MessageListResponse)
    def list_messages(
        thread_id: str,
        repo_dep: ConversationRunRepository = Depends(get_repository),
        limit: int = Query(default=50, ge=1, le=100),
    ) -> MessageListResponse:
        if repo_dep.get_thread(thread_id) is None:
            raise HTTPException(status_code=404, detail="Thread not found.")
        messages = repo_dep.list_messages(thread_id, limit=limit)
        return MessageListResponse(messages=[_message_dict(message) for message in messages])

    @app.get("/runs/{run_id}/steps", response_model=StepListResponse)
    def list_run_steps(
        run_id: str,
        repo_dep: ConversationRunRepository = Depends(get_repository),
    ) -> StepListResponse:
        if repo_dep.get_run(run_id) is None:
            raise HTTPException(status_code=404, detail="Run not found.")
        return StepListResponse(
            steps=[_step_dict(step) for step in repo_dep.list_steps(run_id)]
        )

    @app.get("/runs/{run_id}", response_model=dict[str, Any])
    def get_run(
        run_id: str,
        repo_dep: ConversationRunRepository = Depends(get_repository),
    ) -> dict[str, Any]:
        run = repo_dep.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found.")
        return _model_dict(run)

    return app


@lru_cache(maxsize=1)
def _default_app() -> FastAPI:
    return create_app()


app = _default_app()


def _build_conversation_service(
    repository: ConversationRunRepository,
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
        final_answer=_api_final_answer(result.planner_state.final_answer),
        last_error=result.planner_state.last_error,
        request_intent=(
            result.planner_state.request_intent.model_dump(mode="json")
            if result.planner_state.request_intent is not None
            else None
        ),
        execution_plan=(
            result.planner_state.execution_plan.model_dump(mode="json")
            if result.planner_state.execution_plan is not None
            else None
        ),
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


def _readiness_checks(repository: ConversationRunRepository) -> dict[str, Any]:
    settings = get_settings()
    checks: dict[str, Any] = {
        "conversation_db": repository.health_check(),
        "storage_backends": {
            "status": "ok",
            **storage_backend_summary(settings),
        },
        "data_dir": _path_check(settings.data_dir),
        "papers_dir": _path_check(settings.papers_dir),
        "chroma_path": _path_check(settings.chroma_path),
        "llm_provider": {
            "status": "ok",
            "provider": settings.llm_provider,
        },
    }
    if settings.readiness_check_vector_store:
        checks["vector_store"] = _vector_store_check()
    return checks


def _configure_runtime_cache_paths() -> None:
    settings = get_settings()
    os.environ.setdefault("HF_HOME", settings.hf_home)
    os.environ.setdefault(
        "SENTENCE_TRANSFORMERS_HOME",
        settings.sentence_transformers_home,
    )


def _warmup_runtime_models() -> None:
    settings = get_settings()
    if not settings.bge_preload_on_startup:
        return
    try:
        from app.tools.embedding_tools import load_bge_embedder

        load_bge_embedder()
        logger.info(
            "bge_preload_completed",
            extra={
                "event": "bge_preload_completed",
                "bge_model_path": settings.bge_model_path,
                "bge_offline": settings.bge_offline,
            },
        )
    except Exception as exc:
        logger.exception(
            "bge_preload_failed",
            extra={
                "event": "bge_preload_failed",
                "error": str(exc),
            },
        )
        raise


def _api_final_answer(final_answer: Any) -> Any:
    settings = get_settings()
    if settings.api_include_full_evidence_text:
        return final_answer
    if not isinstance(final_answer, dict):
        return final_answer

    compact = dict(final_answer)
    evidence_chunks = compact.get("evidence_chunks")
    if not isinstance(evidence_chunks, list):
        return compact

    compact["evidence_chunks"] = [
        _compact_evidence_chunk(chunk, max_chars=settings.api_evidence_text_max_chars)
        for chunk in evidence_chunks
    ]
    return compact


def _compact_evidence_chunk(chunk: Any, *, max_chars: int) -> Any:
    if not isinstance(chunk, dict):
        return chunk
    compact_chunk = dict(chunk)
    text = compact_chunk.get("text")
    if not isinstance(text, str):
        return compact_chunk
    if max_chars <= 0:
        compact_chunk.pop("text", None)
        compact_chunk["text_truncated"] = True
        return compact_chunk
    if len(text) <= max_chars:
        compact_chunk["text_truncated"] = False
        return compact_chunk
    compact_chunk["text"] = text[:max_chars].rstrip() + "..."
    compact_chunk["text_truncated"] = True
    return compact_chunk


def _path_check(path_value: str) -> dict[str, Any]:
    path = Path(path_value)
    try:
        path.mkdir(parents=True, exist_ok=True)
        return {
            "status": "ok",
            "path": str(path),
            "exists": path.exists(),
            "writable": path.exists() and path.is_dir(),
        }
    except Exception as exc:
        return {
            "status": "error",
            "path": str(path),
            "exists": path.exists(),
            "writable": False,
            "error": str(exc),
        }


def _vector_store_check() -> dict[str, Any]:
    try:
        store = create_vector_store()
        return {
            "status": "ok",
            "backend": get_settings().vector_store_backend,
            "count": store.count(),
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc)}
