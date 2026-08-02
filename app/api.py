from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.agent.executor import ToolExecutor
from app.agent.grounded_answer import GroundedAnswerService, StreamingGroundedAnswerService
from app.agent.langgraph_runner import LangGraphAgentRunner
from app.agent.planner import Planner
from app.agent.planner_state import PlannerState
from app.agent.state import AgentState, Paper
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
from app.services.ingestion_jobs import IngestionJob, IngestionJobQueue
from app.storage.factory import (
    create_conversation_repository,
    create_ingestion_job_store,
    create_paper_store,
    create_vector_store,
    storage_backend_summary,
)
from app.tools.production.ingestion_tools import ensure_papers_retrievable
from app.tools.production.knowledge_base_tools import save_papers_to_kb
from app.tools.fetch_selected_papers import remove_fetched_papers


logger = logging.getLogger(__name__)
DEFAULT_CHAT_STREAM_TIMEOUT_SECONDS = 120.0
STREAM_STATUS_INTERVAL_SECONDS = 5.0


class ChatRequest(BaseModel):
    thread_id: str | None = None
    message: str = Field(min_length=1)
    title: str | None = None
    user_id: str | None = None
    active_paper_ids: list[str] = Field(default_factory=list, max_length=20)
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
    execution_strategy: str | None = None
    knowledge_coverage: dict[str, Any] | None = None
    execution_branch: str | None = None
    discovered_papers: list[dict[str, Any]] = Field(default_factory=list)
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


class PaperListResponse(BaseModel):
    papers: list[dict[str, Any]]


class DeleteResponse(BaseModel):
    status: str
    summary: str
    detail: dict[str, Any] = Field(default_factory=dict)


class SaveDiscoveredPapersRequest(BaseModel):
    thread_id: str | None = None
    papers: list[dict[str, Any]] = Field(min_length=1, max_length=20)
    paper_ids: list[str] | None = Field(default=None, max_length=20)
    knowledge_base_id: str = Field(default="default", min_length=1)
    prepare_for_rag: bool = False


class SaveDiscoveredPapersResponse(BaseModel):
    status: str
    saved: dict[str, Any]
    prepared: dict[str, Any] | None = None
    prepare_job: dict[str, Any] | None = None
    papers: list[dict[str, Any]]
    summary: str


class IngestionJobResponse(BaseModel):
    job: dict[str, Any]


def create_app(
    *,
    conversation_service: ConversationAgentService | None = None,
    repository: ConversationRunRepository | None = None,
    ingestion_job_queue: IngestionJobQueue | None = None,
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
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    repo = repository
    service = conversation_service
    jobs = ingestion_job_queue

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

    def get_ingestion_job_queue() -> IngestionJobQueue:
        nonlocal jobs
        if jobs is None:
            jobs = _build_ingestion_job_queue(get_repository())
        return jobs

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        index_path = static_dir / "index.html"
        if not index_path.exists():
            raise HTTPException(status_code=404, detail="Web UI is not installed.")
        return HTMLResponse(index_path.read_text(encoding="utf-8"))

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
                active_paper_ids=request.active_paper_ids,
                max_steps=request.max_steps,
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return _chat_response(result)

    @app.post("/chat/stream")
    def chat_stream(
        request: ChatRequest,
        repo_dep: ConversationRunRepository = Depends(get_repository),
    ) -> StreamingResponse:
        return StreamingResponse(
            _chat_event_stream(request=request, repository=repo_dep),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/papers", response_model=PaperListResponse)
    def list_papers(
        limit: int = Query(default=50, ge=1, le=200),
    ) -> PaperListResponse:
        try:
            store = create_paper_store()
            records = store.list_paper_records(limit=limit)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return PaperListResponse(papers=records)

    @app.delete("/papers/unsaved", response_model=DeleteResponse)
    def delete_unsaved_papers() -> DeleteResponse:
        try:
            store = create_paper_store()
            saved_ids = store.get_saved_paper_ids()
            paper_ids = [
                paper_id
                for paper_id in store.get_all_paper_ids()
                if paper_id not in saved_ids
            ]
            detail = _delete_paper_workspace(
                paper_ids=paper_ids,
                store=store,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return DeleteResponse(
            status="success",
            summary=(
                f"Removed {detail['metadata_removed']} unsaved paper metadata "
                f"records; kept {len(saved_ids)} saved papers."
            ),
            detail={
                **detail,
                "kept_saved_paper_count": len(saved_ids),
            },
        )

    @app.delete("/papers/{paper_id}", response_model=DeleteResponse)
    def delete_paper(paper_id: str) -> DeleteResponse:
        try:
            store = create_paper_store()
            detail = _delete_paper_workspace(
                paper_ids=[paper_id],
                store=store,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        if detail["metadata_removed"] == 0 and detail["files_removed"] == 0:
            raise HTTPException(status_code=404, detail="Paper not found.")
        return DeleteResponse(
            status="success",
            summary=(
                f"Removed paper {paper_id} from metadata, fetched files, "
                "and vector index."
            ),
            detail=detail,
        )

    @app.post("/papers/save-discovered", response_model=SaveDiscoveredPapersResponse)
    def save_discovered_papers(
        request: SaveDiscoveredPapersRequest,
        repo_dep: ConversationRunRepository = Depends(get_repository),
        job_queue: IngestionJobQueue = Depends(get_ingestion_job_queue),
    ) -> SaveDiscoveredPapersResponse:
        try:
            if request.thread_id and repo_dep.get_thread(request.thread_id) is None:
                raise ValueError(
                    f"Conversation thread '{request.thread_id}' does not exist."
                )
            papers = [_paper_from_client_payload(paper) for paper in request.papers]
            paper_ids = _requested_paper_ids(papers, request.paper_ids)
            state = AgentState(topic=request.knowledge_base_id)
            state.set_candidate_papers(papers)
            saved = save_papers_to_kb(
                state,
                paper_ids=paper_ids,
                knowledge_base_id=request.knowledge_base_id,
            )
            prepared = None
            prepare_job = None
            if request.prepare_for_rag:
                prepare_job = job_queue.submit(
                    papers=papers,
                    paper_ids=paper_ids,
                    knowledge_base_id=request.knowledge_base_id,
                    thread_id=request.thread_id,
                )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        status = _merge_save_prepare_status(saved, prepared)
        stored_records = _records_for_paper_ids(paper_ids)
        return SaveDiscoveredPapersResponse(
            status=status,
            saved=saved,
            prepared=prepared,
            prepare_job=(
                prepare_job.model_dump(mode="json")
                if prepare_job is not None
                else None
            ),
            papers=stored_records,
            summary=_save_discovered_summary(
                saved=saved,
                prepared=prepared,
                prepare_job=prepare_job.model_dump(mode="json")
                if prepare_job is not None
                else None,
            ),
        )

    @app.get("/ingestion-jobs/{job_id}", response_model=IngestionJobResponse)
    def get_ingestion_job(
        job_id: str,
        job_queue: IngestionJobQueue = Depends(get_ingestion_job_queue),
    ) -> IngestionJobResponse:
        job = job_queue.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Ingestion job not found.")
        return IngestionJobResponse(job=job.model_dump(mode="json"))


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

    @app.delete("/threads/{thread_id}", response_model=DeleteResponse)
    def delete_thread(
        thread_id: str,
        repo_dep: ConversationRunRepository = Depends(get_repository),
    ) -> DeleteResponse:
        if not repo_dep.delete_thread(thread_id):
            raise HTTPException(status_code=404, detail="Thread not found.")
        return DeleteResponse(
            status="success",
            summary=f"Deleted chat thread {thread_id}.",
            detail={"thread_id": thread_id},
        )

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


def _build_streaming_conversation_service(
    repository: ConversationRunRepository,
    *,
    on_token,
) -> ConversationAgentService:
    llm_client = create_default_llm_client()
    runner = LangGraphAgentRunner(
        planner=Planner(llm_client),
        executor=ToolExecutor(),
        answer_service=StreamingGroundedAnswerService(
            llm_client=llm_client,
            on_token=on_token,
        ),
    )
    return ConversationAgentService(
        conversation_repository=repository,
        run_repository=repository,
        runner=runner,
        context_builder=ConversationContextBuilder(repository),
    )


def _build_ingestion_job_queue(
    repository: ConversationRunRepository | None = None,
) -> IngestionJobQueue:
    settings = get_settings()
    return IngestionJobQueue(
        prepare_func=ensure_papers_retrievable,
        worker_count=settings.ingestion_job_worker_count,
        store=create_ingestion_job_store(settings),
        on_complete=(
            (lambda job: _record_ingestion_context_update(repository, job))
            if repository is not None
            else None
        ),
    )


def _record_ingestion_context_update(
    repository: ConversationRunRepository,
    job: IngestionJob,
) -> None:
    if job.thread_id is None or job.status not in {"success", "partial_success"}:
        return
    ready_ids = _ready_ingestion_paper_ids(job.result or {})
    if not ready_ids:
        return
    if repository.get_thread(job.thread_id) is None:
        return
    repository.append_message(
        thread_id=job.thread_id,
        role="system",
        content="Prepared papers for RAG: " + ", ".join(ready_ids),
        metadata_json={
            "message_type": "paper_context_update",
            "hidden_from_ui": True,
            "paper_context_source": "ingestion_job",
            "context_priority": 100,
            "ingestion_job_id": job.job_id,
            "knowledge_base_id": job.knowledge_base_id,
            "paper_ids": ready_ids,
            "active_paper_ids": ready_ids,
            "saved_paper_ids": ready_ids,
            "retrievable_paper_ids": ready_ids,
        },
    )
    logger.info(
        "paper_context_updated",
        extra={
            "structured": {
                "thread_id": job.thread_id,
                "job_id": job.job_id,
                "paper_count": len(ready_ids),
            }
        },
    )


def _ready_ingestion_paper_ids(result: dict[str, Any]) -> list[str]:
    ready_ids: list[str] = []
    for key in ("ready_paper_ids", "already_ready_paper_ids"):
        values = result.get(key) or []
        if not isinstance(values, list):
            continue
        for value in values:
            if isinstance(value, str) and value and value not in ready_ids:
                ready_ids.append(value)
    return ready_ids


def _chat_event_stream(
    *,
    request: ChatRequest,
    repository: ConversationRunRepository,
):
    events: queue.Queue[tuple[str, dict[str, Any]]] = queue.Queue()
    timeout_seconds = _chat_stream_timeout_seconds()
    started_at = time.monotonic()

    def on_token(token: str) -> None:
        events.put(("token", {"text": token}))

    def worker() -> None:
        try:
            service = _build_streaming_conversation_service(
                repository,
                on_token=on_token,
            )
            result = service.run_turn(
                user_content=request.message,
                thread_id=request.thread_id,
                title=request.title,
                user_id=request.user_id,
                active_paper_ids=request.active_paper_ids,
                max_steps=request.max_steps,
            )
            events.put(("final", _chat_response(result).model_dump(mode="json")))
        except Exception as exc:
            events.put(("error", {"message": str(exc)}))
        finally:
            events.put(("done", {}))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    yield _sse_event(
        "status",
        {
            "message": "started",
            "timeout_seconds": timeout_seconds,
        },
    )

    while True:
        elapsed = time.monotonic() - started_at
        remaining = timeout_seconds - elapsed
        if remaining <= 0:
            yield _sse_event(
                "error",
                {
                    "message": (
                        "The request took too long to answer. Please narrow the "
                        "question, select fewer papers, or try again."
                    ),
                    "timeout_seconds": timeout_seconds,
                },
            )
            yield _sse_event("done", {})
            break

        try:
            event_name, payload = events.get(
                timeout=min(STREAM_STATUS_INTERVAL_SECONDS, remaining)
            )
        except queue.Empty:
            yield _sse_event(
                "status",
                {
                    "message": "Still working",
                    "elapsed_seconds": round(time.monotonic() - started_at, 1),
                    "timeout_seconds": timeout_seconds,
                },
            )
            continue
        yield _sse_event(event_name, payload)
        if event_name == "done":
            break


def _chat_stream_timeout_seconds() -> float:
    raw_timeout = os.getenv("CHAT_STREAM_TIMEOUT_SECONDS")
    if raw_timeout is None:
        return DEFAULT_CHAT_STREAM_TIMEOUT_SECONDS
    try:
        return max(float(raw_timeout), 0.05)
    except ValueError:
        return DEFAULT_CHAT_STREAM_TIMEOUT_SECONDS


def _sse_event(event_name: str, payload: dict[str, Any]) -> str:
    return (
        f"event: {event_name}\n"
        f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
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
        execution_strategy=(
            result.planner_state.execution_strategy.value
            if result.planner_state.execution_strategy is not None
            else None
        ),
        knowledge_coverage=(
            result.planner_state.knowledge_coverage.model_dump(mode="json")
            if result.planner_state.knowledge_coverage is not None
            else None
        ),
        execution_branch=result.planner_state.execution_branch,
        discovered_papers=_discovered_papers(result.planner_state),
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


def _discovered_papers(planner_state: PlannerState) -> list[dict[str, Any]]:
    if not any(
        record.decision.tool_name == "discover_papers"
        for record in planner_state.tool_history
    ):
        return []
    display_limit = _discovered_paper_display_limit(planner_state)
    papers = (
        planner_state.runtime_state.selected_papers
        or planner_state.runtime_state.candidate_papers
    )
    return [_compact_discovered_paper(paper) for paper in papers[:display_limit]]


def _discovered_paper_display_limit(planner_state: PlannerState) -> int:
    for record in reversed(planner_state.tool_history):
        if record.decision.tool_name != "discover_papers":
            continue
        max_selected = record.decision.arguments.get("max_selected")
        if isinstance(max_selected, int):
            return max(1, min(max_selected, 20))
    return 20


def _compact_discovered_paper(paper: Paper) -> dict[str, Any]:
    return {
        "paper_id": paper.paper_id,
        "title": paper.title,
        "authors": paper.authors,
        "source": paper.source,
        "url": paper.url,
        "abstract": paper.abstract,
        "published_date": paper.published_date,
        "doi": paper.doi,
        "arxiv_id": paper.arxiv_id,
        "semantic_scholar_id": paper.semantic_scholar_id,
        "external_ids": paper.external_ids,
        "provenance": paper.provenance,
        "venue": paper.venue,
        "citation_count": paper.citation_count,
        "open_access_pdf_url": paper.open_access_pdf_url,
    }


def _paper_from_client_payload(payload: dict[str, Any]) -> Paper:
    normalized = dict(payload)
    if "url" not in normalized and "source_url" in normalized:
        normalized["url"] = normalized["source_url"]
    allowed_fields = set(Paper.model_fields)
    normalized = {
        key: value
        for key, value in normalized.items()
        if key in allowed_fields
    }
    try:
        return Paper.model_validate(normalized)
    except Exception as exc:
        raise ValueError(f"Invalid discovered paper payload: {exc}") from exc


def _requested_paper_ids(
    papers: list[Paper],
    requested_ids: list[str] | None,
) -> list[str]:
    by_id: dict[str, Paper] = {}
    for paper in papers:
        if not paper.paper_id:
            raise ValueError("Discovered paper is missing paper_id.")
        by_id[paper.paper_id] = paper
    if not requested_ids:
        return list(by_id)
    missing = [paper_id for paper_id in requested_ids if paper_id not in by_id]
    if missing:
        raise ValueError(f"Requested paper ids were not included: {missing}")
    return list(dict.fromkeys(requested_ids))


def _merge_save_prepare_status(
    saved: dict[str, Any],
    prepared: dict[str, Any] | None,
) -> str:
    statuses = [saved.get("status")]
    if prepared is not None:
        statuses.append(prepared.get("status"))
    if any(status == "failed" for status in statuses):
        return "failed"
    if any(status == "partial_success" for status in statuses):
        return "partial_success"
    return "success"


def _records_for_paper_ids(paper_ids: list[str]) -> list[dict[str, Any]]:
    store = create_paper_store()
    records = []
    for paper_id in paper_ids:
        record = store.get_paper_record(paper_id)
        if record is not None:
            records.append(record)
    return records


def _delete_paper_workspace(
    *,
    paper_ids: list[str],
    store: Any,
) -> dict[str, Any]:
    unique_paper_ids = list(dict.fromkeys(paper_id for paper_id in paper_ids if paper_id))
    files_observation = remove_fetched_papers(
        state=AgentState(topic="cleanup"),
        paper_ids=unique_paper_ids,
        output_dir=get_settings().papers_dir,
    )
    vector_removed = 0
    vector_errors: list[str] = []
    try:
        vector_store = create_vector_store()
        for paper_id in unique_paper_ids:
            vector_removed += vector_store.delete_by_paper(paper_id)
    except Exception as exc:
        vector_errors.append(str(exc))

    metadata_removed = store.remove_papers(unique_paper_ids)
    return {
        "requested_paper_ids": unique_paper_ids,
        "requested": len(unique_paper_ids),
        "metadata_removed": metadata_removed,
        "files_removed": int(files_observation.get("removed") or 0),
        "files": files_observation,
        "vectors_removed": vector_removed,
        "vector_errors": vector_errors,
    }


def _save_discovered_summary(
    *,
    saved: dict[str, Any],
    prepared: dict[str, Any] | None,
    prepare_job: dict[str, Any] | None = None,
) -> str:
    if prepare_job is not None:
        return (
            f"{saved.get('summary', 'Saved discovered papers.')} "
            f"Queued RAG preparation job {prepare_job.get('job_id')}."
        )
    if prepared is None:
        return saved.get("summary", "Saved discovered papers.")
    return (
        f"{saved.get('summary', 'Saved discovered papers.')} "
        f"{prepared.get('summary', 'Prepared papers for retrieval.')}"
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
        "ingestion_jobs": create_ingestion_job_store(settings).health_check(),
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
