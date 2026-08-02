from __future__ import annotations

import logging
import queue
import threading
from datetime import datetime, timezone
from typing import Any, Callable, Literal, Protocol
from uuid import uuid4

from pydantic import BaseModel, Field

from app.agent.state import AgentState, Paper


logger = logging.getLogger(__name__)

IngestionJobStatus = Literal["queued", "running", "success", "partial_success", "failed"]
PrepareFunc = Callable[[AgentState], dict[str, Any]]
CompletionCallback = Callable[["IngestionJob"], None]
IngestionJobPayload = tuple[list[dict[str, Any]], list[str], str]


class IngestionJobStore(Protocol):
    def create(self, job: "IngestionJob", payload: IngestionJobPayload) -> None: ...

    def get(self, job_id: str) -> "IngestionJob | None": ...

    def mark_running(self, job_id: str) -> IngestionJobPayload | None: ...

    def mark_completed(
        self,
        job_id: str,
        *,
        status: IngestionJobStatus,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> "IngestionJob | None": ...

    def list_resumable_job_ids(self) -> list[str]: ...

    def health_check(self) -> dict[str, Any]: ...


class IngestionJob(BaseModel):
    """Runtime status for one background paper-ingestion job."""

    job_id: str
    status: IngestionJobStatus = "queued"
    paper_ids: list[str] = Field(default_factory=list)
    knowledge_base_id: str = "default"
    thread_id: str | None = None
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result: dict[str, Any] | None = None
    error: str | None = None


class IngestionJobQueue:
    """Small in-process queue for PDF/text/chunk/embedding preparation."""

    def __init__(
        self,
        *,
        prepare_func: Callable[
            [AgentState],
            dict[str, Any],
        ],
        worker_count: int = 1,
        on_complete: CompletionCallback | None = None,
        store: IngestionJobStore | None = None,
    ) -> None:
        if store is None:
            from app.services.ingestion_job_store import InMemoryIngestionJobStore

            store = InMemoryIngestionJobStore()
        self._prepare_func = prepare_func
        self._on_complete = on_complete
        self._store = store
        self._queue: queue.Queue[str] = queue.Queue()
        self._lock = threading.Lock()
        self._worker_count = max(worker_count, 1)
        self._started = False

    def start(self) -> None:
        """Start daemon worker threads once."""

        with self._lock:
            if self._started:
                return
            self._started = True

        for index in range(self._worker_count):
            thread = threading.Thread(
                target=self._run_worker,
                name=f"ingestion-job-worker-{index + 1}",
                daemon=True,
            )
            thread.start()
        for job_id in self._store.list_resumable_job_ids():
            self._queue.put(job_id)

    def submit(
        self,
        *,
        papers: list[Paper],
        paper_ids: list[str],
        knowledge_base_id: str,
        thread_id: str | None = None,
    ) -> IngestionJob:
        """Persist job metadata and enqueue it for background preparation."""

        now = _utcnow()
        job = IngestionJob(
            job_id=str(uuid4()),
            status="queued",
            paper_ids=list(dict.fromkeys(paper_ids)),
            knowledge_base_id=knowledge_base_id,
            thread_id=thread_id,
            created_at=now,
            updated_at=now,
        )
        payload = (
            [paper.model_dump(mode="json") for paper in papers],
            job.paper_ids,
            knowledge_base_id,
        )

        self._store.create(job, payload)

        self.start()
        self._queue.put(job.job_id)
        return job

    def get(self, job_id: str) -> IngestionJob | None:
        """Return a snapshot of a job, if it exists in this process."""

        return self._store.get(job_id)

    def _run_worker(self) -> None:
        while True:
            job_id = self._queue.get()
            try:
                self._run_job(job_id)
            except Exception:
                logger.exception("Unhandled ingestion job worker error.")
            finally:
                self._queue.task_done()

    def _run_job(self, job_id: str) -> None:
        payload = self._store.mark_running(job_id)
        if payload is None:
            return

        paper_payloads, paper_ids, knowledge_base_id = payload
        state = AgentState(topic=knowledge_base_id)
        state.set_candidate_papers(
            [Paper.model_validate(paper_payload) for paper_payload in paper_payloads]
        )

        try:
            result = self._prepare_func(state, paper_ids=paper_ids)
        except Exception as exc:
            completed_job = self._mark_completed(
                job_id,
                status="failed",
                error=str(exc),
            )
            self._notify_completion(completed_job)
            return

        status = _job_status_from_result(result)
        completed_job = self._mark_completed(job_id, status=status, result=result)
        self._notify_completion(completed_job)

    def _mark_completed(
        self,
        job_id: str,
        *,
        status: IngestionJobStatus,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> IngestionJob | None:
        return self._store.mark_completed(
            job_id,
            status=status,
            result=result,
            error=error,
        )

    def _notify_completion(self, job: IngestionJob | None) -> None:
        if job is None or self._on_complete is None:
            return
        try:
            self._on_complete(job)
        except Exception:
            logger.exception("Unhandled ingestion job completion callback error.")


def _job_status_from_result(result: dict[str, Any]) -> IngestionJobStatus:
    status = result.get("status")
    if status in {"success", "partial_success", "failed"}:
        return status
    return "partial_success"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)
