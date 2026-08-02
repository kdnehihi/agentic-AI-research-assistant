from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from app.config import get_settings
from app.services.ingestion_jobs import IngestionJob, IngestionJobStatus


IngestionJobPayload = tuple[list[dict[str, Any]], list[str], str]


class IngestionJobStore(Protocol):
    """Persistence boundary for ingestion job status and payload."""

    def create(self, job: IngestionJob, payload: IngestionJobPayload) -> None: ...

    def get(self, job_id: str) -> IngestionJob | None: ...

    def mark_running(self, job_id: str) -> IngestionJobPayload | None: ...

    def mark_completed(
        self,
        job_id: str,
        *,
        status: IngestionJobStatus,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> IngestionJob | None: ...

    def list_resumable_job_ids(self) -> list[str]: ...

    def health_check(self) -> dict[str, Any]: ...


class InMemoryIngestionJobStore:
    """Process-local ingestion job store used by default in local development."""

    def __init__(self) -> None:
        self._jobs: dict[str, IngestionJob] = {}
        self._payloads: dict[str, IngestionJobPayload] = {}
        self._lock = threading.Lock()

    def create(self, job: IngestionJob, payload: IngestionJobPayload) -> None:
        with self._lock:
            self._jobs[job.job_id] = job
            self._payloads[job.job_id] = payload

    def get(self, job_id: str) -> IngestionJob | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.model_copy(deep=True) if job is not None else None

    def mark_running(self, job_id: str) -> IngestionJobPayload | None:
        now = _utcnow()
        with self._lock:
            job = self._jobs.get(job_id)
            payload = self._payloads.get(job_id)
            if job is None or payload is None:
                return None
            self._jobs[job_id] = job.model_copy(
                update={
                    "status": "running",
                    "started_at": now,
                    "updated_at": now,
                }
            )
            return payload

    def mark_completed(
        self,
        job_id: str,
        *,
        status: IngestionJobStatus,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> IngestionJob | None:
        now = _utcnow()
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            completed_job = job.model_copy(
                update={
                    "status": status,
                    "result": result,
                    "error": error,
                    "completed_at": now,
                    "updated_at": now,
                }
            )
            self._jobs[job_id] = completed_job
            self._payloads.pop(job_id, None)
            return completed_job.model_copy(deep=True)

    def list_resumable_job_ids(self) -> list[str]:
        with self._lock:
            return [
                job_id
                for job_id, job in self._jobs.items()
                if job.status in {"queued", "running"} and job_id in self._payloads
            ]

    def health_check(self) -> dict[str, Any]:
        return {"status": "ok", "backend": "memory"}


class SQLiteIngestionJobStore:
    """SQLite ingestion job store for durable local jobs."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path or get_settings().ingestion_job_db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ingestion_jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    paper_ids_json TEXT NOT NULL,
                    knowledge_base_id TEXT NOT NULL,
                    thread_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    result_json TEXT,
                    error TEXT,
                    payload_json TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_status_updated
                ON ingestion_jobs (status, updated_at)
                """
            )

    def create(self, job: IngestionJob, payload: IngestionJobPayload) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO ingestion_jobs (
                    job_id, status, paper_ids_json, knowledge_base_id, thread_id,
                    created_at, updated_at, started_at, completed_at, result_json,
                    error, payload_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                _job_row_values(job, payload_json=_payload_json(payload)),
            )

    def get(self, job_id: str) -> IngestionJob | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT job_id, status, paper_ids_json, knowledge_base_id, thread_id,
                       created_at, updated_at, started_at, completed_at, result_json,
                       error
                FROM ingestion_jobs
                WHERE job_id = ?
                """,
                (job_id,),
            ).fetchone()
        return _job_from_row(row) if row else None

    def mark_running(self, job_id: str) -> IngestionJobPayload | None:
        now = _utcnow()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT payload_json
                FROM ingestion_jobs
                WHERE job_id = ? AND payload_json IS NOT NULL
                """,
                (job_id,),
            ).fetchone()
            if row is None:
                return None
            conn.execute(
                """
                UPDATE ingestion_jobs
                SET status = ?, started_at = COALESCE(started_at, ?), updated_at = ?
                WHERE job_id = ?
                """,
                ("running", _dt(now), _dt(now), job_id),
            )
        return _payload_from_json(row[0])

    def mark_completed(
        self,
        job_id: str,
        *,
        status: IngestionJobStatus,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> IngestionJob | None:
        now = _utcnow()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE ingestion_jobs
                SET status = ?, result_json = ?, error = ?, completed_at = ?,
                    updated_at = ?, payload_json = NULL
                WHERE job_id = ?
                """,
                (
                    status,
                    json.dumps(result, sort_keys=True) if result is not None else None,
                    error,
                    _dt(now),
                    _dt(now),
                    job_id,
                ),
            )
        return self.get(job_id)

    def list_resumable_job_ids(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT job_id
                FROM ingestion_jobs
                WHERE status IN ('queued', 'running') AND payload_json IS NOT NULL
                ORDER BY created_at ASC
                """
            ).fetchall()
        return [str(row[0]) for row in rows]

    def health_check(self) -> dict[str, Any]:
        try:
            with self._connect() as conn:
                conn.execute("SELECT 1").fetchone()
            writable = self.db_path.exists() and os.access(self.db_path.parent, os.W_OK)
            return {
                "status": "ok" if writable else "degraded",
                "backend": "sqlite",
                "path": str(self.db_path),
                "writable": writable,
            }
        except Exception as exc:
            return {
                "status": "error",
                "backend": "sqlite",
                "path": str(self.db_path),
                "error": str(exc),
            }


class PostgresIngestionJobStore:
    """Postgres ingestion job store for cloud deployments."""

    def __init__(
        self,
        *,
        database_url: str | None = None,
        engine: Engine | None = None,
        initialize: bool = True,
    ) -> None:
        settings = get_settings()
        self.database_url = database_url or settings.database_url
        if engine is None and not self.database_url:
            raise ValueError("PostgresIngestionJobStore requires DATABASE_URL.")
        self.engine = engine or create_engine(
            self.database_url,
            pool_pre_ping=True,
            future=True,
        )
        if initialize:
            self._init_db()

    def _init_db(self) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS ingestion_jobs (
                        job_id TEXT PRIMARY KEY,
                        status TEXT NOT NULL,
                        paper_ids_json JSONB NOT NULL,
                        knowledge_base_id TEXT NOT NULL,
                        thread_id TEXT,
                        created_at TIMESTAMPTZ NOT NULL,
                        updated_at TIMESTAMPTZ NOT NULL,
                        started_at TIMESTAMPTZ,
                        completed_at TIMESTAMPTZ,
                        result_json JSONB,
                        error TEXT,
                        payload_json JSONB
                    )
                    """
                )
            )
            conn.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_status_updated
                    ON ingestion_jobs (status, updated_at)
                    """
                )
            )

    def create(self, job: IngestionJob, payload: IngestionJobPayload) -> None:
        values = _job_mapping(job, payload_json=_payload_json(payload))
        values["paper_ids_json"] = json.dumps(values["paper_ids_json"], sort_keys=True)
        values["payload_json"] = json.dumps(values["payload_json"], sort_keys=True)
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO ingestion_jobs (
                        job_id, status, paper_ids_json, knowledge_base_id, thread_id,
                        created_at, updated_at, started_at, completed_at, result_json,
                        error, payload_json
                    )
                    VALUES (
                        :job_id, :status, CAST(:paper_ids_json AS jsonb),
                        :knowledge_base_id, :thread_id, :created_at, :updated_at,
                        :started_at, :completed_at, :result_json, :error,
                        CAST(:payload_json AS jsonb)
                    )
                    """
                ),
                values,
            )

    def get(self, job_id: str) -> IngestionJob | None:
        with self.engine.connect() as conn:
            row = conn.execute(
                text(
                    """
                    SELECT job_id, status, paper_ids_json, knowledge_base_id, thread_id,
                           created_at, updated_at, started_at, completed_at,
                           result_json, error
                    FROM ingestion_jobs
                    WHERE job_id = :job_id
                    """
                ),
                {"job_id": job_id},
            ).mappings().one_or_none()
        return _job_from_mapping(row) if row else None

    def mark_running(self, job_id: str) -> IngestionJobPayload | None:
        now = _utcnow()
        with self.engine.begin() as conn:
            row = conn.execute(
                text(
                    """
                    SELECT payload_json
                    FROM ingestion_jobs
                    WHERE job_id = :job_id AND payload_json IS NOT NULL
                    """
                ),
                {"job_id": job_id},
            ).mappings().one_or_none()
            if row is None:
                return None
            conn.execute(
                text(
                    """
                    UPDATE ingestion_jobs
                    SET status = :status,
                        started_at = COALESCE(started_at, :started_at),
                        updated_at = :updated_at
                    WHERE job_id = :job_id
                    """
                ),
                {
                    "status": "running",
                    "started_at": now,
                    "updated_at": now,
                    "job_id": job_id,
                },
            )
        return _payload_from_json(row["payload_json"])

    def mark_completed(
        self,
        job_id: str,
        *,
        status: IngestionJobStatus,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> IngestionJob | None:
        now = _utcnow()
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    UPDATE ingestion_jobs
                    SET status = :status,
                        result_json = CAST(:result_json AS jsonb),
                        error = :error,
                        completed_at = :completed_at,
                        updated_at = :updated_at,
                        payload_json = NULL
                    WHERE job_id = :job_id
                    """
                ),
                {
                    "status": status,
                    "result_json": (
                        json.dumps(result, sort_keys=True)
                        if result is not None
                        else None
                    ),
                    "error": error,
                    "completed_at": now,
                    "updated_at": now,
                    "job_id": job_id,
                },
            )
        return self.get(job_id)

    def list_resumable_job_ids(self) -> list[str]:
        with self.engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT job_id
                    FROM ingestion_jobs
                    WHERE status IN ('queued', 'running') AND payload_json IS NOT NULL
                    ORDER BY created_at ASC
                    """
                )
            ).mappings().all()
        return [str(row["job_id"]) for row in rows]

    def health_check(self) -> dict[str, Any]:
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1")).scalar_one()
            return {"status": "ok", "backend": "postgres"}
        except Exception as exc:
            return {"status": "error", "backend": "postgres", "error": str(exc)}


class _PayloadModel(BaseModel):
    papers: list[dict[str, Any]]
    paper_ids: list[str]
    knowledge_base_id: str


def _payload_json(payload: IngestionJobPayload) -> dict[str, Any]:
    papers, paper_ids, knowledge_base_id = payload
    return {
        "papers": papers,
        "paper_ids": paper_ids,
        "knowledge_base_id": knowledge_base_id,
    }


def _payload_from_json(value: str | dict[str, Any]) -> IngestionJobPayload:
    payload = value if isinstance(value, dict) else json.loads(value)
    model = _PayloadModel.model_validate(payload)
    return (model.papers, model.paper_ids, model.knowledge_base_id)


def _job_row_values(
    job: IngestionJob,
    *,
    payload_json: dict[str, Any] | None,
) -> tuple[Any, ...]:
    return (
        job.job_id,
        job.status,
        json.dumps(job.paper_ids, sort_keys=True),
        job.knowledge_base_id,
        job.thread_id,
        _dt(job.created_at),
        _dt(job.updated_at),
        _dt(job.started_at) if job.started_at else None,
        _dt(job.completed_at) if job.completed_at else None,
        json.dumps(job.result, sort_keys=True) if job.result is not None else None,
        job.error,
        json.dumps(payload_json, sort_keys=True) if payload_json is not None else None,
    )


def _job_mapping(
    job: IngestionJob,
    *,
    payload_json: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "job_id": job.job_id,
        "status": job.status,
        "paper_ids_json": job.paper_ids,
        "knowledge_base_id": job.knowledge_base_id,
        "thread_id": job.thread_id,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "started_at": job.started_at,
        "completed_at": job.completed_at,
        "result_json": (
            json.dumps(job.result, sort_keys=True) if job.result is not None else None
        ),
        "error": job.error,
        "payload_json": payload_json,
    }


def _job_from_row(row: Any) -> IngestionJob:
    return IngestionJob(
        job_id=row[0],
        status=row[1],
        paper_ids=json.loads(row[2] or "[]"),
        knowledge_base_id=row[3],
        thread_id=row[4],
        created_at=_parse_dt(row[5]),
        updated_at=_parse_dt(row[6]),
        started_at=_parse_dt(row[7]) if row[7] else None,
        completed_at=_parse_dt(row[8]) if row[8] else None,
        result=json.loads(row[9]) if row[9] else None,
        error=row[10],
    )


def _job_from_mapping(row: Any) -> IngestionJob:
    return IngestionJob(
        job_id=row["job_id"],
        status=row["status"],
        paper_ids=_json_list(row["paper_ids_json"]),
        knowledge_base_id=row["knowledge_base_id"],
        thread_id=row["thread_id"],
        created_at=_parse_dt(row["created_at"]),
        updated_at=_parse_dt(row["updated_at"]),
        started_at=_parse_dt(row["started_at"]) if row["started_at"] else None,
        completed_at=(
            _parse_dt(row["completed_at"]) if row["completed_at"] else None
        ),
        result=_json_dict(row["result_json"]),
        error=row["error"],
    )


def _json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    if not value:
        return []
    parsed = json.loads(value)
    return [item for item in parsed if isinstance(item, str)]


def _json_dict(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    parsed = json.loads(value)
    return parsed if isinstance(parsed, dict) else None


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _dt(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _parse_dt(value: datetime | str) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value)
