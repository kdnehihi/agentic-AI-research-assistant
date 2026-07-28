from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from app.config import get_settings
from app.conversations.models import (
    AgentRun,
    AgentStep,
    ConversationMessage,
    ConversationThread,
    MessageRole,
)
from app.conversations.sqlite_repository import DEFAULT_USER_ID, sanitize_json


class PostgresConversationRepository:
    """Postgres implementation for conversation messages and agent traces."""

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
            raise ValueError("PostgresConversationRepository requires DATABASE_URL.")
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
                    CREATE TABLE IF NOT EXISTS conversation_threads (
                        thread_id TEXT PRIMARY KEY,
                        user_id TEXT,
                        title TEXT NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL,
                        updated_at TIMESTAMPTZ NOT NULL,
                        status TEXT NOT NULL,
                        conversation_summary TEXT,
                        summary_updated_at TIMESTAMPTZ
                    )
                    """
                )
            )
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS conversation_messages (
                        message_id TEXT PRIMARY KEY,
                        thread_id TEXT NOT NULL REFERENCES conversation_threads(thread_id)
                            ON DELETE CASCADE,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL,
                        sequence_number INTEGER NOT NULL,
                        metadata_json JSONB NOT NULL,
                        UNIQUE (thread_id, sequence_number)
                    )
                    """
                )
            )
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS agent_runs (
                        run_id TEXT PRIMARY KEY,
                        thread_id TEXT NOT NULL REFERENCES conversation_threads(thread_id)
                            ON DELETE CASCADE,
                        user_request_message_id TEXT NOT NULL
                            REFERENCES conversation_messages(message_id)
                            ON DELETE CASCADE,
                        status TEXT NOT NULL,
                        started_at TIMESTAMPTZ NOT NULL,
                        completed_at TIMESTAMPTZ,
                        latency_ms REAL,
                        token_usage_json JSONB,
                        estimated_cost REAL,
                        error_type TEXT,
                        error_message TEXT,
                        graph_thread_id TEXT
                    )
                    """
                )
            )
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS agent_steps (
                        step_id TEXT PRIMARY KEY,
                        run_id TEXT NOT NULL REFERENCES agent_runs(run_id)
                            ON DELETE CASCADE,
                        step_number INTEGER NOT NULL,
                        node_name TEXT NOT NULL,
                        decision_type TEXT,
                        tool_name TEXT,
                        arguments_json JSONB,
                        observation_status TEXT,
                        observation_json JSONB,
                        latency_ms REAL,
                        created_at TIMESTAMPTZ NOT NULL,
                        UNIQUE (run_id, step_number)
                    )
                    """
                )
            )
            conn.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS idx_conversation_messages_thread_sequence
                    ON conversation_messages (thread_id, sequence_number)
                    """
                )
            )
            conn.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS idx_conversation_threads_user_updated
                    ON conversation_threads (user_id, updated_at DESC)
                    """
                )
            )
            conn.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS idx_agent_runs_thread_started_at
                    ON agent_runs (thread_id, started_at)
                    """
                )
            )
            conn.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS idx_agent_steps_run_step_number
                    ON agent_steps (run_id, step_number)
                    """
                )
            )

    def health_check(self) -> dict[str, Any]:
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1")).scalar_one()
            return {"status": "ok", "backend": "postgres"}
        except Exception as exc:
            return {"status": "error", "backend": "postgres", "error": str(exc)}

    def create_thread(
        self,
        *,
        title: str,
        user_id: str | None = None,
        thread_id: str | None = None,
    ) -> ConversationThread:
        now = _utc_now()
        thread = ConversationThread(
            thread_id=thread_id or str(uuid4()),
            user_id=user_id or DEFAULT_USER_ID,
            title=title,
            created_at=now,
            updated_at=now,
            status="active",
        )
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO conversation_threads (
                        thread_id, user_id, title, created_at, updated_at, status
                    )
                    VALUES (
                        :thread_id, :user_id, :title, :created_at, :updated_at, :status
                    )
                    """
                ),
                {
                    "thread_id": thread.thread_id,
                    "user_id": thread.user_id,
                    "title": thread.title,
                    "created_at": thread.created_at,
                    "updated_at": thread.updated_at,
                    "status": thread.status,
                },
            )
        return thread

    def get_thread(self, thread_id: str) -> ConversationThread | None:
        with self.engine.connect() as conn:
            row = conn.execute(
                text(
                    """
                    SELECT thread_id, user_id, title, created_at, updated_at, status,
                           conversation_summary, summary_updated_at
                    FROM conversation_threads
                    WHERE thread_id = :thread_id
                    """
                ),
                {"thread_id": thread_id},
            ).mappings().one_or_none()
        return _thread_from_mapping(row) if row else None

    def list_threads(
        self,
        *,
        user_id: str | None = None,
        limit: int = 50,
    ) -> list[ConversationThread]:
        where = "WHERE user_id = :user_id" if user_id is not None else ""
        params = {"user_id": user_id, "limit": limit}
        with self.engine.connect() as conn:
            rows = conn.execute(
                text(
                    f"""
                    SELECT thread_id, user_id, title, created_at, updated_at, status,
                           conversation_summary, summary_updated_at
                    FROM conversation_threads
                    {where}
                    ORDER BY updated_at DESC
                    LIMIT :limit
                    """
                ),
                params,
            ).mappings().all()
        return [_thread_from_mapping(row) for row in rows]

    def append_message(
        self,
        *,
        thread_id: str,
        role: MessageRole,
        content: str,
        metadata_json: dict[str, Any] | None = None,
        message_id: str | None = None,
    ) -> ConversationMessage:
        now = _utc_now()
        safe_metadata = sanitize_json(metadata_json or {})
        with self.engine.begin() as conn:
            sequence_number = int(
                conn.execute(
                    text(
                        """
                        SELECT COALESCE(MAX(sequence_number), 0)
                        FROM conversation_messages
                        WHERE thread_id = :thread_id
                        """
                    ),
                    {"thread_id": thread_id},
                ).scalar_one()
                or 0
            ) + 1
            message = ConversationMessage(
                message_id=message_id or str(uuid4()),
                thread_id=thread_id,
                role=role,
                content=content,
                created_at=now,
                sequence_number=sequence_number,
                metadata_json=safe_metadata,
            )
            conn.execute(
                text(
                    """
                    INSERT INTO conversation_messages (
                        message_id, thread_id, role, content, created_at,
                        sequence_number, metadata_json
                    )
                    VALUES (
                        :message_id, :thread_id, :role, :content, :created_at,
                        :sequence_number, CAST(:metadata_json AS jsonb)
                    )
                    """
                ),
                {
                    "message_id": message.message_id,
                    "thread_id": message.thread_id,
                    "role": message.role,
                    "content": message.content,
                    "created_at": message.created_at,
                    "sequence_number": message.sequence_number,
                    "metadata_json": json.dumps(message.metadata_json, sort_keys=True),
                },
            )
            conn.execute(
                text(
                    """
                    UPDATE conversation_threads
                    SET updated_at = :updated_at
                    WHERE thread_id = :thread_id
                    """
                ),
                {"updated_at": now, "thread_id": thread_id},
            )
        return message

    def list_messages(
        self,
        thread_id: str,
        *,
        limit: int | None = None,
        before_sequence: int | None = None,
    ) -> list[ConversationMessage]:
        clauses = ["thread_id = :thread_id"]
        params: dict[str, Any] = {"thread_id": thread_id}
        if before_sequence is not None:
            clauses.append("sequence_number < :before_sequence")
            params["before_sequence"] = before_sequence
        limit_sql = ""
        if limit is not None:
            limit_sql = "LIMIT :limit"
            params["limit"] = limit
        with self.engine.connect() as conn:
            rows = conn.execute(
                text(
                    f"""
                    SELECT message_id, thread_id, role, content, created_at,
                           sequence_number, metadata_json
                    FROM conversation_messages
                    WHERE {' AND '.join(clauses)}
                    ORDER BY sequence_number DESC
                    {limit_sql}
                    """
                ),
                params,
            ).mappings().all()
        return [_message_from_mapping(row) for row in reversed(rows)]

    def update_summary(
        self,
        thread_id: str,
        summary: str,
        *,
        summary_updated_at: datetime,
    ) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    UPDATE conversation_threads
                    SET conversation_summary = :summary,
                        summary_updated_at = :summary_updated_at,
                        updated_at = :updated_at
                    WHERE thread_id = :thread_id
                    """
                ),
                {
                    "summary": _truncate(summary, 4000),
                    "summary_updated_at": summary_updated_at,
                    "updated_at": _utc_now(),
                    "thread_id": thread_id,
                },
            )

    def delete_thread(self, thread_id: str) -> bool:
        with self.engine.begin() as conn:
            result = conn.execute(
                text("DELETE FROM conversation_threads WHERE thread_id = :thread_id"),
                {"thread_id": thread_id},
            )
        return int(result.rowcount or 0) > 0

    def start_run(
        self,
        *,
        thread_id: str,
        user_request_message_id: str,
        run_id: str | None = None,
        graph_thread_id: str | None = None,
    ) -> AgentRun:
        now = _utc_now()
        run = AgentRun(
            run_id=run_id or str(uuid4()),
            thread_id=thread_id,
            user_request_message_id=user_request_message_id,
            status="running",
            started_at=now,
            graph_thread_id=graph_thread_id,
        )
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO agent_runs (
                        run_id, thread_id, user_request_message_id, status,
                        started_at, graph_thread_id
                    )
                    VALUES (
                        :run_id, :thread_id, :user_request_message_id, :status,
                        :started_at, :graph_thread_id
                    )
                    """
                ),
                {
                    "run_id": run.run_id,
                    "thread_id": run.thread_id,
                    "user_request_message_id": run.user_request_message_id,
                    "status": run.status,
                    "started_at": run.started_at,
                    "graph_thread_id": run.graph_thread_id,
                },
            )
        return run

    def append_step(
        self,
        *,
        run_id: str,
        step_number: int,
        node_name: str,
        decision_type: str | None = None,
        tool_name: str | None = None,
        arguments_json: dict[str, Any] | None = None,
        observation_status: str | None = None,
        observation_json: dict[str, Any] | None = None,
        latency_ms: float | None = None,
        step_id: str | None = None,
    ) -> AgentStep:
        step = AgentStep(
            step_id=step_id or str(uuid4()),
            run_id=run_id,
            step_number=step_number,
            node_name=node_name,
            decision_type=decision_type,
            tool_name=tool_name,
            arguments_json=sanitize_json(arguments_json) if arguments_json else None,
            observation_status=observation_status,
            observation_json=sanitize_json(observation_json) if observation_json else None,
            latency_ms=latency_ms,
            created_at=_utc_now(),
        )
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO agent_steps (
                        step_id, run_id, step_number, node_name, decision_type,
                        tool_name, arguments_json, observation_status, observation_json,
                        latency_ms, created_at
                    )
                    VALUES (
                        :step_id, :run_id, :step_number, :node_name, :decision_type,
                        :tool_name, CAST(:arguments_json AS jsonb),
                        :observation_status, CAST(:observation_json AS jsonb),
                        :latency_ms, :created_at
                    )
                    """
                ),
                {
                    "step_id": step.step_id,
                    "run_id": step.run_id,
                    "step_number": step.step_number,
                    "node_name": step.node_name,
                    "decision_type": step.decision_type,
                    "tool_name": step.tool_name,
                    "arguments_json": (
                        json.dumps(step.arguments_json, sort_keys=True)
                        if step.arguments_json is not None
                        else None
                    ),
                    "observation_status": step.observation_status,
                    "observation_json": (
                        json.dumps(step.observation_json, sort_keys=True)
                        if step.observation_json is not None
                        else None
                    ),
                    "latency_ms": step.latency_ms,
                    "created_at": step.created_at,
                },
            )
        return step

    def complete_run(
        self,
        run_id: str,
        *,
        latency_ms: float | None = None,
        token_usage: dict[str, Any] | None = None,
        estimated_cost: float | None = None,
    ) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    UPDATE agent_runs
                    SET status = 'completed',
                        completed_at = :completed_at,
                        latency_ms = :latency_ms,
                        token_usage_json = CAST(:token_usage_json AS jsonb),
                        estimated_cost = :estimated_cost
                    WHERE run_id = :run_id
                    """
                ),
                {
                    "completed_at": _utc_now(),
                    "latency_ms": latency_ms,
                    "token_usage_json": (
                        json.dumps(sanitize_json(token_usage), sort_keys=True)
                        if token_usage
                        else None
                    ),
                    "estimated_cost": estimated_cost,
                    "run_id": run_id,
                },
            )

    def fail_run(
        self,
        run_id: str,
        *,
        error_type: str,
        error_message: str,
        latency_ms: float | None = None,
    ) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    UPDATE agent_runs
                    SET status = 'failed',
                        completed_at = :completed_at,
                        latency_ms = :latency_ms,
                        error_type = :error_type,
                        error_message = :error_message
                    WHERE run_id = :run_id
                    """
                ),
                {
                    "completed_at": _utc_now(),
                    "latency_ms": latency_ms,
                    "error_type": error_type,
                    "error_message": _truncate(error_message, 1000),
                    "run_id": run_id,
                },
            )

    def get_run(self, run_id: str) -> AgentRun | None:
        with self.engine.connect() as conn:
            row = conn.execute(
                text(
                    """
                    SELECT run_id, thread_id, user_request_message_id, status,
                           started_at, completed_at, latency_ms, token_usage_json,
                           estimated_cost, error_type, error_message, graph_thread_id
                    FROM agent_runs
                    WHERE run_id = :run_id
                    """
                ),
                {"run_id": run_id},
            ).mappings().one_or_none()
        return _run_from_mapping(row) if row else None

    def list_steps(self, run_id: str) -> list[AgentStep]:
        with self.engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT step_id, run_id, step_number, node_name, decision_type,
                           tool_name, arguments_json, observation_status,
                           observation_json, latency_ms, created_at
                    FROM agent_steps
                    WHERE run_id = :run_id
                    ORDER BY step_number ASC
                    """
                ),
                {"run_id": run_id},
            ).mappings().all()
        return [_step_from_mapping(row) for row in rows]


def _thread_from_mapping(row) -> ConversationThread:
    return ConversationThread(
        thread_id=row["thread_id"],
        user_id=row["user_id"],
        title=row["title"],
        created_at=_parse_dt(row["created_at"]),
        updated_at=_parse_dt(row["updated_at"]),
        status=row["status"],
        conversation_summary=row["conversation_summary"],
        summary_updated_at=(
            _parse_dt(row["summary_updated_at"])
            if row["summary_updated_at"]
            else None
        ),
    )


def _message_from_mapping(row) -> ConversationMessage:
    return ConversationMessage(
        message_id=row["message_id"],
        thread_id=row["thread_id"],
        role=row["role"],
        content=row["content"],
        created_at=_parse_dt(row["created_at"]),
        sequence_number=row["sequence_number"],
        metadata_json=_json_mapping(row["metadata_json"]),
    )


def _run_from_mapping(row) -> AgentRun:
    return AgentRun(
        run_id=row["run_id"],
        thread_id=row["thread_id"],
        user_request_message_id=row["user_request_message_id"],
        status=row["status"],
        started_at=_parse_dt(row["started_at"]),
        completed_at=_parse_dt(row["completed_at"]) if row["completed_at"] else None,
        latency_ms=row["latency_ms"],
        token_usage=_json_mapping(row["token_usage_json"])
        if row["token_usage_json"]
        else None,
        estimated_cost=row["estimated_cost"],
        error_type=row["error_type"],
        error_message=row["error_message"],
        graph_thread_id=row["graph_thread_id"],
    )


def _step_from_mapping(row) -> AgentStep:
    return AgentStep(
        step_id=row["step_id"],
        run_id=row["run_id"],
        step_number=row["step_number"],
        node_name=row["node_name"],
        decision_type=row["decision_type"],
        tool_name=row["tool_name"],
        arguments_json=_json_mapping(row["arguments_json"])
        if row["arguments_json"]
        else None,
        observation_status=row["observation_status"],
        observation_json=_json_mapping(row["observation_json"])
        if row["observation_json"]
        else None,
        latency_ms=row["latency_ms"],
        created_at=_parse_dt(row["created_at"]),
    )


def _json_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    parsed = json.loads(value)
    return parsed if isinstance(parsed, dict) else {}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_dt(value: datetime | str) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value)


def _truncate(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[: limit - 3] + "..."
