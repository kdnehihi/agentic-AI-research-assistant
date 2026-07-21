from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.config import get_settings
from app.conversations.models import (
    AgentRun,
    AgentStep,
    ConversationMessage,
    ConversationThread,
    MessageRole,
)


DEFAULT_CONVERSATION_DB_PATH = Path("data/metadata/conversations.sqlite3")
DEFAULT_USER_ID = "local-user"
SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "database_password",
    "password",
    "secret",
    "token",
}


class SQLiteConversationRepository:
    """SQLite implementation for conversation messages and agent traces."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path or get_settings().conversation_db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_threads (
                    thread_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    conversation_summary TEXT,
                    summary_updated_at TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_messages (
                    message_id TEXT PRIMARY KEY,
                    thread_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    sequence_number INTEGER NOT NULL,
                    metadata_json TEXT NOT NULL,
                    FOREIGN KEY (thread_id) REFERENCES conversation_threads (thread_id)
                    ON DELETE CASCADE,
                    UNIQUE (thread_id, sequence_number)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_runs (
                    run_id TEXT PRIMARY KEY,
                    thread_id TEXT NOT NULL,
                    user_request_message_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    latency_ms REAL,
                    token_usage_json TEXT,
                    estimated_cost REAL,
                    error_type TEXT,
                    error_message TEXT,
                    graph_thread_id TEXT,
                    FOREIGN KEY (thread_id) REFERENCES conversation_threads (thread_id)
                    ON DELETE CASCADE,
                    FOREIGN KEY (user_request_message_id)
                    REFERENCES conversation_messages (message_id)
                    ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_steps (
                    step_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    step_number INTEGER NOT NULL,
                    node_name TEXT NOT NULL,
                    decision_type TEXT,
                    tool_name TEXT,
                    arguments_json TEXT,
                    observation_status TEXT,
                    observation_json TEXT,
                    latency_ms REAL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES agent_runs (run_id)
                    ON DELETE CASCADE,
                    UNIQUE (run_id, step_number)
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_conversation_messages_thread_sequence
                ON conversation_messages (thread_id, sequence_number)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_conversation_threads_updated_at
                ON conversation_threads (updated_at)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_agent_runs_thread_started_at
                ON agent_runs (thread_id, started_at)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_agent_steps_run_step_number
                ON agent_steps (run_id, step_number)
                """
            )

    def health_check(self) -> dict[str, Any]:
        """Return a lightweight storage readiness check for the conversation DB."""

        try:
            with self._connect() as conn:
                conn.execute("SELECT 1").fetchone()
            writable = self.db_path.exists() and os.access(self.db_path.parent, os.W_OK)
            return {
                "status": "ok" if writable else "degraded",
                "path": str(self.db_path),
                "writable": writable,
            }
        except Exception as exc:
            return {
                "status": "error",
                "path": str(self.db_path),
                "writable": False,
                "error": str(exc),
            }

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
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO conversation_threads (
                    thread_id, user_id, title, created_at, updated_at, status
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    thread.thread_id,
                    thread.user_id,
                    thread.title,
                    _dt(thread.created_at),
                    _dt(thread.updated_at),
                    thread.status,
                ),
            )
        return thread

    def get_thread(self, thread_id: str) -> ConversationThread | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT thread_id, user_id, title, created_at, updated_at, status,
                       conversation_summary, summary_updated_at
                FROM conversation_threads
                WHERE thread_id = ?
                """,
                (thread_id,),
            ).fetchone()
        return _thread_from_row(row) if row else None

    def list_threads(
        self,
        *,
        user_id: str | None = None,
        limit: int = 50,
    ) -> list[ConversationThread]:
        params: list[Any] = []
        where = ""
        if user_id is not None:
            where = "WHERE user_id = ?"
            params.append(user_id)
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT thread_id, user_id, title, created_at, updated_at, status,
                       conversation_summary, summary_updated_at
                FROM conversation_threads
                {where}
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
        return [_thread_from_row(row) for row in rows]

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
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT COALESCE(MAX(sequence_number), 0)
                FROM conversation_messages
                WHERE thread_id = ?
                """,
                (thread_id,),
            ).fetchone()
            sequence_number = int(row[0] or 0) + 1
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
                """
                INSERT INTO conversation_messages (
                    message_id, thread_id, role, content, created_at,
                    sequence_number, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message.message_id,
                    message.thread_id,
                    message.role,
                    message.content,
                    _dt(message.created_at),
                    message.sequence_number,
                    json.dumps(message.metadata_json, sort_keys=True),
                ),
            )
            conn.execute(
                """
                UPDATE conversation_threads
                SET updated_at = ?
                WHERE thread_id = ?
                """,
                (_dt(now), thread_id),
            )
        return message

    def list_messages(
        self,
        thread_id: str,
        *,
        limit: int | None = None,
        before_sequence: int | None = None,
    ) -> list[ConversationMessage]:
        clauses = ["thread_id = ?"]
        params: list[Any] = [thread_id]
        if before_sequence is not None:
            clauses.append("sequence_number < ?")
            params.append(before_sequence)
        limit_sql = ""
        if limit is not None:
            limit_sql = "LIMIT ?"
            params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT message_id, thread_id, role, content, created_at,
                       sequence_number, metadata_json
                FROM conversation_messages
                WHERE {' AND '.join(clauses)}
                ORDER BY sequence_number DESC
                {limit_sql}
                """,
                params,
            ).fetchall()
        return [_message_from_row(row) for row in reversed(rows)]

    def update_summary(
        self,
        thread_id: str,
        summary: str,
        *,
        summary_updated_at: datetime,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE conversation_threads
                SET conversation_summary = ?, summary_updated_at = ?, updated_at = ?
                WHERE thread_id = ?
                """,
                (_truncate(summary, 4000), _dt(summary_updated_at), _dt(_utc_now()), thread_id),
            )

    def delete_thread(self, thread_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM conversation_threads WHERE thread_id = ?",
                (thread_id,),
            )
        return cursor.rowcount > 0

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
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO agent_runs (
                    run_id, thread_id, user_request_message_id, status,
                    started_at, graph_thread_id
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    run.run_id,
                    run.thread_id,
                    run.user_request_message_id,
                    run.status,
                    _dt(run.started_at),
                    run.graph_thread_id,
                ),
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
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO agent_steps (
                    step_id, run_id, step_number, node_name, decision_type,
                    tool_name, arguments_json, observation_status, observation_json,
                    latency_ms, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    step.step_id,
                    step.run_id,
                    step.step_number,
                    step.node_name,
                    step.decision_type,
                    step.tool_name,
                    json.dumps(step.arguments_json, sort_keys=True)
                    if step.arguments_json is not None
                    else None,
                    step.observation_status,
                    json.dumps(step.observation_json, sort_keys=True)
                    if step.observation_json is not None
                    else None,
                    step.latency_ms,
                    _dt(step.created_at),
                ),
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
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE agent_runs
                SET status = 'completed', completed_at = ?, latency_ms = ?,
                    token_usage_json = ?, estimated_cost = ?
                WHERE run_id = ?
                """,
                (
                    _dt(_utc_now()),
                    latency_ms,
                    json.dumps(sanitize_json(token_usage), sort_keys=True)
                    if token_usage
                    else None,
                    estimated_cost,
                    run_id,
                ),
            )

    def fail_run(
        self,
        run_id: str,
        *,
        error_type: str,
        error_message: str,
        latency_ms: float | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE agent_runs
                SET status = 'failed', completed_at = ?, latency_ms = ?,
                    error_type = ?, error_message = ?
                WHERE run_id = ?
                """,
                (_dt(_utc_now()), latency_ms, error_type, _truncate(error_message, 1000), run_id),
            )

    def get_run(self, run_id: str) -> AgentRun | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT run_id, thread_id, user_request_message_id, status,
                       started_at, completed_at, latency_ms, token_usage_json,
                       estimated_cost, error_type, error_message, graph_thread_id
                FROM agent_runs
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
        return _run_from_row(row) if row else None

    def list_steps(self, run_id: str) -> list[AgentStep]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT step_id, run_id, step_number, node_name, decision_type,
                       tool_name, arguments_json, observation_status,
                       observation_json, latency_ms, created_at
                FROM agent_steps
                WHERE run_id = ?
                ORDER BY step_number ASC
                """,
                (run_id,),
            ).fetchall()
        return [_step_from_row(row) for row in rows]


def sanitize_json(value: Any) -> Any:
    """Remove secrets and trim large values before persistence."""

    if isinstance(value, dict):
        sanitized = {}
        for key, item in value.items():
            key_text = str(key).lower()
            if any(sensitive in key_text for sensitive in SENSITIVE_KEYS):
                continue
            sanitized[str(key)] = sanitize_json(item)
        return sanitized
    if isinstance(value, list):
        return [sanitize_json(item) for item in value[:25]]
    if isinstance(value, str):
        return _truncate(value, 1000)
    return value


def _thread_from_row(row) -> ConversationThread:
    return ConversationThread(
        thread_id=row[0],
        user_id=row[1],
        title=row[2],
        created_at=_parse_dt(row[3]),
        updated_at=_parse_dt(row[4]),
        status=row[5],
        conversation_summary=row[6],
        summary_updated_at=_parse_dt(row[7]) if row[7] else None,
    )


def _message_from_row(row) -> ConversationMessage:
    return ConversationMessage(
        message_id=row[0],
        thread_id=row[1],
        role=row[2],
        content=row[3],
        created_at=_parse_dt(row[4]),
        sequence_number=row[5],
        metadata_json=json.loads(row[6] or "{}"),
    )


def _run_from_row(row) -> AgentRun:
    return AgentRun(
        run_id=row[0],
        thread_id=row[1],
        user_request_message_id=row[2],
        status=row[3],
        started_at=_parse_dt(row[4]),
        completed_at=_parse_dt(row[5]) if row[5] else None,
        latency_ms=row[6],
        token_usage=json.loads(row[7]) if row[7] else None,
        estimated_cost=row[8],
        error_type=row[9],
        error_message=row[10],
        graph_thread_id=row[11],
    )


def _step_from_row(row) -> AgentStep:
    return AgentStep(
        step_id=row[0],
        run_id=row[1],
        step_number=row[2],
        node_name=row[3],
        decision_type=row[4],
        tool_name=row[5],
        arguments_json=json.loads(row[6]) if row[6] else None,
        observation_status=row[7],
        observation_json=json.loads(row[8]) if row[8] else None,
        latency_ms=row[9],
        created_at=_parse_dt(row[10]),
    )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _dt(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _truncate(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[: limit - 3] + "..."
