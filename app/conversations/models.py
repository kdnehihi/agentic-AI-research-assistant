from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


ThreadStatus = Literal["active", "archived", "deleted"]
MessageRole = Literal["user", "assistant", "system"]
RunStatus = Literal["running", "completed", "failed"]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ConversationThread(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thread_id: str
    user_id: str | None = None
    title: str
    created_at: datetime
    updated_at: datetime
    status: ThreadStatus = "active"
    conversation_summary: str | None = None
    summary_updated_at: datetime | None = None


class ConversationMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message_id: str
    thread_id: str
    role: MessageRole
    content: str
    created_at: datetime
    sequence_number: int
    metadata_json: dict[str, Any] = Field(default_factory=dict)


class AgentRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    thread_id: str
    user_request_message_id: str
    status: RunStatus
    started_at: datetime
    completed_at: datetime | None = None
    latency_ms: float | None = None
    token_usage: dict[str, Any] | None = None
    estimated_cost: float | None = None
    error_type: str | None = None
    error_message: str | None = None
    graph_thread_id: str | None = None


class AgentStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_id: str
    run_id: str
    step_number: int
    node_name: str
    decision_type: str | None = None
    tool_name: str | None = None
    arguments_json: dict[str, Any] | None = None
    observation_status: str | None = None
    observation_json: dict[str, Any] | None = None
    latency_ms: float | None = None
    created_at: datetime


class CompactConversationMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: MessageRole
    content: str
    sequence_number: int
    metadata_json: dict[str, Any] = Field(default_factory=dict)


class ConversationContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thread_id: str
    conversation_summary: str | None = None
    recent_messages: list[CompactConversationMessage] = Field(default_factory=list)
    active_paper_ids: list[str] = Field(default_factory=list)
