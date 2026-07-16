from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol

from app.conversations.models import (
    AgentRun,
    AgentStep,
    ConversationMessage,
    ConversationThread,
    MessageRole,
)


class ConversationRepository(Protocol):
    def create_thread(
        self,
        *,
        title: str,
        user_id: str | None = None,
        thread_id: str | None = None,
    ) -> ConversationThread: ...

    def get_thread(self, thread_id: str) -> ConversationThread | None: ...

    def list_threads(
        self,
        *,
        user_id: str | None = None,
        limit: int = 50,
    ) -> list[ConversationThread]: ...

    def append_message(
        self,
        *,
        thread_id: str,
        role: MessageRole,
        content: str,
        metadata_json: dict[str, Any] | None = None,
        message_id: str | None = None,
    ) -> ConversationMessage: ...

    def list_messages(
        self,
        thread_id: str,
        *,
        limit: int | None = None,
        before_sequence: int | None = None,
    ) -> list[ConversationMessage]: ...

    def update_summary(
        self,
        thread_id: str,
        summary: str,
        *,
        summary_updated_at: datetime,
    ) -> None: ...

    def delete_thread(self, thread_id: str) -> bool: ...


class AgentRunRepository(Protocol):
    def start_run(
        self,
        *,
        thread_id: str,
        user_request_message_id: str,
        run_id: str | None = None,
        graph_thread_id: str | None = None,
    ) -> AgentRun: ...

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
    ) -> AgentStep: ...

    def complete_run(
        self,
        run_id: str,
        *,
        latency_ms: float | None = None,
        token_usage: dict[str, Any] | None = None,
        estimated_cost: float | None = None,
    ) -> None: ...

    def fail_run(
        self,
        run_id: str,
        *,
        error_type: str,
        error_message: str,
        latency_ms: float | None = None,
    ) -> None: ...

    def get_run(self, run_id: str) -> AgentRun | None: ...

    def list_steps(self, run_id: str) -> list[AgentStep]: ...
