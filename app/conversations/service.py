from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from app.agent.langgraph_runner import LangGraphAgentRunner
from app.agent.planner_models import FinishAction
from app.agent.planner_state import PlannerState
from app.config import get_settings
from app.conversations.context_builder import ConversationContextBuilder
from app.conversations.models import ConversationMessage, ConversationThread
from app.conversations.repository import AgentRunRepository, ConversationRepository
from app.conversations.sqlite_repository import DEFAULT_USER_ID, sanitize_json
from app.conversations.summarizer import ConversationSummarizer, SimpleConversationSummarizer


@dataclass
class ConversationAgentResult:
    thread: ConversationThread
    user_message: ConversationMessage
    assistant_message: ConversationMessage | None
    planner_state: PlannerState
    run_id: str


class ConversationAgentService:
    """Application service that persists messages and invokes the LangGraph agent."""

    def __init__(
        self,
        *,
        conversation_repository: ConversationRepository,
        run_repository: AgentRunRepository,
        runner: LangGraphAgentRunner,
        context_builder: ConversationContextBuilder | None = None,
        summarizer: ConversationSummarizer | None = None,
        summary_trigger_messages: int = 12,
        summary_keep_recent: int = 6,
    ) -> None:
        settings = get_settings()
        self.conversation_repository = conversation_repository
        self.run_repository = run_repository
        self.runner = runner
        self.context_builder = context_builder or ConversationContextBuilder(
            conversation_repository,
            recent_message_limit=settings.conversation_recent_message_limit,
        )
        self.summarizer = summarizer or SimpleConversationSummarizer()
        self.summary_trigger_messages = (
            summary_trigger_messages
            if summary_trigger_messages != 12
            else settings.conversation_summary_trigger_messages
        )
        self.summary_keep_recent = (
            summary_keep_recent
            if summary_keep_recent != 6
            else settings.conversation_summary_keep_recent
        )

    def create_thread(
        self,
        *,
        title: str,
        user_id: str | None = None,
    ) -> ConversationThread:
        return self.conversation_repository.create_thread(
            title=title,
            user_id=user_id or DEFAULT_USER_ID,
        )

    def run_turn(
        self,
        *,
        user_content: str,
        thread_id: str | None = None,
        title: str | None = None,
        user_id: str | None = None,
        max_steps: int = 8,
    ) -> ConversationAgentResult:
        thread = (
            self.conversation_repository.get_thread(thread_id)
            if thread_id
            else None
        )
        if thread is None:
            thread = self.create_thread(
                title=title or _title_from_user_content(user_content),
                user_id=user_id,
            )

        user_message = self.conversation_repository.append_message(
            thread_id=thread.thread_id,
            role="user",
            content=user_content,
            metadata_json={"message_type": "user_request"},
        )
        run = self.run_repository.start_run(
            thread_id=thread.thread_id,
            user_request_message_id=user_message.message_id,
            graph_thread_id=f"conversation:{thread.thread_id}:run:{user_message.message_id}",
        )
        context = self.context_builder.build(
            thread_id=thread.thread_id,
            before_sequence=user_message.sequence_number,
        )

        started_at = time.perf_counter()
        assistant_message = None
        try:
            state = self.runner.run(
                user_request=user_content,
                max_steps=max_steps,
                thread_id=thread.thread_id,
                run_id=run.run_id,
                recent_messages=[
                    message.model_dump(mode="json")
                    for message in context.recent_messages
                ],
                conversation_summary=context.conversation_summary,
                active_paper_ids=context.active_paper_ids,
                current_user_message_id=user_message.message_id,
            )
            latency_ms = (time.perf_counter() - started_at) * 1000
            self._persist_steps(run.run_id, state)

            if state.status != "success":
                self.run_repository.fail_run(
                    run.run_id,
                    error_type="agent_run_failed",
                    error_message=state.last_error or "Agent run failed.",
                    latency_ms=latency_ms,
                )
                return ConversationAgentResult(
                    thread=thread,
                    user_message=user_message,
                    assistant_message=None,
                    planner_state=state,
                    run_id=run.run_id,
                )

            metadata = _assistant_metadata(run.run_id, state)
            assistant_message = self.conversation_repository.append_message(
                thread_id=thread.thread_id,
                role="assistant",
                content=_assistant_content(state.final_answer),
                metadata_json=metadata,
            )
            state.final_assistant_message_id = assistant_message.message_id
            self.run_repository.complete_run(run.run_id, latency_ms=latency_ms)
            self._maybe_update_summary(thread.thread_id)
            return ConversationAgentResult(
                thread=self.conversation_repository.get_thread(thread.thread_id) or thread,
                user_message=user_message,
                assistant_message=assistant_message,
                planner_state=state,
                run_id=run.run_id,
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - started_at) * 1000
            self.run_repository.fail_run(
                run.run_id,
                error_type=type(exc).__name__,
                error_message=str(exc),
                latency_ms=latency_ms,
            )
            raise

    def _persist_steps(self, run_id: str, state: PlannerState) -> None:
        for record in state.tool_history:
            self.run_repository.append_step(
                run_id=run_id,
                step_number=record.step,
                node_name="execute_tool",
                decision_type=record.decision.action,
                tool_name=record.decision.tool_name,
                arguments_json=record.decision.arguments,
                observation_status=record.observation.status,
                observation_json=record.observation.model_dump(mode="json"),
                latency_ms=record.latency_ms,
            )
        if isinstance(state.pending_decision, FinishAction):
            self.run_repository.append_step(
                run_id=run_id,
                step_number=state.step_count + 1,
                node_name="finish",
                decision_type="finish",
                arguments_json={
                    "answer_task": state.pending_decision.answer_task,
                    "decision_summary": state.pending_decision.decision_summary,
                },
                observation_status=state.status,
                observation_json={
                    "final_answer_available": state.final_answer is not None,
                    "last_error": state.last_error,
                },
                latency_ms=None,
            )

    def _maybe_update_summary(self, thread_id: str) -> None:
        messages = self.conversation_repository.list_messages(thread_id)
        if len(messages) <= self.summary_trigger_messages:
            return
        thread = self.conversation_repository.get_thread(thread_id)
        older_messages = messages[: -self.summary_keep_recent]
        if not older_messages:
            return
        summary = self.summarizer.summarize(
            existing_summary=thread.conversation_summary if thread else None,
            messages=older_messages,
        )
        if summary:
            self.conversation_repository.update_summary(
                thread_id,
                summary,
                summary_updated_at=datetime.now(timezone.utc),
            )


def _title_from_user_content(content: str) -> str:
    compact = " ".join(content.split())
    return compact[:80] or "New research conversation"


def _assistant_content(final_answer: Any) -> str:
    if isinstance(final_answer, dict):
        answer = final_answer.get("answer")
        if isinstance(answer, str):
            return answer
        return str(answer if answer is not None else final_answer)
    return str(final_answer)


def _assistant_metadata(run_id: str, state: PlannerState) -> dict[str, Any]:
    final_answer = state.final_answer if isinstance(state.final_answer, dict) else {}
    evidence_chunks = final_answer.get("evidence_chunks") or []
    cited_chunk_ids = final_answer.get("cited_chunk_ids") or []
    cited_evidence_ids = final_answer.get("cited_evidence_ids") or []
    cited_paper_ids = []
    for chunk in evidence_chunks:
        if isinstance(chunk, dict) and chunk.get("chunk_id") in cited_chunk_ids:
            paper_id = chunk.get("paper_id")
            if paper_id and paper_id not in cited_paper_ids:
                cited_paper_ids.append(paper_id)
    paper_ids = (
        cited_paper_ids
        or state.retrievable_paper_ids
        or state.known_paper_ids
        or state.active_paper_ids
    )
    return sanitize_json(
        {
            "agent_run_id": run_id,
            "message_type": "assistant_answer",
            "paper_ids": paper_ids,
            "active_paper_ids": paper_ids,
            "evidence_ids": cited_evidence_ids or state.retrieved_evidence_ids,
            "citation_ids": cited_chunk_ids,
        }
    )
