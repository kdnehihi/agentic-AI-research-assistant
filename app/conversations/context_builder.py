from __future__ import annotations

from collections import OrderedDict
from typing import Iterable

from app.conversations.models import (
    CompactConversationMessage,
    ConversationContext,
    ConversationMessage,
)
from app.conversations.repository import ConversationRepository


class ConversationContextBuilder:
    """Build compact planner context from persistent conversation history."""

    def __init__(
        self,
        repository: ConversationRepository,
        *,
        recent_message_limit: int = 8,
    ) -> None:
        self.repository = repository
        self.recent_message_limit = recent_message_limit

    def build(
        self,
        *,
        thread_id: str,
        before_sequence: int | None = None,
    ) -> ConversationContext:
        thread = self.repository.get_thread(thread_id)
        if thread is None:
            raise ValueError(f"Conversation thread '{thread_id}' does not exist.")

        messages = self.repository.list_messages(
            thread_id,
            limit=self.recent_message_limit,
            before_sequence=before_sequence,
        )
        return ConversationContext(
            thread_id=thread_id,
            conversation_summary=thread.conversation_summary,
            recent_messages=[
                CompactConversationMessage(
                    role=message.role,
                    content=message.content,
                    sequence_number=message.sequence_number,
                    metadata_json=_compact_metadata(message.metadata_json),
                )
                for message in messages
            ],
            active_paper_ids=active_paper_ids_from_messages(messages),
        )


def active_paper_ids_from_messages(
    messages: Iterable[ConversationMessage],
    *,
    limit: int = 12,
) -> list[str]:
    """Extract structured active paper ids from message metadata only."""

    paper_ids: OrderedDict[str, None] = OrderedDict()
    for message in messages:
        metadata = message.metadata_json or {}
        for key in ("paper_ids", "active_paper_ids", "cited_paper_ids"):
            values = metadata.get(key) or []
            if isinstance(values, str):
                values = [values]
            for value in values:
                if isinstance(value, str) and value:
                    paper_ids[value] = None
    return list(paper_ids.keys())[-limit:]


def _compact_metadata(metadata: dict) -> dict:
    allowed = {}
    for key in (
        "paper_ids",
        "active_paper_ids",
        "cited_paper_ids",
        "evidence_ids",
        "citation_ids",
        "agent_run_id",
        "message_type",
    ):
        if key in metadata:
            allowed[key] = metadata[key]
    return allowed
