from __future__ import annotations

from typing import Protocol

from app.conversations.models import ConversationMessage
from app.llm.client import LLMClient


class ConversationSummarizer(Protocol):
    def summarize(
        self,
        *,
        existing_summary: str | None,
        messages: list[ConversationMessage],
    ) -> str: ...


class SimpleConversationSummarizer:
    """Deterministic summarizer for tests and local no-LLM workflows."""

    def summarize(
        self,
        *,
        existing_summary: str | None,
        messages: list[ConversationMessage],
    ) -> str:
        lines = []
        if existing_summary:
            lines.append(existing_summary.strip())
        for message in messages:
            if message.content.strip().lower() in {"ok", "thanks", "thank you"}:
                continue
            paper_ids = message.metadata_json.get("paper_ids") or []
            paper_hint = f" papers={paper_ids}" if paper_ids else ""
            lines.append(f"{message.role}: {message.content[:240]}{paper_hint}")
        return "\n".join(line for line in lines if line).strip()


class LLMConversationSummarizer:
    """Replaceable LLM-backed conversation compactor."""

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    def summarize(
        self,
        *,
        existing_summary: str | None,
        messages: list[ConversationMessage],
    ) -> str:
        transcript = "\n".join(
            f"{message.role}: {message.content}" for message in messages
        )
        prompt = f"""
Summarize this research conversation for future context.
Keep paper ids, comparisons, user constraints, unresolved questions, and
research decisions. Do not invent facts. Ignore casual acknowledgements.

Existing summary:
{existing_summary or "(none)"}

Messages:
{transcript}
""".strip()
        return self.llm_client.generate(prompt).strip()
