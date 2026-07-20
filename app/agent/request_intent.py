from __future__ import annotations

import json
import re
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.llm.client import LLMClient


TaskType = Literal[
    "discovery_only",
    "metadata_lookup",
    "factual_answer",
    "summarization",
    "comparison",
    "report",
    "unknown",
]
FinishCondition = Literal[
    "paper_metadata",
    "stored_metadata",
    "retrieved_evidence",
    "paper_summary",
    "report",
    "unknown",
]


class RequestIntent(BaseModel):
    """Planner-facing task intent inferred from the raw user request."""

    model_config = ConfigDict(extra="forbid")

    task_type: TaskType
    topic: str = Field(default="", max_length=300)
    needs_retrieval: bool = False
    needs_ingestion: bool = False
    probe_existing_kb_first: bool = False
    finish_condition: FinishCondition = "unknown"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    rationale: str = Field(default="", max_length=500)


class RequestIntentClassifier(Protocol):
    """Protocol for request intent classifiers."""

    def classify(self, user_request: str) -> RequestIntent:
        """Classify a user request into a compact planner intent."""


class LLMRequestIntentClassifier:
    """LLM-backed classifier for planner-level request intent."""

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    def classify(self, user_request: str) -> RequestIntent:
        """Classify the request using a strict JSON schema prompt."""

        response = self.llm_client.generate(_build_intent_prompt(user_request))
        return parse_request_intent(response)


def parse_request_intent(response_text: str) -> RequestIntent:
    """Parse an LLM response into a RequestIntent."""

    try:
        payload = json.loads(_extract_json(response_text))
        payload = _normalize_intent_payload(payload)
        return RequestIntent.model_validate(payload)
    except (json.JSONDecodeError, ValidationError, ValueError, TypeError) as exc:
        raise ValueError(f"Invalid request intent: {exc}") from exc


def _build_intent_prompt(user_request: str) -> str:
    schema = RequestIntent.model_json_schema()
    payload = {
        "user_request": user_request,
        "schema": schema,
        "instructions": [
            "Classify only the user's task intent, not the research topic domain.",
            "Do not hardcode or normalize domain topics; copy the topic phrase from the request.",
            "Use discovery_only when the user only wants papers to be found or listed.",
            "Use metadata_lookup when the user asks to list, browse, or inspect stored paper metadata.",
            "Use factual_answer when the user asks what, why, how, methods, limitations, findings, or claims.",
            "Use summarization, comparison, or report when the user explicitly asks for those outputs.",
            "needs_retrieval is true when answering requires evidence from paper content.",
            "needs_ingestion is true when newly discovered papers must be prepared before retrieval.",
            "probe_existing_kb_first is true when the assistant should first check already indexed knowledge-base evidence before discovering new papers.",
            "Return only JSON matching the schema.",
        ],
    }
    return (
        "You are a request-intent classifier for a scientific paper assistant.\n"
        "Return only JSON.\n\n"
        + json.dumps(payload, indent=2, sort_keys=True)
    )


def _normalize_intent_payload(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    normalized = dict(payload)
    normalized.setdefault("topic", "")
    normalized.setdefault("needs_retrieval", False)
    normalized.setdefault("needs_ingestion", False)
    normalized.setdefault("probe_existing_kb_first", False)
    normalized.setdefault("finish_condition", "unknown")
    normalized.setdefault("confidence", 0.0)
    normalized.setdefault("rationale", "")
    return normalized


def _extract_json(text: str) -> str:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    if fenced:
        return fenced.group(1)
    return stripped
