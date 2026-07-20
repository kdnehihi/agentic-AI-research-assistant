from __future__ import annotations

import json
import re
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.agent.request_intent import RequestIntent
from app.agent.tool_spec import ToolSpec
from app.llm.client import LLMClient


PlanStepKind = Literal["tool", "finish"]
PlanStepStatus = Literal["pending", "completed", "skipped", "failed"]
ArgumentSource = Literal[
    "known_paper_ids",
    "saved_paper_ids",
    "retrievable_paper_ids",
    "retrieved_evidence_ids",
    "active_paper_ids",
]


class PlanStep(BaseModel):
    """One high-level step in an observation-aware execution plan."""

    model_config = ConfigDict(extra="forbid")

    step_id: str = Field(min_length=1, max_length=80)
    kind: PlanStepKind
    tool_name: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    argument_sources: dict[str, ArgumentSource] = Field(default_factory=dict)
    answer_task: str | None = None
    success_condition: str = Field(default="", max_length=500)
    rationale: str = Field(default="", max_length=500)
    status: PlanStepStatus = "pending"


class ExecutionPlan(BaseModel):
    """Planner-visible high-level plan produced before reactive execution."""

    model_config = ConfigDict(extra="forbid")

    goal: str = Field(min_length=1, max_length=500)
    strategy: str = Field(default="", max_length=1000)
    steps: list[PlanStep] = Field(default_factory=list, max_length=10)


class ExecutionPlanGenerator(Protocol):
    """Protocol for high-level plan generators."""

    def generate_plan(
        self,
        *,
        user_request: str,
        request_intent: RequestIntent | None,
        tool_specs: list[ToolSpec],
    ) -> ExecutionPlan:
        """Generate an execution plan for the request."""


class LLMExecutionPlanGenerator:
    """LLM-backed high-level plan generator."""

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    def generate_plan(
        self,
        *,
        user_request: str,
        request_intent: RequestIntent | None,
        tool_specs: list[ToolSpec],
    ) -> ExecutionPlan:
        response = self.llm_client.generate(
            _build_plan_prompt(
                user_request=user_request,
                request_intent=request_intent,
                tool_specs=tool_specs,
            )
        )
        return parse_execution_plan(response)


def parse_execution_plan(response_text: str) -> ExecutionPlan:
    """Parse an LLM response into an ExecutionPlan."""

    try:
        payload = json.loads(_extract_json(response_text))
        payload = _normalize_plan_payload(payload)
        return ExecutionPlan.model_validate(payload)
    except (json.JSONDecodeError, ValidationError, ValueError, TypeError) as exc:
        raise ValueError(f"Invalid execution plan: {exc}") from exc


def _build_plan_prompt(
    *,
    user_request: str,
    request_intent: RequestIntent | None,
    tool_specs: list[ToolSpec],
) -> str:
    tools = [
        {
            "name": spec.name,
            "description": spec.description,
            "input_schema": spec.args_schema.model_json_schema(),
            "read_only": spec.read_only,
            "persistent_side_effect": spec.persistent_side_effect,
            "prerequisites": spec.prerequisites,
        }
        for spec in tool_specs
    ]
    payload = {
        "user_request": user_request,
        "request_intent": (
            request_intent.model_dump(mode="json") if request_intent is not None else None
        ),
        "production_tools": tools,
        "allowed_argument_sources": [
            "known_paper_ids",
            "saved_paper_ids",
            "retrievable_paper_ids",
            "retrieved_evidence_ids",
            "active_paper_ids",
        ],
        "schema": ExecutionPlan.model_json_schema(),
        "instructions": [
            "Create a concise high-level plan before tool execution.",
            "The plan is a guide; reactive execution may stop early or replan after observations.",
            "Use only listed production tools.",
            "Use argument_sources for values that are not known until earlier steps run.",
            "For discovery-only requests, plan discover_papers then finish; do not plan ingestion.",
            "For factual answers about newly found papers, plan discovery, preparation, retrieval, then finish.",
            "For existing-KB factual answers, plan retrieve_evidence then finish.",
            "Return only JSON matching the schema.",
        ],
    }
    return (
        "You are a high-level execution planner for a scientific paper assistant.\n"
        "Return only JSON.\n\n"
        + json.dumps(payload, indent=2, sort_keys=True)
    )


def _normalize_plan_payload(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    normalized = dict(payload)
    normalized.setdefault("strategy", "")
    normalized.setdefault("steps", [])
    return normalized


def _extract_json(text: str) -> str:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    if fenced:
        return fenced.group(1)
    return stripped
