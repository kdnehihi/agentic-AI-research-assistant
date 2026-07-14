from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


class CallToolAction(BaseModel):
    """Planner decision requesting one production tool call."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["call_tool"] = "call_tool"
    tool_name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)
    decision_summary: str = Field(min_length=1, max_length=500)


class FinishAction(BaseModel):
    """Planner decision indicating enough context exists to answer."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["finish"] = "finish"
    answer_task: str = Field(min_length=1)
    decision_summary: str = Field(min_length=1, max_length=500)


PlannerDecision = Annotated[
    CallToolAction | FinishAction,
    Field(discriminator="action"),
]

PlannerDecisionAdapter = TypeAdapter(PlannerDecision)


ObservationStatus = Literal[
    "success",
    "partial_success",
    "invalid_arguments",
    "prerequisite_missing",
    "tool_error",
    "no_progress",
]


class ToolObservation(BaseModel):
    """Compact planner-facing observation from a domain tool result."""

    model_config = ConfigDict(extra="forbid")

    tool_name: str
    status: ObservationStatus
    summary: str
    result: dict[str, Any] = Field(default_factory=dict)
    state_changes: dict[str, Any] = Field(default_factory=dict)
    error_type: str | None = None
    retryable: bool = False
