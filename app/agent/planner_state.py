from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.agent.planner_models import CallToolAction, PlannerDecision, ToolObservation
from app.agent.state import AgentState


class ToolExecutionRecord(BaseModel):
    """Trace record for one executed planner action."""

    model_config = ConfigDict(extra="forbid")

    step: int
    decision: CallToolAction
    observation: ToolObservation
    call_fingerprint: str
    latency_ms: float | None = None


class PlannerState(BaseModel):
    """Dynamic planner orchestration state wrapping the runtime AgentState."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    user_request: str
    runtime_state: AgentState

    known_paper_ids: list[str] = Field(default_factory=list)
    saved_paper_ids: list[str] = Field(default_factory=list)
    retrievable_paper_ids: list[str] = Field(default_factory=list)
    retrieved_evidence_ids: list[str] = Field(default_factory=list)
    retrieved_evidence: list[dict[str, Any]] = Field(default_factory=list)
    summary_paper_ids: list[str] = Field(default_factory=list)
    report_available: bool = False

    latest_observation: ToolObservation | None = None
    tool_history: list[ToolExecutionRecord] = Field(default_factory=list)
    pending_decision: PlannerDecision | None = None

    step_count: int = 0
    max_steps: int = 8
    status: Literal["running", "ready_to_answer", "success", "failed"] = "running"
    final_answer: Any | None = None
    last_error: str | None = None
