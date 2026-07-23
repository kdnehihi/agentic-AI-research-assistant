from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.agent.execution_plan import ExecutionPlan
from app.agent.planner_models import CallToolAction, PlannerDecision, ToolObservation
from app.agent.request_intent import RequestIntent
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
    thread_id: str | None = None
    run_id: str | None = None
    current_user_message_id: str | None = None
    final_assistant_message_id: str | None = None
    recent_messages: list[dict[str, Any]] = Field(default_factory=list)
    conversation_summary: str | None = None
    active_paper_ids: list[str] = Field(default_factory=list)
    request_intent: RequestIntent | None = None
    execution_plan: ExecutionPlan | None = None
    execution_branch: str | None = None
    current_plan_step_id: str | None = None

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
    retry_decision: CallToolAction | None = None

    step_count: int = 0
    max_steps: int = 8
    status: Literal["running", "ready_to_answer", "success", "failed"] = "running"
    final_answer: Any | None = None
    last_error: str | None = None
