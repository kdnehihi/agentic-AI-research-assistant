# This file defines the runtime state for the paper research agent.
# It includes normalized papers, paper summaries, tool logs, and AgentState.

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


AgentStatus = Literal[
    "initialized",
    "running",
    "paused",
    "success",
    "partial_success",
    "failed",
]

ToolStatus = Literal[
    "not_started",
    "success",
    "partial_success",
    "failed",
    "skipped",
]


class Paper(BaseModel):
    """A normalized paper record used across the agent."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    title: str
    paper_id: str | None = None

    authors: list[str] = Field(default_factory=list)

    source: str
    url: str

    abstract: str | None = None
    full_text_path: str | None = None
    published_date: str | None = None

    score: float = 0.0
    score_components: dict[str, float] = Field(default_factory=dict)
    relevant_reasons: list[str] = Field(default_factory=list)


class PaperSummary(BaseModel):
    """A structured summary of a selected paper."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    title: str
    paper_id: str | None = None

    one_sentence_summary: str | None = None
    detailed_summary: str | None = None

    method: str | None = None
    main_contribution: str | None = None
    why_it_matters: str | None = None
    limitations: str | None = None

    based_on: Literal["title_only", "abstract_only", "full_text"] = "abstract_only"


class SearchPlan(BaseModel):
    """Structured search terms prepared before calling a paper API."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    original_query: str
    core_terms: list[str] = Field(default_factory=list)
    context_terms: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    arxiv_query: str | None = None
    planner: Literal["rule_based", "llm"] = "rule_based"


class ToolLog(BaseModel):
    """A log entry for one tool call."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    tool_name: str
    input_args: dict[str, Any] = Field(default_factory=dict)

    status: ToolStatus
    output_summary: str | None = None

    error: str | None = None
    latency_ms: float | None = None

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class AgentState(BaseModel):
    """The runtime state of one paper research agent run."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    run_id: str = Field(default_factory=lambda: str(uuid4()))

    topic: str
    max_papers: int = 5

    status: AgentStatus = "initialized"
    current_step: str | None = None

    searched_sources: list[str] = Field(default_factory=list)

    candidate_papers: list[Paper] = Field(default_factory=list)
    selected_papers: list[Paper] = Field(default_factory=list)
    paper_summaries: list[PaperSummary] = Field(default_factory=list)

    search_plan: SearchPlan | None = None
    report: str | None = None
    eval_results: dict[str, Any] | None = None

    tool_call_count: int = 0
    tool_logs: list[ToolLog] = Field(default_factory=list)

    error: str | None = None

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def touch(self) -> None:
        """Update the updated_at timestamp to the current time."""
        self.updated_at = datetime.now(timezone.utc)

    def add_tool_log(self, tool_log: ToolLog) -> None:
        """Add a tool log entry and increment the tool call count."""
        self.tool_logs.append(tool_log)
        self.tool_call_count += 1
        self.touch()

    def add_error(self, error: str) -> None:
        """Add an error message and mark the run as failed."""
        self.error = error
        self.status = "failed"
        self.current_step = None
        self.touch()

    def can_call_tools(self, max_tool_calls: int) -> bool:
        """Check if the agent can call more tools."""
        return (
            self.status in ["running", "paused"]
            and self.tool_call_count < max_tool_calls
        )

    def debug_summary(self) -> str:
        """Return a compact debug summary of the agent's state."""
        return (
            f"Run ID: {self.run_id}\n"
            f"Topic: {self.topic}\n"
            f"Max Papers: {self.max_papers}\n"
            f"Status: {self.status}\n"
            f"Current Step: {self.current_step}\n"
            f"Searched Sources: {self.searched_sources}\n"
            f"Candidate Papers: {len(self.candidate_papers)}\n"
            f"Selected Papers: {len(self.selected_papers)}\n"
            f"Paper Summaries: {len(self.paper_summaries)}\n"
            f"Has Report: {self.report is not None}\n"
            f"Has Eval Results: {self.eval_results is not None}\n"
            f"Tool Call Count: {self.tool_call_count}\n"
            f"Tool Logs Count: {len(self.tool_logs)}\n"
            f"Error: {self.error}\n"
            f"Created At: {self.created_at.isoformat()}\n"
            f"Updated At: {self.updated_at.isoformat()}"
        )

    def set_status(self, status: AgentStatus) -> None:
        """Set the status of the agent and update the timestamp."""
        self.status = status
        self.touch()

    def set_current_step(self, step: str | None) -> None:
        """Set the current step of the agent and update the timestamp."""
        self.current_step = step
        self.touch()

    def add_searched_source(self, source: str) -> None:
        """Add a searched source if it has not already been recorded."""
        if source not in self.searched_sources:
            self.searched_sources.append(source)
        self.touch()

    def add_candidate_paper(self, paper: Paper) -> None:
        """Add a candidate paper to the agent state."""
        self.candidate_papers.append(paper)
        self.touch()

    def set_candidate_papers(self, papers: list[Paper]) -> None:
        """Replace the candidate papers."""
        self.candidate_papers = papers
        self.touch()

    def add_selected_paper(self, paper: Paper) -> None:
        """Add a selected paper to the agent state."""
        self.selected_papers.append(paper)
        self.touch()

    def set_selected_papers(self, papers: list[Paper]) -> None:
        """Replace the selected papers."""
        self.selected_papers = papers
        self.touch()

    def add_paper_summary(self, summary: PaperSummary) -> None:
        """Add a paper summary to the agent state."""
        self.paper_summaries.append(summary)
        self.touch()

    def set_paper_summaries(self, summaries: list[PaperSummary]) -> None:
        """Replace the paper summaries."""
        self.paper_summaries = summaries
        self.touch()

    def set_search_plan(self, search_plan: SearchPlan) -> None:
        """Set the structured search plan."""
        self.search_plan = search_plan
        self.touch()

    def set_report(self, report: str) -> None:
        """Set the generated report."""
        self.report = report
        self.touch()

    def set_eval_results(self, eval_results: dict[str, Any]) -> None:
        """Set evaluation results for this run."""
        self.eval_results = eval_results
        self.touch()

    def mark_running(self) -> None:
        """Mark the run as running."""
        self.status = "running"
        self.touch()

    def mark_paused(self) -> None:
        """Mark the run as paused."""
        self.status = "paused"
        self.touch()

    def mark_success(self) -> None:
        """Mark the run as successfully completed."""
        self.status = "success"
        self.current_step = None
        self.touch()

    def mark_partial_success(self) -> None:
        """Mark the run as partially completed."""
        self.status = "partial_success"
        self.current_step = None
        self.touch()

    def mark_failed(self, message: str | None = None) -> None:
        """Mark the run as failed and optionally set an error message."""
        self.status = "failed"
        self.current_step = None
        if message:
            self.error = message
        self.touch()
