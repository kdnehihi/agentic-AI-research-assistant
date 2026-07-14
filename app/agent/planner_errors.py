"""Planner-specific exceptions for the dynamic agent runner."""


class PlannerError(Exception):
    """Base class for dynamic planner failures."""


class UnknownPlannerToolError(PlannerError):
    """Raised when the planner requests a tool outside the allowed catalog."""


class PlannerDecisionValidationError(PlannerError):
    """Raised when an LLM response cannot be parsed as a planner decision."""


class PlannerLLMError(PlannerError):
    """Raised when the planner LLM call fails."""


class ToolArgumentValidationError(PlannerError):
    """Raised when model-provided tool arguments fail schema validation."""


class RepeatedToolCallError(PlannerError):
    """Raised when a tool call is repeated without progress."""


class MaxPlannerStepsError(PlannerError):
    """Raised when the planner reaches its step budget."""


class FinishPolicyError(PlannerError):
    """Raised when a finish decision is not yet supported by artifacts."""

