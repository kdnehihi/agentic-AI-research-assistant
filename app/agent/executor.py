from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from app.agent.observation_factory import ObservationFactory
from app.agent.planner_models import CallToolAction, ToolObservation
from app.agent.planner_state import PlannerState, ToolExecutionRecord
from app.tools.registry import ToolRegistry


class ToolExecutor:
    """Safe executor for one model-selected production tool call."""

    def __init__(
        self,
        *,
        registry: ToolRegistry | None = None,
        observation_factory: ObservationFactory | None = None,
    ) -> None:
        self.registry = registry or ToolRegistry()
        self.observation_factory = observation_factory or ObservationFactory()

    def production_tool_specs(self):
        """Return non-destructive production specs visible to planner v1."""

        specs = []
        for name in self.registry.list_tools(category="production"):
            spec = self.registry.get_tool_spec(name)
            if not spec.destructive and not spec.requires_confirmation:
                specs.append(spec)
        return specs

    def execute(
        self,
        *,
        state: PlannerState,
        decision: CallToolAction,
    ) -> ToolObservation:
        """Validate and execute one tool call, updating planner state."""

        step = state.step_count + 1
        observation: ToolObservation
        fingerprint = ""
        try:
            try:
                spec = self.registry.get_tool_spec(decision.tool_name)
            except ValueError:
                observation = self.observation_factory.from_error(
                    tool_name=decision.tool_name,
                    status="tool_error",
                    summary=f"Tool '{decision.tool_name}' is not registered.",
                    error_type="unknown_tool",
                )
                spec = None

            if spec is None:
                pass
            elif (
                spec.category != "production"
                or spec.destructive
                or spec.requires_confirmation
            ):
                observation = self.observation_factory.from_error(
                    tool_name=decision.tool_name,
                    status="tool_error",
                    summary=f"Tool '{decision.tool_name}' is not allowed for planner use.",
                    error_type="tool_not_allowed",
                )
            else:
                validated_args = spec.args_schema.model_validate(decision.arguments)
                kwargs = validated_args.model_dump(exclude_unset=True)
                fingerprint = call_fingerprint(decision.tool_name, kwargs)
                repeated = self._repeated_successful_call(state, fingerprint)
                if repeated:
                    observation = ToolObservation(
                        tool_name=decision.tool_name,
                        status="no_progress",
                        summary=(
                            "This successful tool call was already executed with "
                            "the same arguments and produced no new state."
                        ),
                        error_type="repeated_tool_call",
                    )
                else:
                    raw_result = self.registry.execute(
                        decision.tool_name,
                        state.runtime_state,
                        **kwargs,
                    )
                    observation = self.observation_factory.from_tool_result(
                        tool_name=decision.tool_name,
                        raw_result=raw_result,
                    )
        except ValidationError as exc:
            observation = self.observation_factory.from_error(
                tool_name=decision.tool_name,
                status="invalid_arguments",
                summary="Tool arguments failed schema validation.",
                error_type="invalid_tool_arguments",
            )
            observation.result["validation_error"] = str(exc)
        except Exception as exc:
            observation = self.observation_factory.from_error(
                tool_name=decision.tool_name,
                status="tool_error",
                summary=f"Tool execution failed: {exc}",
                error_type="unexpected_tool_error",
                retryable=True,
            )

        self._apply_state_changes(state, observation.state_changes)
        state.step_count = step
        state.latest_observation = observation
        state.tool_history.append(
            ToolExecutionRecord(
                step=step,
                decision=decision,
                observation=observation,
                call_fingerprint=fingerprint or call_fingerprint(
                    decision.tool_name,
                    decision.arguments,
                ),
            )
        )
        return observation

    def _repeated_successful_call(self, state: PlannerState, fingerprint: str) -> bool:
        return any(
            record.call_fingerprint == fingerprint
            and record.observation.status == "success"
            for record in state.tool_history
        )

    def _apply_state_changes(
        self,
        state: PlannerState,
        changes: dict[str, Any],
    ) -> None:
        _extend_unique(state.known_paper_ids, changes.get("known_paper_ids_added"))
        _extend_unique(state.saved_paper_ids, changes.get("saved_paper_ids_added"))
        _extend_unique(
            state.retrievable_paper_ids,
            changes.get("retrievable_paper_ids_added"),
        )
        _extend_unique(
            state.retrieved_evidence_ids,
            changes.get("retrieved_evidence_ids_added"),
        )
        _extend_unique(state.summary_paper_ids, changes.get("summary_paper_ids_added"))
        if changes.get("retrieved_evidence_added"):
            _extend_evidence(state.retrieved_evidence, changes["retrieved_evidence_added"])
        if changes.get("report_available"):
            state.report_available = True


def call_fingerprint(tool_name: str, arguments: dict[str, Any]) -> str:
    """Build a deterministic fingerprint for duplicate-call protection."""

    return json.dumps(
        {"tool_name": tool_name, "arguments": arguments},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _extend_unique(target: list[str], values: Any) -> None:
    if not values:
        return
    for value in values:
        if isinstance(value, str) and value not in target:
            target.append(value)


def _extend_evidence(target: list[dict[str, Any]], values: list[dict[str, Any]]) -> None:
    existing = {item.get("chunk_id") for item in target}
    for value in values:
        chunk_id = value.get("chunk_id")
        if chunk_id and chunk_id not in existing:
            target.append(value)
            existing.add(chunk_id)
