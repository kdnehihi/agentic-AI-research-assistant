from __future__ import annotations

import json
import re

from pydantic import ValidationError

from app.agent.planner_errors import PlannerDecisionValidationError, PlannerLLMError
from app.agent.planner_models import PlannerDecision, PlannerDecisionAdapter
from app.agent.planner_prompt import build_planner_prompt
from app.agent.planner_state import PlannerState
from app.agent.planner_view import build_planner_view
from app.agent.tool_spec import ToolSpec
from app.llm.client import LLMClient


class Planner:
    """LLM-backed one-step planner."""

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    def decide(
        self,
        state: PlannerState,
        tool_specs: list[ToolSpec],
    ) -> PlannerDecision:
        """Ask the LLM for exactly one structured planner decision."""

        prompt = build_planner_prompt(
            user_request=state.user_request,
            tool_specs=tool_specs,
            planner_view=build_planner_view(state),
        )
        try:
            response = self.llm_client.generate(prompt)
        except Exception as exc:
            raise PlannerLLMError(str(exc)) from exc
        return parse_planner_decision(response)


def parse_planner_decision(response_text: str) -> PlannerDecision:
    """Parse a model response into a discriminated planner decision."""

    try:
        payload = json.loads(_extract_json(response_text))
        return PlannerDecisionAdapter.validate_python(payload)
    except (json.JSONDecodeError, ValidationError, ValueError) as exc:
        raise PlannerDecisionValidationError(str(exc)) from exc


def _extract_json(text: str) -> str:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    if fenced:
        return fenced.group(1)
    return stripped

