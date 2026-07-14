from __future__ import annotations

from app.agent.executor import ToolExecutor
from app.agent.finish_policy import validate_finish
from app.agent.grounded_answer import GroundedAnswerService
from app.agent.planner import Planner
from app.agent.planner_models import CallToolAction, FinishAction
from app.agent.planner_state import PlannerState
from app.agent.state import AgentState


class DynamicAgentRunner:
    """Code-first iterative runner for one-action-at-a-time planning."""

    def __init__(
        self,
        *,
        planner: Planner,
        executor: ToolExecutor | None = None,
        answer_service: GroundedAnswerService | None = None,
    ) -> None:
        self.planner = planner
        self.executor = executor or ToolExecutor()
        self.answer_service = answer_service or GroundedAnswerService()

    def run(
        self,
        *,
        user_request: str,
        runtime_state: AgentState | None = None,
        max_steps: int = 8,
    ) -> PlannerState:
        """Run the dynamic planner loop until finish or step budget exhaustion."""

        state = PlannerState(
            user_request=user_request,
            runtime_state=runtime_state or AgentState(topic=user_request),
            max_steps=max_steps,
        )
        tool_specs = self.executor.production_tool_specs()

        while state.step_count < state.max_steps:
            try:
                decision = self.planner.decide(state, tool_specs)
            except Exception as exc:
                state.status = "failed"
                state.last_error = f"Planner decision failed: {exc}"
                return state

            state.pending_decision = decision
            if isinstance(decision, FinishAction):
                ok, reason = validate_finish(state, decision)
                if not ok:
                    state.status = "failed"
                    state.last_error = reason
                    return state

                state.status = "ready_to_answer"
                try:
                    state.final_answer = self.answer_service.generate(
                        state=state,
                        answer_task=decision.answer_task,
                    )
                except Exception as exc:
                    state.status = "failed"
                    state.last_error = f"Grounded generation failed: {exc}"
                    return state
                state.status = "success"
                return state

            if isinstance(decision, CallToolAction):
                self.executor.execute(state=state, decision=decision)
                continue

            state.status = "failed"
            state.last_error = "Planner returned an unsupported decision."
            return state

        state.status = "failed"
        state.last_error = "Maximum planner steps reached."
        return state

