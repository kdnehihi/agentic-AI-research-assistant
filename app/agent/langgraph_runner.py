from __future__ import annotations

from typing import Any, Literal, TypedDict

from app.agent.executor import ToolExecutor
from app.agent.finish_policy import validate_finish
from app.agent.grounded_answer import GroundedAnswerService
from app.agent.planner import Planner
from app.agent.planner_models import CallToolAction, FinishAction
from app.agent.planner_policy import choose_policy_action
from app.agent.planner_state import PlannerState
from app.agent.state import AgentState
from app.agent.tool_spec import ToolSpec


GraphRoute = Literal["decide", "execute_tool", "finish", "max_steps", "done"]


class LangGraphRunnerState(TypedDict):
    """State passed between LangGraph orchestration nodes."""

    planner_state: PlannerState
    tool_specs: list[ToolSpec]


class LangGraphAgentRunner:
    """LangGraph orchestration for the dynamic one-action planner."""

    def __init__(
        self,
        *,
        planner: Planner,
        executor: ToolExecutor | None = None,
        answer_service: GroundedAnswerService | None = None,
        policy_enabled: bool = True,
    ) -> None:
        self.planner = planner
        self.executor = executor or ToolExecutor()
        self.answer_service = answer_service or GroundedAnswerService()
        self.policy_enabled = policy_enabled
        self.graph = self._compile_graph()

    def run(
        self,
        *,
        user_request: str,
        runtime_state: AgentState | None = None,
        max_steps: int = 8,
        thread_id: str | None = None,
        run_id: str | None = None,
        current_user_message_id: str | None = None,
        recent_messages: list[dict[str, Any]] | None = None,
        conversation_summary: str | None = None,
        active_paper_ids: list[str] | None = None,
    ) -> PlannerState:
        """Run the planner graph until success, failure, or step budget exhaustion."""

        planner_state = PlannerState(
            user_request=user_request,
            runtime_state=runtime_state or AgentState(topic=user_request),
            max_steps=max_steps,
            thread_id=thread_id,
            run_id=run_id,
            current_user_message_id=current_user_message_id,
            recent_messages=recent_messages or [],
            conversation_summary=conversation_summary,
            active_paper_ids=active_paper_ids or [],
        )
        result = self.graph.invoke(
            {
                "planner_state": planner_state,
                "tool_specs": self.executor.production_tool_specs(),
            }
        )
        return result["planner_state"]

    def _compile_graph(self):
        try:
            from langgraph.graph import END, StateGraph
        except ImportError as exc:
            raise ImportError(
                "LangGraphAgentRunner requires langgraph. "
                "Install it with `pip install langgraph`."
            ) from exc

        graph = StateGraph(LangGraphRunnerState)
        graph.add_node("decide", self._decide)
        graph.add_node("execute_tool", self._execute_tool)
        graph.add_node("finish", self._finish)
        graph.add_node("max_steps", self._max_steps)

        graph.set_entry_point("decide")
        graph.add_conditional_edges(
            "decide",
            self._route_after_decide,
            {
                "execute_tool": "execute_tool",
                "finish": "finish",
                "max_steps": "max_steps",
                "done": END,
            },
        )
        graph.add_conditional_edges(
            "execute_tool",
            self._route_after_execute,
            {
                "decide": "decide",
                "execute_tool": "execute_tool",
                "finish": "finish",
                "max_steps": "max_steps",
                "done": END,
            },
        )
        graph.add_edge("finish", END)
        graph.add_edge("max_steps", END)
        return graph.compile()

    def _decide(self, graph_state: LangGraphRunnerState) -> LangGraphRunnerState:
        state = graph_state["planner_state"]
        if state.step_count >= state.max_steps:
            return graph_state

        decision = choose_policy_action(state) if self.policy_enabled else None
        if decision is None:
            try:
                decision = self.planner.decide(state, graph_state["tool_specs"])
            except Exception as exc:
                state.status = "failed"
                state.last_error = f"Planner decision failed: {exc}"
                return graph_state

        state.pending_decision = decision
        return graph_state

    def _route_after_decide(self, graph_state: LangGraphRunnerState) -> GraphRoute:
        state = graph_state["planner_state"]
        if state.status == "failed":
            return "done"
        if isinstance(state.pending_decision, FinishAction):
            return "finish"
        if state.step_count >= state.max_steps:
            return "max_steps"
        if isinstance(state.pending_decision, CallToolAction):
            return "execute_tool"
        state.status = "failed"
        state.last_error = "Planner returned an unsupported decision."
        return "done"

    def _execute_tool(self, graph_state: LangGraphRunnerState) -> LangGraphRunnerState:
        state = graph_state["planner_state"]
        decision = state.pending_decision
        if not isinstance(decision, CallToolAction):
            state.status = "failed"
            state.last_error = "Tool execution requires a call_tool planner decision."
            return graph_state
        self.executor.execute(state=state, decision=decision)
        return graph_state

    def _route_after_execute(self, graph_state: LangGraphRunnerState) -> GraphRoute:
        state = graph_state["planner_state"]
        if state.status == "failed":
            return "done"
        if self._route_prerequisite_recovery(graph_state):
            return "execute_tool"
        if self._route_policy_action(graph_state):
            return self._route_after_decide(graph_state)
        if state.step_count >= state.max_steps:
            return "max_steps"
        return "decide"

    def _route_policy_action(self, graph_state: LangGraphRunnerState) -> bool:
        if not self.policy_enabled:
            return False
        state = graph_state["planner_state"]
        decision = choose_policy_action(state)
        if decision is None:
            return False
        state.pending_decision = decision
        return True

    def _route_prerequisite_recovery(
        self,
        graph_state: LangGraphRunnerState,
    ) -> bool:
        state = graph_state["planner_state"]
        observation = state.latest_observation
        decision = state.pending_decision
        if observation is None or not isinstance(decision, CallToolAction):
            return False

        retry_decision = state.retry_decision
        if (
            decision.tool_name == "ensure_papers_retrievable"
            and retry_decision is not None
            and observation.status in {"success", "partial_success"}
        ):
            state.pending_decision = retry_decision
            state.retry_decision = None
            return True

        if (
            decision.tool_name == "retrieve_evidence"
            and observation.status == "prerequisite_missing"
            and observation.error_type == "paper_not_retrievable"
        ):
            missing_paper_ids = observation.result.get("missing_paper_ids") or []
            if not missing_paper_ids:
                return False
            state.retry_decision = decision
            state.pending_decision = CallToolAction(
                tool_name="ensure_papers_retrievable",
                arguments={"paper_ids": missing_paper_ids},
                decision_summary=(
                    "Prepare papers that were missing from the retrieval index, "
                    "then retry the original retrieval."
                ),
            )
            return True

        return False

    def _finish(self, graph_state: LangGraphRunnerState) -> LangGraphRunnerState:
        state = graph_state["planner_state"]
        decision = state.pending_decision
        if not isinstance(decision, FinishAction):
            state.status = "failed"
            state.last_error = "Finish requires a finish planner decision."
            return graph_state

        ok, reason = validate_finish(state, decision)
        if not ok:
            state.status = "failed"
            state.last_error = reason
            return graph_state

        state.status = "ready_to_answer"
        try:
            state.final_answer = self.answer_service.generate(
                state=state,
                answer_task=decision.answer_task,
            )
        except Exception as exc:
            state.status = "failed"
            state.last_error = f"Grounded generation failed: {exc}"
            return graph_state
        state.status = "success"
        return graph_state

    def _max_steps(self, graph_state: LangGraphRunnerState) -> LangGraphRunnerState:
        state = graph_state["planner_state"]
        state.status = "failed"
        state.last_error = "Maximum planner steps reached."
        return graph_state
