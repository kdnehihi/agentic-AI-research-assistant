from __future__ import annotations

from typing import Any, Literal, TypedDict

from app.config import get_settings
from app.agent.execution_plan import (
    ExecutionPlanGenerator,
    LLMExecutionPlanGenerator,
    validate_execution_plan,
)
from app.agent.execution_router import (
    build_fast_execution_plan,
    build_strategy_execution_plan,
)
from app.agent.execution_strategy import ExecutionStrategy
from app.agent.executor import ToolExecutor
from app.agent.finish_policy import validate_finish
from app.agent.grounded_answer import GroundedAnswerService
from app.agent.knowledge_coverage import (
    KnowledgeCoverageEvaluator,
    request_requires_freshness,
)
from app.agent.planner import Planner
from app.agent.planner_models import CallToolAction, FinishAction
from app.agent.planner_policy import choose_policy_action
from app.agent.request_intent import (
    LLMRequestIntentClassifier,
    RequestIntentClassifier,
)
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
        intent_classifier: RequestIntentClassifier | None = None,
        plan_generator: ExecutionPlanGenerator | None = None,
        coverage_evaluator: KnowledgeCoverageEvaluator | None = None,
        policy_enabled: bool = True,
    ) -> None:
        self.planner = planner
        self.executor = executor or ToolExecutor()
        self.answer_service = answer_service or GroundedAnswerService()
        self.intent_classifier = intent_classifier or _default_intent_classifier(planner)
        self.plan_generator = plan_generator or _default_plan_generator(planner)
        self.coverage_evaluator = coverage_evaluator or _default_coverage_evaluator(
            planner
        )
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
        self._classify_request_intent(planner_state)
        tool_specs = self.executor.production_tool_specs()
        result = self.graph.invoke(
            {
                "planner_state": planner_state,
                "tool_specs": tool_specs,
            }
        )
        return result["planner_state"]

    def _classify_request_intent(self, state: PlannerState) -> None:
        if self.intent_classifier is None:
            return
        try:
            state.request_intent = self.intent_classifier.classify(state.user_request)
        except Exception:
            state.request_intent = None

    def _generate_execution_plan(
        self,
        state: PlannerState,
        tool_specs: list[ToolSpec],
    ) -> None:
        if self.plan_generator is None:
            return
        try:
            plan = self.plan_generator.generate_plan(
                user_request=state.user_request,
                request_intent=state.request_intent,
                tool_specs=tool_specs,
            )
            state.execution_plan = validate_execution_plan(
                plan,
                tool_specs=tool_specs,
                request_intent=state.request_intent,
            )
        except Exception:
            state.execution_plan = None

    def _compile_graph(self):
        try:
            from langgraph.graph import END, StateGraph
        except ImportError as exc:
            raise ImportError(
                "LangGraphAgentRunner requires langgraph. "
                "Install it with `pip install langgraph`."
            ) from exc

        graph = StateGraph(LangGraphRunnerState)
        graph.add_node("route_request", self._route_request)
        graph.add_node("decide", self._decide)
        graph.add_node("execute_tool", self._execute_tool)
        graph.add_node("finish", self._finish)
        graph.add_node("max_steps", self._max_steps)

        graph.set_entry_point("route_request")
        graph.add_conditional_edges(
            "route_request",
            self._route_after_request,
            {
                "decide": "decide",
                "execute_tool": "execute_tool",
                "finish": "finish",
                "max_steps": "max_steps",
                "done": END,
            },
        )
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

    def _route_request(
        self,
        graph_state: LangGraphRunnerState,
    ) -> LangGraphRunnerState:
        state = graph_state["planner_state"]
        if state.step_count >= state.max_steps or state.status == "failed":
            return graph_state

        if state.execution_plan is None and self._should_try_llm_plan_first(state):
            self._generate_execution_plan(state, graph_state["tool_specs"])
            if state.execution_plan is not None:
                state.execution_branch = "llm_execution_plan_first"
                return graph_state

        if state.execution_plan is None:
            self._choose_initial_execution_strategy(state)
            state.execution_plan = build_strategy_execution_plan(state)

        if state.execution_plan is None:
            state.execution_plan = build_fast_execution_plan(state)

        if state.execution_plan is None:
            self._generate_execution_plan(state, graph_state["tool_specs"])
            if state.execution_plan is not None:
                state.execution_branch = "llm_execution_plan"

        return graph_state

    def _route_after_request(
        self,
        graph_state: LangGraphRunnerState,
    ) -> GraphRoute:
        state = graph_state["planner_state"]
        if state.status == "failed":
            return "done"
        if self._route_policy_action(graph_state):
            return self._route_after_decide(graph_state)
        if state.step_count >= state.max_steps:
            return "max_steps"
        return "decide"

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
        self._mark_current_plan_step_after_tool(state)
        return graph_state

    def _route_after_execute(self, graph_state: LangGraphRunnerState) -> GraphRoute:
        state = graph_state["planner_state"]
        if state.status == "failed":
            return "done"
        if self._route_prerequisite_recovery(graph_state):
            return "execute_tool"
        if self._route_supervisor_handoff(graph_state):
            return self._route_after_decide(graph_state)
        if state.status == "failed":
            return "done"
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

    def _choose_initial_execution_strategy(self, state: PlannerState) -> None:
        if state.execution_strategy is not None:
            return
        intent = state.request_intent
        if intent is None or intent.confidence < 0.5:
            return
        if intent.task_type == "discovery_only":
            state.execution_strategy = ExecutionStrategy.DISCOVERY_ONLY
            return
        if intent.task_type == "metadata_lookup":
            return
        if intent.task_type == "summarization":
            return
        if not intent.needs_retrieval:
            return
        if intent.probe_existing_kb_first:
            state.execution_strategy = ExecutionStrategy.KNOWLEDGE_ONLY
            return
        if request_requires_freshness(state.user_request):
            state.execution_strategy = ExecutionStrategy.DISCOVER_THEN_ANSWER
            return
        state.execution_strategy = ExecutionStrategy.KNOWLEDGE_ONLY

    def _should_try_llm_plan_first(self, state: PlannerState) -> bool:
        if self.plan_generator is None:
            return False
        intent = state.request_intent
        if intent is None:
            return True
        if intent.probe_existing_kb_first:
            return False
        if intent.task_type in {"comparison", "report", "unknown"}:
            return True
        return False

    def _route_supervisor_handoff(
        self,
        graph_state: LangGraphRunnerState,
    ) -> bool:
        state = graph_state["planner_state"]
        observation = state.latest_observation
        if observation is None or observation.tool_name != "retrieve_evidence":
            return False
        if observation.status not in {
            "success",
            "partial_success",
            "prerequisite_missing",
        }:
            return False

        decision = self.coverage_evaluator.evaluate(
            state=state,
            observation=observation,
        )
        state.knowledge_coverage = decision
        state.execution_strategy = decision.recommended_strategy

        if decision.coverage == "sufficient":
            return False
        if state.step_count >= state.max_steps:
            return False
        if decision.recommended_strategy != ExecutionStrategy.DISCOVER_THEN_ANSWER:
            return False
        if _has_executed_tool(state, "discover_papers"):
            state.status = "failed"
            state.last_error = (
                "Knowledge coverage remained insufficient after discovery and "
                "retrieval; refusing another discovery handoff."
            )
            return False

        state.execution_plan = build_strategy_execution_plan(state)
        state.current_plan_step_id = None
        action = choose_policy_action(state) if self.policy_enabled else None
        if action is None:
            return False
        state.pending_decision = action
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
            state.current_plan_step_id = None
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
        self._mark_current_plan_step_completed(state)
        return graph_state

    def _max_steps(self, graph_state: LangGraphRunnerState) -> LangGraphRunnerState:
        state = graph_state["planner_state"]
        state.status = "failed"
        state.last_error = "Maximum planner steps reached."
        return graph_state

    def _mark_current_plan_step_after_tool(self, state: PlannerState) -> None:
        observation = state.latest_observation
        if observation is None:
            return
        if observation.status in {"success", "partial_success", "no_progress"}:
            self._mark_current_plan_step_completed(state)
        elif observation.status in {"invalid_arguments", "tool_error"}:
            self._mark_current_plan_step_failed(state)

    def _mark_current_plan_step_completed(self, state: PlannerState) -> None:
        self._mark_current_plan_step(state, "completed")

    def _mark_current_plan_step_failed(self, state: PlannerState) -> None:
        self._mark_current_plan_step(state, "failed")

    def _mark_current_plan_step(self, state: PlannerState, status: str) -> None:
        if state.execution_plan is None or state.current_plan_step_id is None:
            return
        updated_steps = []
        for step in state.execution_plan.steps:
            if step.step_id == state.current_plan_step_id:
                updated_steps.append(step.model_copy(update={"status": status}))
            else:
                updated_steps.append(step)
        state.execution_plan = state.execution_plan.model_copy(
            update={"steps": updated_steps}
        )
        state.current_plan_step_id = None


def _default_intent_classifier(planner: Planner) -> RequestIntentClassifier | None:
    llm_client = getattr(planner, "llm_client", None)
    if llm_client is None:
        return None
    return LLMRequestIntentClassifier(llm_client)


def _default_plan_generator(planner: Planner) -> ExecutionPlanGenerator | None:
    llm_client = getattr(planner, "llm_client", None)
    if llm_client is None:
        return None
    return LLMExecutionPlanGenerator(llm_client)


def _default_coverage_evaluator(planner: Planner) -> KnowledgeCoverageEvaluator:
    llm_client = getattr(planner, "llm_client", None)
    return KnowledgeCoverageEvaluator(
        llm_client=llm_client,
        use_llm_judge=get_settings().llm_coverage_judge_enabled,
    )


def _has_executed_tool(state: PlannerState, tool_name: str) -> bool:
    return any(record.decision.tool_name == tool_name for record in state.tool_history)
