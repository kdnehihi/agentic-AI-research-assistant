import pytest

from app.agent.executor import ToolExecutor
from app.agent.execution_plan import ExecutionPlan, PlanStep
from app.agent.grounded_answer import GroundedAnswerService
from app.agent.langgraph_runner import LangGraphAgentRunner
from app.agent.planner_models import CallToolAction, FinishAction
from app.agent.request_intent import RequestIntent
from app.agent.state import AgentState
from app.agent.tool_spec import DiscoverPapersArgs, EnsurePapersRetrievableArgs
from tests.test_planner_executor import FakeRegistry


pytest.importorskip("langgraph")


class ScriptedPlanner:
    def __init__(self, decisions):
        self.decisions = list(decisions)

    def decide(self, state, tool_specs):
        return self.decisions.pop(0)


class FakeAnswerService(GroundedAnswerService):
    def __init__(self):
        pass

    def generate(self, *, state, answer_task):
        return {"answer": "Graph answer [E1].", "answer_task": answer_task}


class StaticIntentClassifier:
    def __init__(self, intent):
        self.intent = intent
        self.requests = []

    def classify(self, user_request):
        self.requests.append(user_request)
        return self.intent


class StaticPlanGenerator:
    def __init__(self, plan):
        self.plan = plan
        self.requests = []

    def generate_plan(self, *, user_request, request_intent, tool_specs):
        self.requests.append(
            {
                "user_request": user_request,
                "request_intent": request_intent,
                "tool_names": [spec.name for spec in tool_specs],
            }
        )
        return self.plan


def _discovery_only_intent(topic="paper topic"):
    return RequestIntent(
        task_type="discovery_only",
        topic=topic,
        needs_retrieval=False,
        needs_ingestion=False,
        probe_existing_kb_first=False,
        finish_condition="paper_metadata",
        confidence=0.95,
        rationale="The user only asked to find papers.",
    )


def _factual_answer_intent(topic="paper topic", *, probe_existing_kb_first=True):
    return RequestIntent(
        task_type="factual_answer",
        topic=topic,
        needs_retrieval=True,
        needs_ingestion=True,
        probe_existing_kb_first=probe_existing_kb_first,
        finish_condition="retrieved_evidence",
        confidence=0.95,
        rationale="The user asked for paper-content evidence.",
    )


def test_langgraph_runner_retrieve_then_finish_generates_answer_without_ingestion():
    registry = FakeRegistry()
    runner = LangGraphAgentRunner(
        planner=ScriptedPlanner(
            [
                CallToolAction(
                    tool_name="retrieve_evidence",
                    arguments={"query": "q"},
                    decision_summary="Need evidence.",
                ),
                FinishAction(
                    answer_task="Answer q.",
                    decision_summary="Evidence exists.",
                ),
            ]
        ),
        executor=ToolExecutor(registry=registry),
        answer_service=FakeAnswerService(),
        policy_enabled=False,
    )

    state = runner.run(user_request="What does p1 say?", runtime_state=AgentState(topic="q"))

    assert state.status == "success"
    assert state.final_answer["answer"] == "Graph answer [E1]."
    assert [call[0] for call in registry.calls] == ["retrieve_evidence"]


def test_langgraph_runner_adds_active_paper_and_section_filters_to_retrieval():
    registry = FakeRegistry()
    fallback_plan = ExecutionPlan(
        goal="Extract a paper section.",
        strategy="Retrieve the requested section from the active paper.",
        steps=[
            PlanStep(
                step_id="retrieve_intro",
                kind="tool",
                tool_name="retrieve_evidence",
                arguments={},
            ),
            PlanStep(
                step_id="finish",
                kind="finish",
                answer_task="Return the introduction.",
            ),
        ],
    )
    plan_generator = StaticPlanGenerator(fallback_plan)
    runner = LangGraphAgentRunner(
        planner=ScriptedPlanner([]),
        executor=ToolExecutor(registry=registry),
        answer_service=FakeAnswerService(),
        intent_classifier=StaticIntentClassifier(
            _factual_answer_intent("SoK paper", probe_existing_kb_first=False)
        ),
        plan_generator=plan_generator,
    )

    state = runner.run(
        user_request="Give me the introduction of the SoK paper.",
        runtime_state=AgentState(topic="SoK paper"),
        active_paper_ids=["arxiv:2603.07379v1"],
    )

    assert state.status == "success"
    assert state.execution_branch == "fast_scoped_retrieval"
    assert plan_generator.requests == []
    assert registry.calls[0] == (
        "retrieve_evidence",
        {
            "query": "Give me the introduction of the SoK paper.",
            "top_k": 5,
            "paper_ids": ["arxiv:2603.07379v1"],
            "section_groups": ["introduction"],
        },
    )


def test_langgraph_runner_auto_recovers_unindexed_retrieval():
    registry = FakeRegistry()
    registry.specs["ensure_papers_retrievable"] = registry.specs[
        "retrieve_evidence"
    ].model_copy(
        update={
            "name": "ensure_papers_retrievable",
            "args_schema": EnsurePapersRetrievableArgs,
        }
    )
    responses = [
        {
            "status": "failed",
            "error_type": "paper_not_retrievable",
            "missing_paper_ids": ["p1"],
            "evidence": [],
            "summary": "Retrieval prerequisite failed because papers are not indexed.",
        },
        {
            "status": "success",
            "ready_paper_ids": ["p1"],
            "summary": "Prepared 1 papers for semantic retrieval; failed 0.",
        },
        {
            "status": "success",
            "query": "q",
            "retrieved": 1,
            "evidence": [{"chunk_id": "c1", "paper_id": "p1", "text": "Evidence"}],
            "summary": "Retrieved 1 evidence chunks.",
        },
    ]

    def execute(tool_name, state, **kwargs):
        registry.calls.append((tool_name, kwargs))
        return responses.pop(0)

    registry.execute = execute
    runner = LangGraphAgentRunner(
        planner=ScriptedPlanner(
            [
                CallToolAction(
                    tool_name="retrieve_evidence",
                    arguments={"query": "q", "paper_ids": ["p1"]},
                    decision_summary="Try retrieval.",
                ),
                FinishAction(
                    answer_task="Answer q.",
                    decision_summary="Recovered evidence exists.",
                ),
            ]
        ),
        executor=ToolExecutor(registry=registry),
        answer_service=FakeAnswerService(),
        policy_enabled=False,
    )

    state = runner.run(user_request="What does p1 say?", runtime_state=AgentState(topic="q"))

    assert state.status == "success"
    assert [call[0] for call in registry.calls] == [
        "retrieve_evidence",
        "ensure_papers_retrievable",
        "retrieve_evidence",
    ]
    assert registry.calls[1][1] == {"paper_ids": ["p1"]}
    assert state.retrievable_paper_ids == ["p1"]
    assert state.retrieved_evidence_ids == ["c1"]


def test_langgraph_runner_discovery_only_can_finish_without_retrieval():
    registry = FakeRegistry()
    registry.specs["discover_papers"] = registry.specs["retrieve_evidence"].model_copy(
        update={"name": "discover_papers"}
    )
    registry.responses["discover_papers"] = {
        "status": "success",
        "selected_paper_ids": ["p1"],
        "candidate_paper_ids": ["p1"],
        "summary": "found",
    }
    runner = LangGraphAgentRunner(
        planner=ScriptedPlanner(
            [
                CallToolAction(
                    tool_name="discover_papers",
                    arguments={"query": "ignored"},
                    decision_summary="Find papers.",
                ),
                FinishAction(
                    answer_task="List discovered papers.",
                    decision_summary="Papers found.",
                ),
            ]
        ),
        executor=ToolExecutor(registry=registry),
        answer_service=FakeAnswerService(),
        policy_enabled=False,
    )

    state = runner.run(user_request="Find papers about agent memory")

    assert state.status == "success"
    assert state.known_paper_ids == ["p1"]
    assert [call[0] for call in registry.calls] == ["discover_papers"]


def test_langgraph_runner_policy_finishes_discovery_only_after_discovery():
    registry = FakeRegistry()
    registry.specs["discover_papers"] = registry.specs["retrieve_evidence"].model_copy(
        update={"name": "discover_papers", "args_schema": DiscoverPapersArgs}
    )
    registry.specs["ensure_papers_retrievable"] = registry.specs[
        "retrieve_evidence"
    ].model_copy(
        update={
            "name": "ensure_papers_retrievable",
            "args_schema": EnsurePapersRetrievableArgs,
        }
    )
    registry.responses["discover_papers"] = {
        "status": "success",
        "selected_paper_ids": ["p-transformer"],
        "candidate_paper_ids": ["p-transformer"],
        "summary": "Discovered 1 candidate papers and selected 1 papers.",
    }
    runner = LangGraphAgentRunner(
        planner=ScriptedPlanner(
            [
                CallToolAction(
                    tool_name="discover_papers",
                    arguments={"user_query": "transformer", "max_results": 4},
                    decision_summary="Find transformer papers.",
                ),
                CallToolAction(
                    tool_name="ensure_papers_retrievable",
                    arguments={"paper_ids": ["p-transformer"]},
                    decision_summary="This should be skipped for discovery-only tasks.",
                ),
            ]
        ),
        executor=ToolExecutor(registry=registry),
        answer_service=FakeAnswerService(),
        intent_classifier=StaticIntentClassifier(_discovery_only_intent("transformer")),
    )

    state = runner.run(user_request="Find paper about transformer")

    assert state.status == "success"
    assert state.known_paper_ids == ["p-transformer"]
    assert state.execution_branch == "fast_discovery"
    assert [call[0] for call in registry.calls] == ["discover_papers"]
    assert registry.calls[0][1] == {"user_query": "transformer", "max_results": 5}


def test_langgraph_runner_policy_can_finish_after_ensure_at_step_budget():
    registry = FakeRegistry()
    registry.specs["ensure_papers_retrievable"] = registry.specs[
        "retrieve_evidence"
    ].model_copy(
        update={
            "name": "ensure_papers_retrievable",
            "args_schema": EnsurePapersRetrievableArgs,
        }
    )
    registry.responses["ensure_papers_retrievable"] = {
        "status": "success",
        "ready_paper_ids": ["p-transformer"],
        "already_ready_paper_ids": [],
        "summary": "Prepared 1 papers for semantic retrieval; failed 0.",
    }
    runner = LangGraphAgentRunner(
        planner=ScriptedPlanner(
            [
                CallToolAction(
                    tool_name="ensure_papers_retrievable",
                    arguments={"paper_ids": ["p-transformer"]},
                    decision_summary="Prepare transformer paper.",
                )
            ]
        ),
        executor=ToolExecutor(registry=registry),
        answer_service=FakeAnswerService(),
        intent_classifier=StaticIntentClassifier(
            _discovery_only_intent("transformer").model_copy(
                update={"confidence": 0.4}
            )
        ),
    )

    state = runner.run(user_request="Find paper about transformer", max_steps=1)

    assert state.status == "success"
    assert state.known_paper_ids == ["p-transformer"]
    assert state.retrievable_paper_ids == ["p-transformer"]
    assert [call[0] for call in registry.calls] == ["ensure_papers_retrievable"]


def test_langgraph_runner_executes_high_level_plan_without_planner_steps():
    registry = FakeRegistry()
    registry.specs["discover_papers"] = registry.specs["retrieve_evidence"].model_copy(
        update={"name": "discover_papers", "args_schema": DiscoverPapersArgs}
    )
    registry.specs["ensure_papers_retrievable"] = registry.specs[
        "retrieve_evidence"
    ].model_copy(
        update={
            "name": "ensure_papers_retrievable",
            "args_schema": EnsurePapersRetrievableArgs,
        }
    )
    responses = [
        {
            "status": "success",
            "selected_paper_ids": ["p-ntp"],
            "candidate_paper_ids": ["p-ntp"],
            "summary": "Discovered one paper.",
        },
        {
            "status": "success",
            "ready_paper_ids": ["p-ntp"],
            "summary": "Prepared one paper.",
        },
        {
            "status": "success",
            "query": "main findings",
            "retrieved": 1,
            "evidence": [{"chunk_id": "c-ntp", "paper_id": "p-ntp", "text": "Evidence"}],
            "summary": "Retrieved evidence.",
        },
    ]

    def execute(tool_name, state, **kwargs):
        registry.calls.append((tool_name, kwargs))
        return responses.pop(0)

    registry.execute = execute
    plan = ExecutionPlan(
        goal="Answer from newly discovered papers.",
        strategy="Discover, prepare, retrieve, finish.",
        steps=[
            PlanStep(
                step_id="discover",
                kind="tool",
                tool_name="discover_papers",
                arguments={"user_query": "neural theorem proving"},
            ),
            PlanStep(
                step_id="prepare",
                kind="tool",
                tool_name="ensure_papers_retrievable",
                argument_sources={"paper_ids": "known_paper_ids"},
            ),
            PlanStep(
                step_id="retrieve",
                kind="tool",
                tool_name="retrieve_evidence",
                arguments={"query": "main findings"},
                argument_sources={"paper_ids": "retrievable_paper_ids"},
            ),
            PlanStep(
                step_id="finish",
                kind="finish",
                answer_task="Answer from retrieved evidence.",
            ),
        ],
    )
    planner = ScriptedPlanner([])
    plan_generator = StaticPlanGenerator(plan)
    runner = LangGraphAgentRunner(
        planner=planner,
        executor=ToolExecutor(registry=registry),
        answer_service=FakeAnswerService(),
        intent_classifier=StaticIntentClassifier(
            _factual_answer_intent(
                "neural theorem proving",
                probe_existing_kb_first=False,
            )
        ),
        plan_generator=plan_generator,
    )

    state = runner.run(
        user_request=(
            "Find recent papers about neural theorem proving and explain the findings."
        )
    )

    assert state.status == "success"
    assert state.execution_branch == "llm_execution_plan"
    assert len(plan_generator.requests) == 1
    assert [call[0] for call in registry.calls] == [
        "discover_papers",
        "ensure_papers_retrievable",
        "retrieve_evidence",
    ]
    assert registry.calls[1][1] == {"paper_ids": ["p-ntp"]}
    assert registry.calls[2][1] == {"query": "main findings", "paper_ids": ["p-ntp"]}
    assert [step.status for step in state.execution_plan.steps[:3]] == [
        "completed",
        "completed",
        "completed",
    ]


def test_langgraph_runner_max_steps_fails_gracefully():
    runner = LangGraphAgentRunner(
        planner=ScriptedPlanner(
            [
                CallToolAction(
                    tool_name="retrieve_evidence",
                    arguments={"query": "q1"},
                    decision_summary="again",
                ),
                CallToolAction(
                    tool_name="retrieve_evidence",
                    arguments={"query": "q2"},
                    decision_summary="again",
                ),
            ]
        ),
        executor=ToolExecutor(registry=FakeRegistry()),
        answer_service=FakeAnswerService(),
        policy_enabled=False,
    )

    state = runner.run(user_request="Loop", max_steps=2)

    assert state.status == "failed"
    assert state.last_error == "Maximum planner steps reached."


def test_langgraph_runner_rejects_finish_too_early_for_factual_task():
    runner = LangGraphAgentRunner(
        planner=ScriptedPlanner(
            [
                FinishAction(
                    answer_task="Answer what the paper says.",
                    decision_summary="Too early.",
                )
            ]
        ),
        executor=ToolExecutor(registry=FakeRegistry()),
        answer_service=FakeAnswerService(),
        policy_enabled=False,
    )

    state = runner.run(user_request="What does the paper say?")

    assert state.status == "failed"
    assert "Finish requires" in state.last_error


def test_langgraph_runner_probes_kb_then_discovers_when_answer_is_missing():
    registry = FakeRegistry()
    registry.specs["discover_papers"] = registry.specs["retrieve_evidence"].model_copy(
        update={"name": "discover_papers", "args_schema": DiscoverPapersArgs}
    )
    registry.specs["ensure_papers_retrievable"] = registry.specs[
        "retrieve_evidence"
    ].model_copy(
        update={
            "name": "ensure_papers_retrievable",
            "args_schema": EnsurePapersRetrievableArgs,
        }
    )
    responses = [
        {
            "status": "success",
            "query": "What did the new memory paper discover?",
            "retrieved": 0,
            "evidence": [],
            "summary": "Retrieved 0 evidence chunks.",
        },
        {
            "status": "success",
            "candidate_paper_ids": ["p-new"],
            "selected_paper_ids": ["p-new"],
            "summary": "Discovered 1 candidate papers and selected 1 papers.",
        },
        {
            "status": "success",
            "ready_paper_ids": ["p-new"],
            "summary": "Prepared 1 papers for semantic retrieval; failed 0.",
        },
        {
            "status": "success",
            "query": "What did the new memory paper discover?",
            "retrieved": 1,
            "evidence": [{"chunk_id": "c-new", "paper_id": "p-new", "text": "Evidence"}],
            "summary": "Retrieved 1 evidence chunks.",
        },
    ]

    def execute(tool_name, state, **kwargs):
        registry.calls.append((tool_name, kwargs))
        return responses.pop(0)

    registry.execute = execute
    runner = LangGraphAgentRunner(
        planner=ScriptedPlanner(
            [
                CallToolAction(
                    tool_name="discover_papers",
                    arguments={
                        "user_query": "new memory paper",
                        "max_results": 5,
                        "max_selected": 1,
                    },
                    decision_summary="No KB evidence exists, so discover a new paper.",
                ),
                CallToolAction(
                    tool_name="ensure_papers_retrievable",
                    arguments={"paper_ids": ["p-new"]},
                    decision_summary="Prepare the discovered paper.",
                ),
                CallToolAction(
                    tool_name="retrieve_evidence",
                    arguments={
                        "query": "What did the new memory paper discover?",
                        "paper_ids": ["p-new"],
                    },
                    decision_summary="Retrieve evidence from the prepared paper.",
                ),
                FinishAction(
                    answer_task="Answer from the newly retrieved evidence.",
                    decision_summary="Evidence is available.",
                ),
            ]
        ),
        executor=ToolExecutor(registry=registry),
        answer_service=FakeAnswerService(),
        intent_classifier=StaticIntentClassifier(_factual_answer_intent("new memory paper")),
    )

    state = runner.run(user_request="What did the new memory paper discover?")

    assert state.status == "success"
    assert [call[0] for call in registry.calls] == [
        "retrieve_evidence",
        "discover_papers",
        "ensure_papers_retrievable",
        "retrieve_evidence",
    ]
    assert registry.calls[0][1] == {
        "query": "What did the new memory paper discover?",
        "top_k": 5,
    }
    assert state.known_paper_ids == ["p-new"]
    assert state.retrievable_paper_ids == ["p-new"]
    assert state.retrieved_evidence_ids == ["c-new"]
