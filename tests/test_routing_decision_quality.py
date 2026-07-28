import pytest

from app.agent.executor import ToolExecutor
from app.agent.execution_plan import ExecutionPlan, PlanStep
from app.agent.grounded_answer import GroundedAnswerService
from app.agent.langgraph_runner import LangGraphAgentRunner
from app.agent.planner_models import CallToolAction, FinishAction
from app.agent.request_intent import RequestIntent
from app.agent.state import AgentState
from app.agent.tool_spec import (
    DiscoverPapersArgs,
    EnsurePapersRetrievableArgs,
    RetrieveEvidenceArgs,
    SavePapersToKbArgs,
    ToolSpec,
)


pytest.importorskip("langgraph")


class EmptyPlanner:
    def decide(self, state, tool_specs):
        raise AssertionError("The deterministic route should not call the planner.")


class ScriptedPlanner:
    def __init__(self, decisions):
        self.decisions = list(decisions)

    def decide(self, state, tool_specs):
        return self.decisions.pop(0)


class StaticIntentClassifier:
    def __init__(self, intent):
        self.intent = intent

    def classify(self, user_request):
        del user_request
        return self.intent


class StaticPlanGenerator:
    def __init__(self, plan):
        self.plan = plan

    def generate_plan(self, *, user_request, request_intent, tool_specs):
        del user_request, request_intent, tool_specs
        return self.plan


class FakeAnswerService(GroundedAnswerService):
    def __init__(self):
        pass

    def generate(self, *, state, answer_task):
        return {"answer": "Grounded answer.", "answer_task": answer_task}


class RoutingRegistry:
    def __init__(self, responses):
        self.calls = []
        self.responses = list(responses)
        self.specs = {
            "retrieve_evidence": ToolSpec(
                name="retrieve_evidence",
                description="Retrieve evidence.",
                args_schema=RetrieveEvidenceArgs,
                read_only=True,
                category="production",
            ),
            "discover_papers": ToolSpec(
                name="discover_papers",
                description="Discover papers.",
                args_schema=DiscoverPapersArgs,
                read_only=False,
                runtime_state_mutation=True,
                category="production",
            ),
            "ensure_papers_retrievable": ToolSpec(
                name="ensure_papers_retrievable",
                description="Prepare papers for retrieval.",
                args_schema=EnsurePapersRetrievableArgs,
                read_only=False,
                runtime_state_mutation=True,
                persistent_side_effect=True,
                category="production",
            ),
            "save_papers_to_kb": ToolSpec(
                name="save_papers_to_kb",
                description="Save paper metadata.",
                args_schema=SavePapersToKbArgs,
                read_only=False,
                persistent_side_effect=True,
                category="production",
            ),
        }

    def list_tools(self, category=None):
        return [
            name
            for name, spec in self.specs.items()
            if category is None or spec.category == category
        ]

    def get_tool_spec(self, name):
        return self.specs[name]

    def execute(self, tool_name, state, **kwargs):
        del state
        self.calls.append((tool_name, kwargs))
        if (
            tool_name == "save_papers_to_kb"
            and (
                not self.responses
                or self.responses[0]["tool_name"] != "save_papers_to_kb"
            )
        ):
            paper_ids = kwargs.get("paper_ids") or []
            return {
                "status": "success",
                "inserted_paper_ids": paper_ids,
                "updated_paper_ids": [],
                "already_present_paper_ids": [],
                "failed": [],
                "summary": f"Saved {len(paper_ids)} papers.",
            }
        response = self.responses.pop(0)
        assert response["tool_name"] == tool_name
        return response["result"]


def test_sufficient_existing_kb_evidence_finishes_without_discovery_when_no_plan_preempts_probe():
    registry = RoutingRegistry(
        [
            {
                "tool_name": "retrieve_evidence",
                "result": _retrieval_response(["c-kb"], ["p-kb"]),
            }
        ]
    )
    runner = _runner(
        registry=registry,
        intent=_factual_intent("DPO", probe_existing_kb_first=True),
        plan=None,
    )

    state = runner.run(user_request="Based on my saved papers, explain DPO.")

    assert state.status == "success"
    assert [call[0] for call in registry.calls] == ["retrieve_evidence"]
    assert state.retrieved_evidence_ids == ["c-kb"]


def test_sufficient_existing_kb_evidence_short_circuits_discovery_plan():
    registry = RoutingRegistry(
        [
            {
                "tool_name": "retrieve_evidence",
                "result": _retrieval_response(["c-kb"], ["p-kb"]),
            }
        ]
    )
    runner = _runner(
        registry=registry,
        intent=_factual_intent("DPO", probe_existing_kb_first=True),
        plan=_discover_then_answer_plan("DPO"),
    )

    state = runner.run(user_request="Based on my saved papers, explain DPO.")

    assert state.status == "success"
    assert [call[0] for call in registry.calls] == ["retrieve_evidence"]
    assert state.execution_branch == "strategy_knowledge_only"
    assert state.retrieved_evidence_ids == ["c-kb"]


def test_no_existing_kb_evidence_can_discover_ingest_and_retrieve_with_reactive_planner():
    registry = RoutingRegistry(
        [
            {
                "tool_name": "retrieve_evidence",
                "result": _retrieval_response([], []),
            },
            {
                "tool_name": "discover_papers",
                "result": _discovery_response(["p-new"]),
            },
            {
                "tool_name": "ensure_papers_retrievable",
                "result": _ensure_response(["p-new"]),
            },
            {
                "tool_name": "retrieve_evidence",
                "result": _retrieval_response(["c-new"], ["p-new"]),
            },
        ]
    )
    runner = LangGraphAgentRunner(
        planner=ScriptedPlanner(
            [
                CallToolAction(
                    tool_name="discover_papers",
                    arguments={"user_query": "agent memory", "max_selected": 1},
                    decision_summary="No KB evidence exists, discover papers.",
                ),
                CallToolAction(
                    tool_name="ensure_papers_retrievable",
                    arguments={"paper_ids": ["p-new"]},
                    decision_summary="Prepare discovered paper.",
                ),
                CallToolAction(
                    tool_name="retrieve_evidence",
                    arguments={"query": "agent memory", "paper_ids": ["p-new"]},
                    decision_summary="Retrieve from prepared paper.",
                ),
                FinishAction(
                    answer_task="Answer from retrieved evidence.",
                    decision_summary="Evidence is available.",
                ),
            ]
        ),
        executor=ToolExecutor(registry=registry),
        answer_service=FakeAnswerService(),
        intent_classifier=StaticIntentClassifier(
            _factual_intent("agent memory", probe_existing_kb_first=True)
        ),
        plan_generator=None,
    )

    state = runner.run(
        user_request="What did recent agent memory papers discover?",
        max_steps=6,
    )

    assert state.status == "success"
    assert [call[0] for call in registry.calls] == [
        "retrieve_evidence",
        "discover_papers",
        "save_papers_to_kb",
        "ensure_papers_retrievable",
        "retrieve_evidence",
    ]
    assert state.known_paper_ids == ["p-new"]
    assert state.retrievable_paper_ids == ["p-new"]
    assert state.retrieved_evidence_ids == ["c-new"]


def test_no_existing_kb_evidence_can_discover_ingest_and_retrieve_from_plan_after_probe():
    registry = RoutingRegistry(
        [
            {
                "tool_name": "retrieve_evidence",
                "result": _retrieval_response([], []),
            },
            {
                "tool_name": "discover_papers",
                "result": _discovery_response(["p-new"]),
            },
            {
                "tool_name": "ensure_papers_retrievable",
                "result": _ensure_response(["p-new"]),
            },
            {
                "tool_name": "retrieve_evidence",
                "result": _retrieval_response(["c-new"], ["p-new"]),
            },
        ]
    )
    runner = _runner(
        registry=registry,
        intent=_factual_intent("agent memory", probe_existing_kb_first=True),
        plan=_discover_then_answer_plan("agent memory"),
    )

    state = runner.run(
        user_request="What did recent agent memory papers discover?",
        max_steps=6,
    )

    assert state.status == "success"
    assert [call[0] for call in registry.calls] == [
        "retrieve_evidence",
        "discover_papers",
        "save_papers_to_kb",
        "ensure_papers_retrievable",
        "retrieve_evidence",
    ]
    assert state.known_paper_ids == ["p-new"]
    assert state.retrievable_paper_ids == ["p-new"]
    assert state.retrieved_evidence_ids == ["c-new"]


def test_metadata_exists_but_unindexed_paper_is_prepared_then_retrieved():
    registry = RoutingRegistry(
        [
            {
                "tool_name": "retrieve_evidence",
                "result": {
                    "status": "failed",
                    "error_type": "paper_not_retrievable",
                    "missing_paper_ids": ["p-saved"],
                    "evidence": [],
                    "summary": "Retrieval prerequisite failed because papers are not indexed.",
                },
            },
            {
                "tool_name": "ensure_papers_retrievable",
                "result": _ensure_response(["p-saved"]),
            },
            {
                "tool_name": "retrieve_evidence",
                "result": _retrieval_response(["c-saved"], ["p-saved"]),
            },
        ]
    )
    runner = _runner(
        registry=registry,
        intent=_factual_intent("saved paper", probe_existing_kb_first=False),
        plan=None,
    )

    state = runner.run(
        user_request="Explain the limitations of the saved paper.",
        active_paper_ids=["p-saved"],
        max_steps=4,
    )

    assert state.status == "success"
    assert [call[0] for call in registry.calls] == [
        "retrieve_evidence",
        "ensure_papers_retrievable",
        "retrieve_evidence",
    ]
    assert registry.calls[0][1]["paper_ids"] == ["p-saved"]
    assert state.retrievable_paper_ids == ["p-saved"]
    assert state.retrieved_evidence_ids == ["c-saved"]


def test_handoff_respects_max_step_budget_without_uncontrolled_looping():
    registry = RoutingRegistry(
        [
            {
                "tool_name": "retrieve_evidence",
                "result": _retrieval_response([], []),
            },
            {
                "tool_name": "discover_papers",
                "result": _discovery_response(["p-new"]),
            },
        ]
    )
    runner = _runner(
        registry=registry,
        intent=_factual_intent("new method", probe_existing_kb_first=True),
        plan=_discover_then_answer_plan("new method"),
    )

    state = runner.run(
        user_request="Find recent work on the new method and explain it.",
        max_steps=2,
    )

    assert state.status == "failed"
    assert state.last_error == "Maximum planner steps reached."
    assert [call[0] for call in registry.calls] == [
        "retrieve_evidence",
        "discover_papers",
    ]


def test_partial_comparison_evidence_should_discover_missing_side_before_finishing():
    registry = RoutingRegistry(
        [
            {
                "tool_name": "retrieve_evidence",
                "result": _retrieval_response(["c-a"], ["p-a"]),
            },
            {
                "tool_name": "discover_papers",
                "result": _discovery_response(["p-b"]),
            },
            {
                "tool_name": "ensure_papers_retrievable",
                "result": _ensure_response(["p-b"]),
            },
            {
                "tool_name": "retrieve_evidence",
                "result": _retrieval_response(["c-a2", "c-b"], ["p-a", "p-b"]),
            },
        ]
    )
    runner = _runner(
        registry=registry,
        intent=_comparison_intent("DPO variants", probe_existing_kb_first=True),
        plan=_discover_then_answer_plan("DPO variants"),
    )

    state = runner.run(
        user_request="Compare DPO and the newer DPO variants across my papers and recent work.",
        max_steps=6,
    )

    assert state.status == "success"
    assert [call[0] for call in registry.calls] == [
        "retrieve_evidence",
        "discover_papers",
        "save_papers_to_kb",
        "ensure_papers_retrievable",
        "retrieve_evidence",
    ]
    assert state.retrieved_evidence_ids == ["c-a", "c-a2", "c-b"]


def test_latest_request_with_active_paper_should_not_stay_scoped_to_old_context():
    registry = RoutingRegistry(
        [
            {
                "tool_name": "discover_papers",
                "result": _discovery_response(["p-latest"]),
            },
            {
                "tool_name": "ensure_papers_retrievable",
                "result": _ensure_response(["p-latest"]),
            },
            {
                "tool_name": "retrieve_evidence",
                "result": _retrieval_response(
                    ["c-old", "c-latest"],
                    ["p-old", "p-latest"],
                ),
            },
        ]
    )
    runner = _runner(
        registry=registry,
        intent=_factual_intent("latest DPO variants", probe_existing_kb_first=False),
        plan=_discover_then_answer_plan("latest DPO variants"),
    )

    state = runner.run(
        user_request="Compare the latest DPO variants with this saved paper.",
        active_paper_ids=["p-old"],
        max_steps=5,
    )

    assert state.status == "success"
    assert [call[0] for call in registry.calls] == [
        "discover_papers",
        "save_papers_to_kb",
        "ensure_papers_retrievable",
        "retrieve_evidence",
    ]
    assert registry.calls[3][1]["paper_ids"] == ["p-old", "p-latest"]
    assert state.retrieved_evidence_ids == ["c-old", "c-latest"]


def test_discover_then_answer_fails_if_new_retrieval_still_has_no_coverage():
    registry = RoutingRegistry(
        [
            {
                "tool_name": "retrieve_evidence",
                "result": _retrieval_response([], []),
            },
            {
                "tool_name": "discover_papers",
                "result": _discovery_response(["p-new"]),
            },
            {
                "tool_name": "ensure_papers_retrievable",
                "result": _ensure_response(["p-new"]),
            },
            {
                "tool_name": "retrieve_evidence",
                "result": _retrieval_response([], []),
            },
        ]
    )
    runner = _runner(
        registry=registry,
        intent=_factual_intent("unanswerable new method", probe_existing_kb_first=True),
        plan=_discover_then_answer_plan("unanswerable new method"),
    )

    state = runner.run(
        user_request="What did the recent unanswerable new method discover?",
        max_steps=8,
    )

    assert state.status == "failed"
    assert state.final_answer is None
    assert state.last_error == (
        "Knowledge coverage remained insufficient after discovery and retrieval; "
        "refusing another discovery handoff."
    )
    assert [call[0] for call in registry.calls] == [
        "retrieve_evidence",
        "discover_papers",
        "save_papers_to_kb",
        "ensure_papers_retrievable",
        "retrieve_evidence",
    ]


def _runner(
    *,
    registry: RoutingRegistry,
    intent: RequestIntent,
    plan: ExecutionPlan | None,
) -> LangGraphAgentRunner:
    return LangGraphAgentRunner(
        planner=EmptyPlanner() if plan is not None else ScriptedPlanner([]),
        executor=ToolExecutor(registry=registry),
        answer_service=FakeAnswerService(),
        intent_classifier=StaticIntentClassifier(intent),
        plan_generator=StaticPlanGenerator(plan) if plan is not None else None,
    )


def _factual_intent(topic: str, *, probe_existing_kb_first: bool) -> RequestIntent:
    return RequestIntent(
        task_type="factual_answer",
        topic=topic,
        needs_retrieval=True,
        needs_ingestion=True,
        probe_existing_kb_first=probe_existing_kb_first,
        finish_condition="retrieved_evidence",
        confidence=0.95,
        rationale="The user asks for a factual answer grounded in paper content.",
    )


def _comparison_intent(topic: str, *, probe_existing_kb_first: bool) -> RequestIntent:
    return _factual_intent(
        topic,
        probe_existing_kb_first=probe_existing_kb_first,
    ).model_copy(update={"task_type": "comparison"})


def _discover_then_answer_plan(topic: str) -> ExecutionPlan:
    return ExecutionPlan(
        goal=f"Answer from discovered or existing evidence about {topic}.",
        strategy="Probe existing KB, discover missing papers, prepare them, retrieve, then answer.",
        steps=[
            PlanStep(
                step_id="discover",
                kind="tool",
                tool_name="discover_papers",
                arguments={"user_query": topic, "max_results": 5, "max_selected": 2},
            ),
            PlanStep(
                step_id="prepare",
                kind="tool",
                tool_name="ensure_papers_retrievable",
                argument_sources={"paper_ids": "candidate_paper_ids"},
            ),
            PlanStep(
                step_id="retrieve",
                kind="tool",
                tool_name="retrieve_evidence",
                arguments={"query": topic, "top_k": 5},
                argument_sources={"paper_ids": "retrievable_paper_ids"},
            ),
            PlanStep(
                step_id="finish",
                kind="finish",
                answer_task=f"Answer from evidence about {topic}.",
            ),
        ],
    )


def _retrieval_response(chunk_ids: list[str], paper_ids: list[str]) -> dict:
    evidence = [
        {
            "chunk_id": chunk_id,
            "paper_id": paper_ids[min(index, len(paper_ids) - 1)] if paper_ids else "",
            "text": f"Evidence {chunk_id}",
            "final_score": 0.9,
            "semantic_score": 0.8,
            "metadata_score": 0.1,
        }
        for index, chunk_id in enumerate(chunk_ids)
    ]
    return {
        "status": "success",
        "query": "q",
        "retrieved": len(evidence),
        "evidence": evidence,
        "summary": f"Retrieved {len(evidence)} evidence chunks.",
    }


def _discovery_response(paper_ids: list[str]) -> dict:
    return {
        "status": "success",
        "candidate_paper_ids": paper_ids,
        "selected_paper_ids": paper_ids,
        "candidate_count": len(paper_ids),
        "selected_count": len(paper_ids),
        "summary": f"Discovered {len(paper_ids)} papers.",
    }


def _ensure_response(paper_ids: list[str]) -> dict:
    return {
        "status": "success",
        "ready_paper_ids": paper_ids,
        "already_ready_paper_ids": [],
        "summary": f"Prepared {len(paper_ids)} papers for semantic retrieval; failed 0.",
    }
