from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from app.agent.dynamic_runner import DynamicAgentRunner
from app.agent.executor import ToolExecutor
from app.agent.grounded_answer import GroundedAnswerService
from app.agent.planner_models import CallToolAction, FinishAction, PlannerDecision
from app.agent.planner_state import PlannerState
from app.agent.planner_view import build_planner_view
from app.agent.tool_catalog import build_tool_specs
from app.agent.tool_spec import ToolCategory, ToolSpec
from app.tools.registry import ToolFunction


@dataclass(frozen=True)
class PlannerEvalCase:
    """One deterministic planner contract scenario."""

    name: str
    user_request: str
    planner_decisions: list[PlannerDecision]
    tool_responses: dict[str, list[dict[str, Any]]]
    expected_tools: list[str]
    expected_status: str = "success"
    expected_first_planner_view: dict[str, Any] = field(default_factory=dict)
    max_steps: int = 10


@dataclass
class PlannerEvalResult:
    """Evaluation result for one planner contract scenario."""

    name: str
    passed: bool
    status: str
    expected_status: str
    tool_sequence: list[str]
    expected_tools: list[str]
    last_error: str | None
    failures: list[str]
    final_answer: Any

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "status": self.status,
            "expected_status": self.expected_status,
            "tool_sequence": self.tool_sequence,
            "expected_tools": self.expected_tools,
            "last_error": self.last_error,
            "failures": self.failures,
            "final_answer": self.final_answer,
        }


class ScriptedEvalPlanner:
    """Deterministic planner used to evaluate orchestration contracts."""

    def __init__(self, decisions: Iterable[PlannerDecision]) -> None:
        self.decisions = deque(decisions)
        self.views: list[dict[str, Any]] = []

    def decide(self, state: PlannerState, tool_specs: list[ToolSpec]) -> PlannerDecision:
        del tool_specs
        self.views.append(build_planner_view(state))
        if not self.decisions:
            return FinishAction(
                answer_task=state.user_request,
                decision_summary="No scripted decisions remain.",
            )
        return self.decisions.popleft()


class EvalRegistry:
    """Production-tool registry backed by queued deterministic responses."""

    def __init__(self, responses: dict[str, list[dict[str, Any]]]) -> None:
        self.specs = build_tool_specs()
        self.responses = {
            tool_name: deque(tool_responses)
            for tool_name, tool_responses in responses.items()
        }
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def list_tools(self, category: ToolCategory | None = None) -> list[str]:
        return [
            name
            for name, spec in self.specs.items()
            if category is None or spec.category == category
        ]

    def get_tool_spec(self, tool_name: str) -> ToolSpec:
        return self.specs[tool_name]

    def has_tool(self, tool_name: str) -> bool:
        return tool_name in self.specs

    def execute(self, tool_name: str, state, **kwargs) -> dict[str, Any]:
        del state
        self.calls.append((tool_name, kwargs))
        queue = self.responses.get(tool_name)
        if not queue:
            return {
                "status": "failed",
                "error_type": "missing_eval_response",
                "summary": f"No eval response configured for {tool_name}.",
            }
        return queue.popleft()


class EvalAnswerService(GroundedAnswerService):
    """Answer service that avoids LLM calls during planner contract evals."""

    def __init__(self) -> None:
        pass

    def generate(self, *, state: PlannerState, answer_task: str) -> dict[str, Any]:
        return {
            "answer_task": answer_task,
            "source": "planner_eval",
            "retrieved_evidence_ids": state.retrieved_evidence_ids,
            "known_paper_ids": state.known_paper_ids,
            "retrievable_paper_ids": state.retrievable_paper_ids,
        }


def evaluate_planner_cases(
    cases: Iterable[PlannerEvalCase] | None = None,
) -> list[PlannerEvalResult]:
    """Run deterministic planner contract cases and return pass/fail results."""

    return [evaluate_planner_case(case) for case in (cases or default_eval_cases())]


def evaluate_planner_case(case: PlannerEvalCase) -> PlannerEvalResult:
    registry = EvalRegistry(case.tool_responses)
    planner = ScriptedEvalPlanner(case.planner_decisions)
    runner = DynamicAgentRunner(
        planner=planner,
        executor=ToolExecutor(registry=registry),
        answer_service=EvalAnswerService(),
    )

    state = runner.run(user_request=case.user_request, max_steps=case.max_steps)
    tool_sequence = [tool_name for tool_name, _ in registry.calls]
    failures = _failures_for(
        case=case,
        state=state,
        tool_sequence=tool_sequence,
        planner_views=planner.views,
    )
    return PlannerEvalResult(
        name=case.name,
        passed=not failures,
        status=state.status,
        expected_status=case.expected_status,
        tool_sequence=tool_sequence,
        expected_tools=case.expected_tools,
        last_error=state.last_error,
        failures=failures,
        final_answer=state.final_answer,
    )


def default_eval_cases() -> list[PlannerEvalCase]:
    """Core regression suite for freezing the current dynamic planner behavior."""

    return [
        PlannerEvalCase(
            name="existing_kb_hit_finishes_after_policy_probe",
            user_request=(
                "Use papers already stored in the knowledge base to answer: "
                "what are xMemory limitations?"
            ),
            planner_decisions=[
                FinishAction(
                    answer_task="Answer xMemory limitations from retrieved evidence.",
                    decision_summary="The KB probe found enough evidence.",
                )
            ],
            tool_responses={
                "retrieve_evidence": [
                    _retrieval_response(query="what are xMemory limitations?", chunk_id="c-xmem")
                ]
            },
            expected_tools=["retrieve_evidence"],
            expected_first_planner_view={
                "kb_probe_attempted": True,
                "last_retrieval_count": 1,
            },
        ),
        PlannerEvalCase(
            name="missing_kb_discovers_prepares_and_retrieves",
            user_request=(
                "What did the new long-term memory paper discover? If it is not "
                "in the knowledge base, find the paper first."
            ),
            planner_decisions=[
                CallToolAction(
                    tool_name="discover_papers",
                    arguments={
                        "user_query": "new long-term memory paper",
                        "max_results": 5,
                        "max_selected": 1,
                    },
                    decision_summary="The KB probe returned no evidence.",
                ),
                CallToolAction(
                    tool_name="ensure_papers_retrievable",
                    arguments={"paper_ids": ["p-new"]},
                    decision_summary="Prepare the newly discovered paper.",
                ),
                CallToolAction(
                    tool_name="retrieve_evidence",
                    arguments={
                        "query": "What did the new long-term memory paper discover?",
                        "paper_ids": ["p-new"],
                        "top_k": 5,
                    },
                    decision_summary="Retrieve evidence from the prepared paper.",
                ),
                FinishAction(
                    answer_task="Answer from the newly retrieved evidence.",
                    decision_summary="Evidence is now available.",
                ),
            ],
            tool_responses={
                "retrieve_evidence": [
                    _retrieval_response(
                        query="What did the new long-term memory paper discover?",
                        retrieved=0,
                        evidence=[],
                    ),
                    _retrieval_response(
                        query="What did the new long-term memory paper discover?",
                        chunk_id="c-new",
                        paper_id="p-new",
                    ),
                ],
                "discover_papers": [_discovery_response(["p-new"])],
                "ensure_papers_retrievable": [_ensure_response(["p-new"])],
            },
            expected_tools=[
                "retrieve_evidence",
                "discover_papers",
                "ensure_papers_retrievable",
                "retrieve_evidence",
            ],
            expected_first_planner_view={
                "kb_probe_attempted": True,
                "last_retrieval_count": 0,
            },
        ),
        PlannerEvalCase(
            name="new_paper_request_discovers_before_retrieval",
            user_request=(
                "Find a recent paper about long-term agent memory and answer "
                "what it discovered."
            ),
            planner_decisions=[
                CallToolAction(
                    tool_name="discover_papers",
                    arguments={
                        "user_query": "long-term agent memory",
                        "max_results": 5,
                        "max_selected": 1,
                    },
                    decision_summary="A new paper must be found.",
                ),
                CallToolAction(
                    tool_name="get_paper_metadata",
                    arguments={"paper_ids": ["p-memory"]},
                    decision_summary="Check metadata before ingestion.",
                ),
                CallToolAction(
                    tool_name="save_papers_to_kb",
                    arguments={"paper_ids": ["p-memory"]},
                    decision_summary="Persist the selected paper.",
                ),
                CallToolAction(
                    tool_name="ensure_papers_retrievable",
                    arguments={"paper_ids": ["p-memory"]},
                    decision_summary="Prepare the paper for retrieval.",
                ),
                CallToolAction(
                    tool_name="retrieve_evidence",
                    arguments={
                        "query": "what did the long-term agent memory paper discover?",
                        "paper_ids": ["p-memory"],
                        "top_k": 5,
                    },
                    decision_summary="Retrieve evidence from the prepared paper.",
                ),
                FinishAction(
                    answer_task="Answer from retrieved evidence.",
                    decision_summary="Evidence is available.",
                ),
            ],
            tool_responses={
                "discover_papers": [_discovery_response(["p-memory"])],
                "get_paper_metadata": [_metadata_response(["p-memory"])],
                "save_papers_to_kb": [_save_response(["p-memory"])],
                "ensure_papers_retrievable": [_ensure_response(["p-memory"])],
                "retrieve_evidence": [
                    _retrieval_response(
                        query="what did the long-term agent memory paper discover?",
                        chunk_id="c-memory",
                        paper_id="p-memory",
                    )
                ],
            },
            expected_tools=[
                "discover_papers",
                "get_paper_metadata",
                "save_papers_to_kb",
                "ensure_papers_retrievable",
                "retrieve_evidence",
            ],
        ),
        PlannerEvalCase(
            name="compare_request_discovers_prepares_and_retrieves",
            user_request=(
                "Compare recent papers about agentic retrieval augmented "
                "generation, especially ARAG and multi-agent RAG filtering."
            ),
            planner_decisions=[
                CallToolAction(
                    tool_name="discover_papers",
                    arguments={
                        "user_query": "agentic retrieval augmented generation ARAG",
                        "max_results": 10,
                        "max_selected": 3,
                    },
                    decision_summary="Find comparison papers.",
                ),
                CallToolAction(
                    tool_name="ensure_papers_retrievable",
                    arguments={"paper_ids": ["p-arag", "p-mainrag"]},
                    decision_summary="Prepare selected papers.",
                ),
                CallToolAction(
                    tool_name="retrieve_evidence",
                    arguments={
                        "query": "compare ARAG and multi-agent RAG filtering",
                        "paper_ids": ["p-arag", "p-mainrag"],
                        "top_k": 10,
                    },
                    decision_summary="Retrieve comparison evidence.",
                ),
                FinishAction(
                    answer_task="Compare methods, results, and limitations.",
                    decision_summary="Evidence is available.",
                ),
            ],
            tool_responses={
                "discover_papers": [_discovery_response(["p-arag", "p-mainrag"])],
                "ensure_papers_retrievable": [_ensure_response(["p-arag", "p-mainrag"])],
                "retrieve_evidence": [
                    _retrieval_response(
                        query="compare ARAG and multi-agent RAG filtering",
                        chunk_id="c-arag",
                        paper_id="p-arag",
                    )
                ],
            },
            expected_tools=[
                "discover_papers",
                "ensure_papers_retrievable",
                "retrieve_evidence",
            ],
        ),
    ]


def summarize_eval_results(results: list[PlannerEvalResult]) -> dict[str, Any]:
    failures_by_case = defaultdict(list)
    for result in results:
        if result.failures:
            failures_by_case[result.name].extend(result.failures)
    passed = sum(1 for result in results if result.passed)
    return {
        "passed": passed,
        "failed": len(results) - passed,
        "total": len(results),
        "pass_rate": passed / len(results) if results else 0.0,
        "failures_by_case": dict(failures_by_case),
        "results": [result.to_dict() for result in results],
    }


def _failures_for(
    *,
    case: PlannerEvalCase,
    state: PlannerState,
    tool_sequence: list[str],
    planner_views: list[dict[str, Any]],
) -> list[str]:
    failures = []
    if state.status != case.expected_status:
        failures.append(f"status={state.status}, expected={case.expected_status}")
    if tool_sequence != case.expected_tools:
        failures.append(f"tools={tool_sequence}, expected={case.expected_tools}")
    if case.expected_status == "success" and not state.final_answer:
        failures.append("missing final_answer")
    if case.expected_first_planner_view:
        if not planner_views:
            failures.append("planner was never asked after policy/tool execution")
        else:
            first_view = planner_views[0]
            for key, expected_value in case.expected_first_planner_view.items():
                actual_value = first_view.get(key)
                if actual_value != expected_value:
                    failures.append(
                        f"first_planner_view[{key}]={actual_value!r}, "
                        f"expected={expected_value!r}"
                    )
    return failures


def _discovery_response(paper_ids: list[str]) -> dict[str, Any]:
    return {
        "status": "success",
        "candidate_paper_ids": paper_ids,
        "selected_paper_ids": paper_ids,
        "candidate_count": len(paper_ids),
        "selected_count": len(paper_ids),
        "summary": (
            f"Discovered {len(paper_ids)} candidate papers and selected "
            f"{len(paper_ids)} papers."
        ),
    }


def _metadata_response(paper_ids: list[str]) -> dict[str, Any]:
    return {
        "status": "success",
        "papers": [
            {
                "paper_id": paper_id,
                "title": f"Paper {paper_id}",
                "exists_in_kb": False,
                "indexed": False,
            }
            for paper_id in paper_ids
        ],
        "missing_paper_ids": [],
        "summary": f"Resolved metadata for {len(paper_ids)} papers; missing 0.",
    }


def _save_response(paper_ids: list[str]) -> dict[str, Any]:
    return {
        "status": "success",
        "inserted_paper_ids": paper_ids,
        "updated_paper_ids": [],
        "already_present_paper_ids": [],
        "failed": [],
        "summary": f"Saved {len(paper_ids)} new papers.",
    }


def _ensure_response(paper_ids: list[str]) -> dict[str, Any]:
    return {
        "status": "success",
        "ready_paper_ids": paper_ids,
        "already_ready_paper_ids": [],
        "failed": [],
        "summary": f"Prepared {len(paper_ids)} papers for semantic retrieval; failed 0.",
    }


def _retrieval_response(
    *,
    query: str,
    retrieved: int = 1,
    evidence: list[dict[str, Any]] | None = None,
    chunk_id: str = "c1",
    paper_id: str = "p1",
) -> dict[str, Any]:
    if evidence is None:
        evidence = [
            {
                "chunk_id": chunk_id,
                "paper_id": paper_id,
                "text": "Evidence text.",
                "rank": 1,
            }
        ]
    return {
        "status": "success",
        "query": query,
        "retrieved": retrieved,
        "evidence": evidence,
        "summary": f"Retrieved {retrieved} evidence chunks.",
    }


# Keep the imported callable visible to static analyzers when checking registry shape.
_TOOL_FUNCTION_TYPE: type[ToolFunction] | None = None
