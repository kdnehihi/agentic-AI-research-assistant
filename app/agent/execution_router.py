from __future__ import annotations

import re

from app.agent.execution_plan import ExecutionPlan, PlanStep
from app.agent.execution_strategy import ExecutionStrategy
from app.agent.planner_state import PlannerState
from app.agent.request_intent import RequestIntent
from app.retrieval.query_intent import infer_explicit_section_groups_from_query


FAST_BRANCH_CONFIDENCE_THRESHOLD = 0.80
DEFAULT_DISCOVERY_SELECTED_COUNT = 5
DEFAULT_DISCOVER_THEN_ANSWER_COUNT = 3
MAX_DISCOVERY_SELECTED_COUNT = 20
REQUESTED_COUNT_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "một": 1,
    "mot": 1,
    "hai": 2,
    "ba": 3,
    "bốn": 4,
    "bon": 4,
    "năm": 5,
    "nam": 5,
    "sáu": 6,
    "sau": 6,
    "bảy": 7,
    "bay": 7,
    "tám": 8,
    "tam": 8,
    "chín": 9,
    "chin": 9,
    "mười": 10,
    "muoi": 10,
}
PAPER_NOUN_PATTERN = (
    r"(?:papers?|paper|bài báo|bai bao|bài|bai|nghiên cứu|nghien cuu)"
)
COUNT_MODIFIER_PATTERN = r"(?:\s+[\wÀ-ỹ-]+){0,4}"


def build_strategy_execution_plan(state: PlannerState) -> ExecutionPlan | None:
    """Build a deterministic plan from the supervisor execution strategy."""

    strategy = state.execution_strategy
    if strategy is None:
        return None
    intent = state.request_intent
    if strategy == ExecutionStrategy.DISCOVERY_ONLY:
        state.execution_branch = "strategy_discovery_only"
        return _discovery_plan(state, intent)
    if strategy == ExecutionStrategy.KNOWLEDGE_ONLY:
        state.execution_branch = "strategy_knowledge_only"
        return _knowledge_plan(state)
    if strategy == ExecutionStrategy.DISCOVER_THEN_ANSWER:
        state.execution_branch = "strategy_discover_then_answer"
        return _discover_then_answer_plan(state)
    return None


def build_fast_execution_plan(state: PlannerState) -> ExecutionPlan | None:
    """Build a deterministic execution plan for confident, low-ambiguity requests."""

    intent = state.request_intent
    if intent is None or intent.confidence < FAST_BRANCH_CONFIDENCE_THRESHOLD:
        return None

    if intent.task_type == "discovery_only":
        state.execution_branch = "fast_discovery"
        return _discovery_plan(state, intent)

    if intent.task_type == "metadata_lookup":
        state.execution_branch = "fast_metadata"
        return _metadata_plan(state)

    if intent.task_type == "summarization":
        state.execution_branch = "fast_summarization"
        return _summarization_plan(state, intent)

    if intent.needs_retrieval:
        paper_source = _paper_id_source(state)
        if paper_source is None:
            return None
        state.execution_branch = "fast_scoped_retrieval"
        return _scoped_retrieval_plan(state, paper_source)

    return None


def _discovery_plan(
    state: PlannerState,
    intent: RequestIntent | None,
) -> ExecutionPlan:
    topic = _topic_or_request(state, intent)
    selected_count = _requested_paper_count(
        state.user_request,
        default=DEFAULT_DISCOVERY_SELECTED_COUNT,
    )
    return ExecutionPlan(
        goal=state.user_request,
        strategy="Use the deterministic discovery branch for a metadata-only search.",
        steps=[
            PlanStep(
                step_id="discover",
                kind="tool",
                tool_name="discover_papers",
                arguments={
                    "user_query": topic,
                    "max_results": _candidate_pool_size(selected_count),
                    "max_selected": selected_count,
                },
                success_condition="selected_paper_ids or candidate_paper_ids is not empty",
                rationale="The classified intent only needs paper metadata.",
            ),
            PlanStep(
                step_id="finish",
                kind="finish",
                answer_task=state.user_request,
                success_condition="paper metadata is available",
                rationale="Discovery produced the requested paper list.",
            ),
        ],
    )


def _knowledge_plan(state: PlannerState) -> ExecutionPlan:
    arguments: dict[str, object] = {"query": state.user_request, "top_k": 5}
    section_groups = infer_explicit_section_groups_from_query(state.user_request)
    if section_groups:
        arguments["section_groups"] = list(section_groups)

    paper_source = _paper_id_source(state)
    step_kwargs = {}
    if paper_source is not None:
        step_kwargs["argument_sources"] = {"paper_ids": paper_source}

    return ExecutionPlan(
        goal=state.user_request,
        strategy=(
            "Use existing indexed knowledge first, then finish only if coverage "
            "is sufficient."
        ),
        steps=[
            PlanStep(
                step_id="retrieve_existing",
                kind="tool",
                tool_name="retrieve_evidence",
                arguments=arguments,
                success_condition="retrieved evidence covers the request",
                rationale="The supervisor requires an explicit KB evidence probe.",
                **step_kwargs,
            ),
            PlanStep(
                step_id="finish",
                kind="finish",
                answer_task=state.user_request,
                success_condition="knowledge coverage is sufficient",
                rationale="Existing KB evidence is enough for grounded generation.",
            ),
        ],
    )


def _discover_then_answer_plan(state: PlannerState) -> ExecutionPlan:
    topic = _topic_or_request(state, state.request_intent)
    selected_count = _requested_paper_count(
        state.user_request,
        default=DEFAULT_DISCOVER_THEN_ANSWER_COUNT,
    )
    return ExecutionPlan(
        goal=state.user_request,
        strategy=(
            "Discover missing papers, persist metadata, prepare papers for RAG, "
            "retrieve evidence again, then answer."
        ),
        steps=[
            PlanStep(
                step_id="discover",
                kind="tool",
                tool_name="discover_papers",
                arguments={
                    "user_query": topic,
                    "max_results": _candidate_pool_size(selected_count),
                    "max_selected": selected_count,
                },
                success_condition="selected_paper_ids or candidate_paper_ids is not empty",
                rationale="KB coverage was missing or stale, so external discovery is needed.",
            ),
            PlanStep(
                step_id="save_metadata",
                kind="tool",
                tool_name="save_papers_to_kb",
                arguments={"knowledge_base_id": "default"},
                argument_sources={"paper_ids": "candidate_paper_ids"},
                success_condition="discovered paper metadata is persisted",
                rationale="Persist discovered papers before preparing them for retrieval.",
            ),
            PlanStep(
                step_id="prepare",
                kind="tool",
                tool_name="ensure_papers_retrievable",
                argument_sources={"paper_ids": "candidate_paper_ids"},
                success_condition="papers are retrievable",
                rationale="Discovered papers must be indexed before grounded retrieval.",
            ),
            PlanStep(
                step_id="retrieve_new",
                kind="tool",
                tool_name="retrieve_evidence",
                arguments={"query": state.user_request, "top_k": 5},
                argument_sources={"paper_ids": "retrievable_paper_ids"},
                success_condition="newly indexed evidence is retrieved",
                rationale="Return control to the knowledge workflow for grounded evidence.",
            ),
            PlanStep(
                step_id="finish",
                kind="finish",
                answer_task=state.user_request,
                success_condition="grounded answer is supported by retrieved evidence",
                rationale="Retrieved evidence is now available for generation.",
            ),
        ],
    )


def _metadata_plan(state: PlannerState) -> ExecutionPlan:
    paper_source = _paper_id_source(state)
    if paper_source is not None:
        steps = [
            PlanStep(
                step_id="get_metadata",
                kind="tool",
                tool_name="get_paper_metadata",
                argument_sources={"paper_ids": paper_source},
                success_condition="stored paper metadata is available",
                rationale="The request can be answered from scoped paper metadata.",
            )
        ]
        strategy = "Use scoped metadata from the current paper context."
    else:
        steps = [
            PlanStep(
                step_id="list_papers",
                kind="tool",
                tool_name="list_papers",
                arguments={"limit": 10},
                success_condition="stored paper metadata is available",
                rationale="The request can be answered from the stored paper list.",
            )
        ]
        strategy = "List stored paper metadata without invoking the planner."

    steps.append(
        PlanStep(
            step_id="finish",
            kind="finish",
            answer_task=state.user_request,
            success_condition="metadata artifacts are available",
            rationale="Metadata lookup produced the requested artifacts.",
        )
    )
    return ExecutionPlan(goal=state.user_request, strategy=strategy, steps=steps)


def _summarization_plan(
    state: PlannerState,
    intent: RequestIntent | None,
) -> ExecutionPlan:
    paper_source = _paper_id_source(state)
    summary_mode = _summary_mode_from_request(state.user_request)
    if paper_source is not None:
        return ExecutionPlan(
            goal=state.user_request,
            strategy=(
                "Summarize scoped papers from stored metadata/abstracts without "
                "running full paper ingestion."
            ),
            steps=[
                PlanStep(
                    step_id="summarize",
                    kind="tool",
                    tool_name="summarize_papers",
                    arguments={"summary_mode": summary_mode},
                    argument_sources={"paper_ids": paper_source},
                    success_condition="paper summaries are available",
                    rationale="The request asks for summaries of known papers.",
                ),
                PlanStep(
                    step_id="finish",
                    kind="finish",
                    answer_task=state.user_request,
                    success_condition="paper summaries are available",
                    rationale="Summaries are sufficient for this request.",
                ),
            ],
        )

    topic = _topic_or_request(state, intent)
    selected_count = _requested_paper_count(
        state.user_request,
        default=DEFAULT_DISCOVER_THEN_ANSWER_COUNT,
    )
    return ExecutionPlan(
        goal=state.user_request,
        strategy=(
            "Discover papers, persist metadata, and summarize abstracts without "
            "blocking the chat turn on PDF ingestion or embedding."
        ),
        steps=[
            PlanStep(
                step_id="discover",
                kind="tool",
                tool_name="discover_papers",
                arguments={
                    "user_query": topic,
                    "max_results": _candidate_pool_size(selected_count),
                    "max_selected": selected_count,
                },
                success_condition="selected_paper_ids or candidate_paper_ids is not empty",
                rationale="The request asks for paper discovery plus summary.",
            ),
            PlanStep(
                step_id="save_metadata",
                kind="tool",
                tool_name="save_papers_to_kb",
                arguments={"knowledge_base_id": "default"},
                argument_sources={"paper_ids": "candidate_paper_ids"},
                success_condition="discovered paper metadata is persisted",
                rationale="Persist discovered papers before summarizing them.",
            ),
            PlanStep(
                step_id="summarize",
                kind="tool",
                tool_name="summarize_papers",
                arguments={"summary_mode": summary_mode},
                argument_sources={"paper_ids": "candidate_paper_ids"},
                success_condition="paper summaries are available",
                rationale="Abstract summaries are enough for a discovery-summary request.",
            ),
            PlanStep(
                step_id="finish",
                kind="finish",
                answer_task=state.user_request,
                success_condition="paper summaries are available",
                rationale="Summaries are ready for generation.",
            ),
        ],
    )


def _scoped_retrieval_plan(
    state: PlannerState,
    paper_source: str,
) -> ExecutionPlan:
    arguments: dict[str, object] = {"query": state.user_request, "top_k": 5}
    section_groups = infer_explicit_section_groups_from_query(state.user_request)
    if section_groups:
        arguments["section_groups"] = list(section_groups)

    return ExecutionPlan(
        goal=state.user_request,
        strategy=(
            "Use scoped retrieval from known runtime paper context before any "
            "open-ended planning."
        ),
        steps=[
            PlanStep(
                step_id="retrieve",
                kind="tool",
                tool_name="retrieve_evidence",
                arguments=arguments,
                argument_sources={"paper_ids": paper_source},
                success_condition="retrieved_evidence is not empty",
                rationale="The runtime state already identifies the relevant papers.",
            ),
            PlanStep(
                step_id="finish",
                kind="finish",
                answer_task=state.user_request,
                success_condition="grounded answer returned from retrieved evidence",
                rationale="Retrieved evidence is sufficient for grounded generation.",
            ),
        ],
    )


def _paper_id_source(state: PlannerState) -> str | None:
    if state.active_paper_ids:
        return "active_paper_ids"
    if state.retrievable_paper_ids:
        return "retrievable_paper_ids"
    if state.known_paper_ids:
        return "known_paper_ids"
    if state.saved_paper_ids:
        return "saved_paper_ids"
    return None


def _topic_or_request(
    state: PlannerState,
    intent: RequestIntent | None,
) -> str:
    if intent is not None and intent.topic:
        return intent.topic
    return state.user_request


def _summary_mode_from_request(user_request: str) -> str:
    request = user_request.lower()
    if any(term in request for term in ("limitation", "limitations", "open problem")):
        return "limitations"
    if any(term in request for term in ("method", "methodology", "approach")):
        return "method"
    if any(term in request for term in ("contribution", "contributions")):
        return "contributions"
    return "abstract"


def _requested_paper_count(user_request: str, *, default: int) -> int:
    count = _requested_numeric_count(user_request)
    if count is None:
        count = _requested_word_count(user_request)
    if count is None:
        return default
    return max(1, min(count, MAX_DISCOVERY_SELECTED_COUNT))


def _requested_numeric_count(user_request: str) -> int | None:
    patterns = [
        rf"\b(\d{{1,2}}){COUNT_MODIFIER_PATTERN}\s+{PAPER_NOUN_PATTERN}\b",
        rf"\b{PAPER_NOUN_PATTERN}\s+(\d{{1,2}})\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, user_request, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def _requested_word_count(user_request: str) -> int | None:
    alternatives = "|".join(re.escape(word) for word in REQUESTED_COUNT_WORDS)
    patterns = [
        rf"\b({alternatives}){COUNT_MODIFIER_PATTERN}\s+{PAPER_NOUN_PATTERN}\b",
        rf"\b{PAPER_NOUN_PATTERN}\s+({alternatives})\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, user_request, flags=re.IGNORECASE)
        if match:
            return REQUESTED_COUNT_WORDS[match.group(1).lower()]
    return None


def _candidate_pool_size(selected_count: int) -> int:
    return max(selected_count * 10, 20)
