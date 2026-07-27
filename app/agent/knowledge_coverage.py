from __future__ import annotations

from typing import Any

from app.agent.execution_strategy import (
    ExecutionStrategy,
    KnowledgeCoverageDecision,
)
from app.agent.planner_models import ToolObservation
from app.agent.planner_state import PlannerState


FRESHNESS_CUES = (
    "latest",
    "recent",
    "newest",
    "newer",
    "up-to-date",
    "current",
    "state of the art",
    "sota",
)
COMPARISON_TASK_TYPES = {"comparison", "report"}
MIN_STRONG_SCORE = 0.45
MIN_SUFFICIENT_CHUNKS = 1
MIN_COMPARISON_PAPERS = 2


class KnowledgeCoverageEvaluator:
    """Evaluate whether retrieved evidence is enough for the current request."""

    def evaluate(
        self,
        *,
        state: PlannerState,
        observation: ToolObservation | None = None,
    ) -> KnowledgeCoverageDecision:
        """Return a deterministic coverage decision for supervisor routing."""

        if _paper_not_retrievable(observation):
            missing_ids = _missing_paper_ids(observation)
            return KnowledgeCoverageDecision(
                coverage="insufficient",
                retrieved_evidence_ids=[],
                relevant_paper_ids=[],
                missing_aspects=["requested papers are stored but not indexed"],
                confidence=0.9,
                rationale="Metadata exists, but at least one requested paper is not retrievable.",
                recommended_strategy=ExecutionStrategy.KNOWLEDGE_ONLY,
            ).model_copy(
                update={
                    "relevant_paper_ids": missing_ids,
                }
            )

        evidence = _evidence_records(state=state, observation=observation)
        if not evidence:
            return KnowledgeCoverageDecision(
                coverage="insufficient",
                retrieved_evidence_ids=[],
                relevant_paper_ids=[],
                missing_aspects=["no indexed evidence was retrieved"],
                confidence=0.95,
                rationale="The knowledge-base probe returned no usable evidence.",
                recommended_strategy=_fallback_strategy(state),
            )

        evidence_ids = _evidence_ids(evidence)
        paper_ids = _paper_ids(evidence)
        strong_chunks = [
            item
            for item in evidence
            if _score(item) >= MIN_STRONG_SCORE
        ]
        missing_aspects: list[str] = []

        if len(strong_chunks) < MIN_SUFFICIENT_CHUNKS:
            missing_aspects.append("retrieved chunks have weak relevance scores")

        if _is_comparison_request(state) and len(paper_ids) < MIN_COMPARISON_PAPERS:
            missing_aspects.append("comparison needs evidence from multiple papers")

        if _requires_freshness(state) and not _has_fresh_discovered_support(state, paper_ids):
            missing_aspects.append("request asks for latest or recent external coverage")

        if missing_aspects:
            return KnowledgeCoverageDecision(
                coverage="partial",
                retrieved_evidence_ids=evidence_ids,
                relevant_paper_ids=paper_ids,
                missing_aspects=missing_aspects,
                confidence=0.75,
                rationale="Some evidence exists, but it does not cover the request fully.",
                recommended_strategy=_fallback_strategy(state),
            )

        return KnowledgeCoverageDecision(
            coverage="sufficient",
            retrieved_evidence_ids=evidence_ids,
            relevant_paper_ids=paper_ids,
            missing_aspects=[],
            confidence=0.85,
            rationale="Retrieved evidence is relevant enough to support a grounded answer.",
            recommended_strategy=ExecutionStrategy.KNOWLEDGE_ONLY,
        )


def request_requires_freshness(user_request: str) -> bool:
    """Return whether wording requests recent or latest external coverage."""

    lowered = user_request.lower()
    return any(cue in lowered for cue in FRESHNESS_CUES)


def _paper_not_retrievable(observation: ToolObservation | None) -> bool:
    return (
        observation is not None
        and observation.status == "prerequisite_missing"
        and observation.error_type == "paper_not_retrievable"
    )


def _missing_paper_ids(observation: ToolObservation | None) -> list[str]:
    if observation is None:
        return []
    values = observation.result.get("missing_paper_ids") or []
    return [value for value in values if isinstance(value, str)]


def _evidence_records(
    *,
    state: PlannerState,
    observation: ToolObservation | None,
) -> list[dict[str, Any]]:
    if observation is not None:
        evidence = observation.result.get("evidence")
        if isinstance(evidence, list):
            return [item for item in evidence if isinstance(item, dict)]
    return [item for item in state.retrieved_evidence if isinstance(item, dict)]


def _evidence_ids(evidence: list[dict[str, Any]]) -> list[str]:
    values: list[str] = []
    for item in evidence:
        chunk_id = item.get("chunk_id")
        if isinstance(chunk_id, str) and chunk_id not in values:
            values.append(chunk_id)
    return values


def _paper_ids(evidence: list[dict[str, Any]]) -> list[str]:
    values: list[str] = []
    for item in evidence:
        paper_id = item.get("paper_id")
        if isinstance(paper_id, str) and paper_id and paper_id not in values:
            values.append(paper_id)
    return values


def _score(item: dict[str, Any]) -> float:
    for key in ("final_score", "semantic_score", "metadata_score"):
        value = item.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return 1.0


def _is_comparison_request(state: PlannerState) -> bool:
    intent = state.request_intent
    if intent is not None and intent.task_type in COMPARISON_TASK_TYPES:
        return True
    request = state.user_request.lower()
    return "compare" in request or "versus" in request or " vs " in request


def _requires_freshness(state: PlannerState) -> bool:
    return request_requires_freshness(state.user_request)


def _has_fresh_discovered_support(
    state: PlannerState,
    paper_ids: list[str],
) -> bool:
    if not _requires_freshness(state):
        return True
    known_ids = set(state.known_paper_ids or [])
    if not known_ids:
        return False
    return any(paper_id in known_ids for paper_id in paper_ids)


def _fallback_strategy(state: PlannerState) -> ExecutionStrategy:
    intent = state.request_intent
    if intent is None or not intent.needs_retrieval:
        return ExecutionStrategy.KNOWLEDGE_ONLY
    if state.active_paper_ids and not request_requires_freshness(state.user_request):
        return ExecutionStrategy.KNOWLEDGE_ONLY
    return ExecutionStrategy.DISCOVER_THEN_ANSWER
