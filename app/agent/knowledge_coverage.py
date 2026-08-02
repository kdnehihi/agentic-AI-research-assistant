from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.agent.execution_strategy import (
    ExecutionStrategy,
    KnowledgeCoverage,
    KnowledgeCoverageDecision,
)
from app.agent.planner_models import ToolObservation
from app.agent.planner_state import PlannerState
from app.llm.client import LLMClient


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
LLM_JUDGE_MIN_CONFIDENCE = 0.6
LLM_JUDGE_MAX_EVIDENCE_ITEMS = 5
LLM_JUDGE_MAX_TEXT_CHARS = 900
HARD_MISSING_ASPECTS = {
    "comparison needs evidence from multiple papers",
    "request asks for latest or recent external coverage",
}


class LLMCoverageJudgment(BaseModel):
    """LLM semantic judgment of whether retrieved evidence answers the request."""

    model_config = ConfigDict(extra="forbid")

    coverage: KnowledgeCoverage
    missing_aspects: list[str] = Field(default_factory=list)
    relevant_paper_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    rationale: str = Field(default="", max_length=500)


class KnowledgeCoverageEvaluator:
    """Evaluate whether retrieved evidence is enough for the current request."""

    def __init__(
        self,
        *,
        llm_client: LLMClient | None = None,
        use_llm_judge: bool = True,
        max_evidence_items: int = LLM_JUDGE_MAX_EVIDENCE_ITEMS,
        max_text_chars: int = LLM_JUDGE_MAX_TEXT_CHARS,
    ) -> None:
        self.llm_client = llm_client
        self.use_llm_judge = use_llm_judge
        self.max_evidence_items = max_evidence_items
        self.max_text_chars = max_text_chars

    def evaluate(
        self,
        *,
        state: PlannerState,
        observation: ToolObservation | None = None,
    ) -> KnowledgeCoverageDecision:
        """Return a deterministic coverage decision for supervisor routing."""

        decision = self._deterministic_evaluate(
            state=state,
            observation=observation,
        )
        evidence = _evidence_records(state=state, observation=observation)
        if not self.use_llm_judge or self.llm_client is None or not evidence:
            return decision

        judgment = self._judge_with_llm(
            state=state,
            evidence=evidence,
            deterministic_decision=decision,
        )
        if judgment is None:
            return decision
        return _combine_decisions(
            state=state,
            deterministic_decision=decision,
            llm_judgment=judgment,
        )

    def _deterministic_evaluate(
        self,
        *,
        state: PlannerState,
        observation: ToolObservation | None = None,
    ) -> KnowledgeCoverageDecision:
        """Return the stable rule-based coverage decision."""

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

    def _judge_with_llm(
        self,
        *,
        state: PlannerState,
        evidence: list[dict[str, Any]],
        deterministic_decision: KnowledgeCoverageDecision,
    ) -> LLMCoverageJudgment | None:
        try:
            response = self.llm_client.generate(
                _build_llm_coverage_prompt(
                    state=state,
                    evidence=evidence,
                    deterministic_decision=deterministic_decision,
                    max_evidence_items=self.max_evidence_items,
                    max_text_chars=self.max_text_chars,
                )
            )
            return parse_llm_coverage_judgment(response)
        except Exception:
            return None


def request_requires_freshness(user_request: str) -> bool:
    """Return whether wording requests recent or latest external coverage."""

    lowered = user_request.lower()
    return any(cue in lowered for cue in FRESHNESS_CUES)


def parse_llm_coverage_judgment(response_text: str) -> LLMCoverageJudgment:
    """Parse an LLM response into a coverage judgment."""

    try:
        payload = json.loads(_extract_json(response_text))
        return LLMCoverageJudgment.model_validate(payload)
    except (json.JSONDecodeError, ValidationError, ValueError, TypeError) as exc:
        raise ValueError(f"Invalid LLM coverage judgment: {exc}") from exc


def _combine_decisions(
    *,
    state: PlannerState,
    deterministic_decision: KnowledgeCoverageDecision,
    llm_judgment: LLMCoverageJudgment,
) -> KnowledgeCoverageDecision:
    if llm_judgment.confidence < LLM_JUDGE_MIN_CONFIDENCE:
        return deterministic_decision

    if llm_judgment.coverage == "sufficient":
        return deterministic_decision

    merged_missing = _merge_unique(
        deterministic_decision.missing_aspects,
        llm_judgment.missing_aspects,
    )
    if not merged_missing:
        merged_missing = ["LLM judge found the retrieved evidence incomplete"]

    relevant_paper_ids = (
        llm_judgment.relevant_paper_ids
        or deterministic_decision.relevant_paper_ids
    )
    confidence = min(
        0.95,
        max(deterministic_decision.confidence, llm_judgment.confidence),
    )
    return KnowledgeCoverageDecision(
        coverage=llm_judgment.coverage,
        retrieved_evidence_ids=deterministic_decision.retrieved_evidence_ids,
        relevant_paper_ids=relevant_paper_ids,
        missing_aspects=merged_missing,
        confidence=confidence,
        rationale=(
            "LLM semantic coverage judge found incomplete support: "
            f"{llm_judgment.rationale}"
        ).strip(),
        recommended_strategy=_fallback_strategy(state),
    )


def _build_llm_coverage_prompt(
    *,
    state: PlannerState,
    evidence: list[dict[str, Any]],
    deterministic_decision: KnowledgeCoverageDecision,
    max_evidence_items: int,
    max_text_chars: int,
) -> str:
    request_intent = (
        state.request_intent.model_dump(mode="json")
        if state.request_intent is not None
        else None
    )
    payload = {
        "user_request": state.user_request,
        "request_intent": request_intent,
        "deterministic_decision": deterministic_decision.model_dump(mode="json"),
        "evidence": [
            _compact_evidence_item(item, max_text_chars=max_text_chars)
            for item in evidence[:max_evidence_items]
        ],
        "schema": LLMCoverageJudgment.model_json_schema(),
        "instructions": [
            "Judge only whether the retrieved evidence directly answers the user request.",
            "Use coverage='sufficient' only when the evidence contains enough direct support for a grounded answer.",
            "Use coverage='partial' when evidence is relevant but misses requested aspects.",
            "Use coverage='insufficient' when evidence is off-topic or cannot answer the request.",
            "Do not require external discovery merely because more evidence would be nice.",
            "Return only JSON matching the schema.",
        ],
    }
    return (
        "You are a strict evidence coverage judge for a research RAG system.\n"
        "Return only JSON.\n\n"
        + json.dumps(payload, indent=2, sort_keys=True)
    )


def _compact_evidence_item(
    item: dict[str, Any],
    *,
    max_text_chars: int,
) -> dict[str, Any]:
    text = str(item.get("text") or item.get("document") or "")
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    compact = {
        "chunk_id": item.get("chunk_id"),
        "paper_id": item.get("paper_id"),
        "title": item.get("title") or metadata.get("title"),
        "section": item.get("section") or metadata.get("section"),
        "final_score": item.get("final_score"),
        "semantic_score": item.get("semantic_score"),
        "text": text[:max_text_chars],
    }
    return {key: value for key, value in compact.items() if value not in (None, "")}


def _extract_json(text: str) -> str:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    if fenced:
        return fenced.group(1)
    return stripped


def _merge_unique(*groups: list[str]) -> list[str]:
    merged: list[str] = []
    for group in groups:
        for value in group:
            if isinstance(value, str) and value and value not in merged:
                merged.append(value)
    return merged


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
