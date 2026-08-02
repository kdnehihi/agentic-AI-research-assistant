from app.agent.execution_strategy import ExecutionStrategy
from app.agent.knowledge_coverage import (
    KnowledgeCoverageEvaluator,
    parse_llm_coverage_judgment,
)
from app.agent.planner_models import ToolObservation
from app.agent.planner_state import PlannerState
from app.agent.request_intent import RequestIntent
from app.agent.state import AgentState


def test_coverage_sufficient_for_strong_factual_evidence():
    state = _state("Explain DPO.", _intent("factual_answer"))
    observation = _retrieval_observation(
        [{"chunk_id": "c1", "paper_id": "p1", "final_score": 0.9}]
    )

    decision = KnowledgeCoverageEvaluator().evaluate(
        state=state,
        observation=observation,
    )

    assert decision.coverage == "sufficient"
    assert decision.recommended_strategy == ExecutionStrategy.KNOWLEDGE_ONLY
    assert decision.retrieved_evidence_ids == ["c1"]


def test_coverage_insufficient_when_no_evidence_is_retrieved():
    state = _state("Explain DPO.", _intent("factual_answer"))
    observation = _retrieval_observation([])

    decision = KnowledgeCoverageEvaluator().evaluate(
        state=state,
        observation=observation,
    )

    assert decision.coverage == "insufficient"
    assert decision.recommended_strategy == ExecutionStrategy.DISCOVER_THEN_ANSWER
    assert "no indexed evidence" in decision.missing_aspects[0]


def test_coverage_partial_for_single_sided_comparison():
    state = _state("Compare DPO and IPO.", _intent("comparison"))
    observation = _retrieval_observation(
        [{"chunk_id": "c1", "paper_id": "p-dpo", "final_score": 0.9}]
    )

    decision = KnowledgeCoverageEvaluator().evaluate(
        state=state,
        observation=observation,
    )

    assert decision.coverage == "partial"
    assert decision.recommended_strategy == ExecutionStrategy.DISCOVER_THEN_ANSWER
    assert "comparison needs evidence from multiple papers" in decision.missing_aspects


def test_coverage_keeps_unindexed_metadata_inside_knowledge_workflow():
    state = _state("Explain the saved paper.", _intent("factual_answer"))
    observation = ToolObservation(
        tool_name="retrieve_evidence",
        status="prerequisite_missing",
        result={"missing_paper_ids": ["p-saved"]},
        error_type="paper_not_retrievable",
        summary="Paper is not indexed.",
    )

    decision = KnowledgeCoverageEvaluator().evaluate(
        state=state,
        observation=observation,
    )

    assert decision.coverage == "insufficient"
    assert decision.recommended_strategy == ExecutionStrategy.KNOWLEDGE_ONLY
    assert decision.relevant_paper_ids == ["p-saved"]


def test_coverage_partial_for_latest_request_without_discovered_support():
    state = _state("Compare the latest DPO variants.", _intent("comparison"))
    observation = _retrieval_observation(
        [
            {"chunk_id": "c1", "paper_id": "p-old-a", "final_score": 0.9},
            {"chunk_id": "c2", "paper_id": "p-old-b", "final_score": 0.9},
        ]
    )

    decision = KnowledgeCoverageEvaluator().evaluate(
        state=state,
        observation=observation,
    )

    assert decision.coverage == "partial"
    assert decision.recommended_strategy == ExecutionStrategy.DISCOVER_THEN_ANSWER
    assert "request asks for latest or recent external coverage" in decision.missing_aspects


def test_coverage_sufficient_for_latest_request_after_discovery_support():
    state = _state("Compare the latest DPO variants.", _intent("comparison"))
    state.known_paper_ids = ["p-new-a", "p-new-b"]
    observation = _retrieval_observation(
        [
            {"chunk_id": "c1", "paper_id": "p-new-a", "final_score": 0.9},
            {"chunk_id": "c2", "paper_id": "p-new-b", "final_score": 0.9},
        ]
    )

    decision = KnowledgeCoverageEvaluator().evaluate(
        state=state,
        observation=observation,
    )

    assert decision.coverage == "sufficient"
    assert decision.recommended_strategy == ExecutionStrategy.KNOWLEDGE_ONLY


def test_llm_coverage_judge_downgrades_semantically_incomplete_evidence():
    state = _state("What are the limitations of DyG-RAG?", _intent("factual_answer"))
    observation = _retrieval_observation(
        [
            {
                "chunk_id": "c1",
                "paper_id": "p1",
                "final_score": 0.93,
                "text": "DyG-RAG introduces dynamic graph retrieval for temporal QA.",
            }
        ]
    )
    llm = _FakeCoverageLLM(
        """{
            "coverage": "partial",
            "missing_aspects": ["limitations are not covered"],
            "relevant_paper_ids": ["p1"],
            "confidence": 0.9,
            "rationale": "The evidence introduces the method but does not state limitations."
        }"""
    )

    decision = KnowledgeCoverageEvaluator(llm_client=llm).evaluate(
        state=state,
        observation=observation,
    )

    assert decision.coverage == "partial"
    assert decision.recommended_strategy == ExecutionStrategy.DISCOVER_THEN_ANSWER
    assert "limitations are not covered" in decision.missing_aspects
    assert "LLM semantic coverage judge" in decision.rationale
    assert llm.prompts


def test_llm_coverage_judge_low_confidence_keeps_deterministic_decision():
    state = _state("What are the limitations of DyG-RAG?", _intent("factual_answer"))
    observation = _retrieval_observation(
        [
            {
                "chunk_id": "c1",
                "paper_id": "p1",
                "final_score": 0.93,
                "text": "DyG-RAG lists limitations around temporal sparsity.",
            }
        ]
    )
    llm = _FakeCoverageLLM(
        """{
            "coverage": "partial",
            "missing_aspects": ["uncertain missing aspect"],
            "relevant_paper_ids": ["p1"],
            "confidence": 0.4,
            "rationale": "Maybe incomplete."
        }"""
    )

    decision = KnowledgeCoverageEvaluator(llm_client=llm).evaluate(
        state=state,
        observation=observation,
    )

    assert decision.coverage == "sufficient"
    assert decision.recommended_strategy == ExecutionStrategy.KNOWLEDGE_ONLY


def test_llm_coverage_judge_failure_falls_back_to_deterministic_decision():
    state = _state("Explain DPO.", _intent("factual_answer"))
    observation = _retrieval_observation(
        [{"chunk_id": "c1", "paper_id": "p1", "final_score": 0.9}]
    )

    decision = KnowledgeCoverageEvaluator(llm_client=_FailingCoverageLLM()).evaluate(
        state=state,
        observation=observation,
    )

    assert decision.coverage == "sufficient"
    assert decision.recommended_strategy == ExecutionStrategy.KNOWLEDGE_ONLY


def test_parse_llm_coverage_judgment_accepts_fenced_json():
    judgment = parse_llm_coverage_judgment(
        """```json
        {
            "coverage": "insufficient",
            "missing_aspects": ["off topic"],
            "relevant_paper_ids": [],
            "confidence": 0.8,
            "rationale": "The evidence is unrelated."
        }
        ```"""
    )

    assert judgment.coverage == "insufficient"
    assert judgment.missing_aspects == ["off topic"]


def _state(user_request: str, intent: RequestIntent) -> PlannerState:
    return PlannerState(
        user_request=user_request,
        runtime_state=AgentState(topic=user_request),
        request_intent=intent,
    )


def _intent(task_type: str) -> RequestIntent:
    return RequestIntent(
        task_type=task_type,
        topic="DPO",
        needs_retrieval=True,
        needs_ingestion=True,
        probe_existing_kb_first=True,
        finish_condition="retrieved_evidence",
        confidence=0.95,
        rationale="Test intent.",
    )


def _retrieval_observation(evidence: list[dict]) -> ToolObservation:
    return ToolObservation(
        tool_name="retrieve_evidence",
        status="success",
        result={"evidence": evidence, "retrieved": len(evidence)},
        summary=f"Retrieved {len(evidence)} evidence chunks.",
    )


class _FakeCoverageLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts = []

    def generate(self, prompt: str, **kwargs):
        self.prompts.append(prompt)
        return self.response


class _FailingCoverageLLM:
    def generate(self, prompt: str, **kwargs):
        raise RuntimeError("judge unavailable")
