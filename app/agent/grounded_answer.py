from __future__ import annotations

from typing import Any

from app.agent.planner_state import PlannerState
from app.llm.client import LLMClient, OpenAILLMClient
from app.retrieval.answering import (
    EvidenceChunk,
    build_grounded_answer_prompt,
    cited_ids_from_answer,
)


class GroundedAnswerService:
    """Generate the final answer after the planner chooses to finish."""

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or OpenAILLMClient()

    def generate(self, *, state: PlannerState, answer_task: str) -> dict[str, Any]:
        """Generate or return a final grounded artifact."""

        if state.retrieved_evidence:
            evidence_chunks = _evidence_chunks(state.retrieved_evidence)
            prompt = build_grounded_answer_prompt(
                query=answer_task,
                evidence_chunks=evidence_chunks,
            )
            answer = self.llm_client.generate(prompt).strip()
            cited_evidence_ids = cited_ids_from_answer(answer)
            cited_chunk_ids = [
                chunk.chunk_id
                for chunk in evidence_chunks
                if chunk.evidence_id in cited_evidence_ids
            ]
            return {
                "answer": answer,
                "answer_task": answer_task,
                "source": "retrieved_evidence",
                "evidence_chunks": [chunk.__dict__ for chunk in evidence_chunks],
                "cited_evidence_ids": cited_evidence_ids,
                "cited_chunk_ids": cited_chunk_ids,
            }

        if state.runtime_state.report:
            return {
                "answer": state.runtime_state.report,
                "answer_task": answer_task,
                "source": "generated_report",
            }

        if state.runtime_state.paper_summaries:
            return {
                "answer": [
                    summary.model_dump(mode="json")
                    for summary in state.runtime_state.paper_summaries
                ],
                "answer_task": answer_task,
                "source": "paper_summaries",
            }

        return {
            "answer": {
                "known_paper_ids": state.known_paper_ids,
                "saved_paper_ids": state.saved_paper_ids,
            },
            "answer_task": answer_task,
            "source": "planner_artifacts",
        }


def _evidence_chunks(records: list[dict[str, Any]]) -> list[EvidenceChunk]:
    chunks: list[EvidenceChunk] = []
    for index, record in enumerate(records, start=1):
        chunks.append(
            EvidenceChunk(
                evidence_id=f"E{index}",
                chunk_id=str(record.get("chunk_id") or f"chunk-{index}"),
                paper_id=str(record.get("paper_id") or ""),
                section=str(record.get("section") or ""),
                rank=int(record.get("rank") or index),
                semantic_score=float(record.get("semantic_score") or 0.0),
                metadata_score=float(record.get("metadata_score") or 0.0),
                final_score=float(record.get("final_score") or 0.0),
                text=str(record.get("text") or ""),
                metadata={
                    "title": record.get("title"),
                    "section_group": record.get("section_group"),
                },
            )
        )
    return chunks

