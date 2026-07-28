from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

from app.agent.planner_state import PlannerState
from app.llm.client import LLMClient, create_default_llm_client
from app.retrieval.answering import (
    EvidenceChunk,
    build_grounded_answer_prompt,
    cited_ids_from_answer,
)


class GroundedAnswerService:
    """Generate the final answer after the planner chooses to finish."""

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or create_default_llm_client()

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

        metadata_count = _metadata_observation_count(state)
        if metadata_count:
            return {
                "answer": f"Found {metadata_count} stored paper metadata records.",
                "answer_task": answer_task,
                "source": "stored_metadata",
            }

        discovered_count = len(
            state.runtime_state.selected_papers
            or state.runtime_state.candidate_papers
            or state.candidate_paper_ids
        )
        if discovered_count:
            return {
                "answer": (
                    f"Found {discovered_count} papers. Review the paper cards "
                    "below and save the ones you want to keep for RAG."
                ),
                "answer_task": answer_task,
                "source": "paper_metadata",
            }

        saved_count = len(state.saved_paper_ids or state.retrievable_paper_ids)
        if saved_count:
            return {
                "answer": f"Found {saved_count} saved papers in the workspace.",
                "answer_task": answer_task,
                "source": "stored_metadata",
            }

        return {
            "answer": "No answerable artifacts were produced.",
            "answer_task": answer_task,
            "source": "planner_artifacts",
        }


class StreamingGroundedAnswerService(GroundedAnswerService):
    """Grounded answer service that emits final-answer text chunks."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        *,
        on_token: Callable[[str], None],
    ) -> None:
        super().__init__(llm_client)
        self.on_token = on_token

    def generate(self, *, state: PlannerState, answer_task: str) -> dict[str, Any]:
        """Generate the final answer while forwarding token chunks."""

        if not state.retrieved_evidence:
            return super().generate(state=state, answer_task=answer_task)

        evidence_chunks = _evidence_chunks(state.retrieved_evidence)
        prompt = build_grounded_answer_prompt(
            query=answer_task,
            evidence_chunks=evidence_chunks,
        )
        answer_parts: list[str] = []
        for token in _stream_or_generate(self.llm_client, prompt):
            if not token:
                continue
            answer_parts.append(token)
            self.on_token(token)

        answer = "".join(answer_parts).strip()
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


def _stream_or_generate(llm_client: LLMClient, prompt: str) -> Iterator[str]:
    stream_generate = getattr(llm_client, "stream_generate", None)
    if callable(stream_generate):
        yield from stream_generate(prompt)
        return
    text = llm_client.generate(prompt)
    words = text.split(" ")
    for index, word in enumerate(words):
        suffix = " " if index < len(words) - 1 else ""
        yield word + suffix


def _metadata_observation_count(state: PlannerState) -> int:
    observation = state.latest_observation
    if observation is None or observation.status not in {"success", "partial_success"}:
        return 0
    if observation.tool_name == "list_papers":
        return int(observation.result.get("count") or 0)
    if observation.tool_name == "get_paper_metadata":
        papers = observation.result.get("papers") or []
        return len(papers) if isinstance(papers, list) else 0
    return 0


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
