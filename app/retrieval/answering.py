from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from app.llm.client import LLMClient
from app.retrieval.models import RetrievedChunk, RetrievalRequest


@dataclass(frozen=True)
class EvidenceChunk:
    """Context chunk exposed to the LLM with a stable citation id."""

    evidence_id: str
    chunk_id: str
    paper_id: str
    section: str
    rank: int
    semantic_score: float
    metadata_score: float
    final_score: float
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RAGAnswer:
    """Structured answer payload with the generated text and cited evidence."""

    query: str
    answer: str
    evidence_chunks: list[EvidenceChunk]
    cited_evidence_ids: list[str]
    cited_chunk_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert the answer and evidence records into JSON-friendly data."""

        return {
            "query": self.query,
            "answer": self.answer,
            "evidence_chunks": [
                asdict(evidence_chunk)
                for evidence_chunk in self.evidence_chunks
            ],
            "cited_evidence_ids": self.cited_evidence_ids,
            "cited_chunk_ids": self.cited_chunk_ids,
        }


class RetrievalAugmentedAnswerer:
    """Coordinate retrieval, prompt construction, LLM generation, and citations."""

    def __init__(
        self,
        retriever: Any,
        llm_client: LLMClient,
    ) -> None:
        self.retriever = retriever
        self.llm_client = llm_client

    def answer(
        self,
        request: RetrievalRequest,
        *,
        max_context_chars: int = 12000,
        max_chunk_chars: int = 1800,
        llm_kwargs: dict[str, Any] | None = None,
    ) -> RAGAnswer:
        """Retrieve evidence for a query and ask the LLM for a grounded answer."""

        retrieved_chunks = self.retriever.retrieve(request)
        evidence_chunks = build_evidence_chunks(
            retrieved_chunks=retrieved_chunks,
            max_context_chars=max_context_chars,
            max_chunk_chars=max_chunk_chars,
        )
        prompt = build_grounded_answer_prompt(
            query=request.query,
            evidence_chunks=evidence_chunks,
        )
        answer_text = self.llm_client.generate(
            prompt,
            **(llm_kwargs or {}),
        ).strip()
        cited_evidence_ids = cited_ids_from_answer(answer_text)
        cited_chunk_ids = [
            evidence_chunk.chunk_id
            for evidence_chunk in evidence_chunks
            if evidence_chunk.evidence_id in cited_evidence_ids
        ]

        return RAGAnswer(
            query=request.query,
            answer=answer_text,
            evidence_chunks=evidence_chunks,
            cited_evidence_ids=cited_evidence_ids,
            cited_chunk_ids=cited_chunk_ids,
        )


def build_evidence_chunks(
    retrieved_chunks: list[RetrievedChunk],
    max_context_chars: int = 12000,
    max_chunk_chars: int = 1800,
) -> list[EvidenceChunk]:
    """Convert retrieved chunks into a size-bounded evidence list for prompting."""

    if max_context_chars <= 0:
        raise ValueError("max_context_chars must be positive.")
    if max_chunk_chars <= 0:
        raise ValueError("max_chunk_chars must be positive.")

    evidence_chunks: list[EvidenceChunk] = []
    used_chars = 0
    for chunk in retrieved_chunks:
        remaining_context_chars = max_context_chars - used_chars
        if remaining_context_chars <= 0:
            break

        text = _truncate_text(
            chunk.document,
            min(max_chunk_chars, remaining_context_chars),
        )
        if not text:
            continue

        evidence_chunks.append(
            EvidenceChunk(
                evidence_id=f"E{len(evidence_chunks) + 1}",
                chunk_id=chunk.chunk_id,
                paper_id=chunk.paper_id,
                section=str(chunk.metadata.get("section", "")),
                rank=chunk.rank,
                semantic_score=chunk.semantic_score,
                metadata_score=chunk.metadata_score,
                final_score=chunk.final_score,
                text=text,
                metadata=dict(chunk.metadata),
            )
        )
        used_chars += len(text)

    return evidence_chunks


def build_grounded_answer_prompt(
    query: str,
    evidence_chunks: list[EvidenceChunk],
) -> str:
    """Build the strict RAG prompt that forces evidence citations or refusal."""

    evidence_text = "\n\n".join(
        _format_evidence_chunk(evidence_chunk)
        for evidence_chunk in evidence_chunks
    )
    if not evidence_text:
        evidence_text = "No retrieved evidence was available."

    return f"""
You are a careful research assistant answering questions about academic papers.

Answer the user question using only the evidence chunks below.

Rules:
- Every factual sentence must cite at least one evidence id like [E1].
- Only answer claims that are directly supported by the evidence chunks.
- If the evidence is missing, indirect, or only loosely related, output exactly this sentence and nothing else: "I do not have enough evidence from the retrieved chunks to answer that."
- Do not use outside knowledge.
- Do not infer implementation details, datasets, metrics, hardware, or dates unless they are explicitly stated in the evidence.
- Prefer a concise answer.

Question:
{query}

Evidence:
{evidence_text}

Answer:
""".strip()


def cited_ids_from_answer(answer: str) -> list[str]:
    """Extract unique evidence ids cited in model output, preserving order."""

    seen: set[str] = set()
    cited_ids: list[str] = []
    for match in re.finditer(r"\[E(\d+)\]", answer):
        evidence_id = f"E{match.group(1)}"
        if evidence_id not in seen:
            seen.add(evidence_id)
            cited_ids.append(evidence_id)
    return cited_ids


def _format_evidence_chunk(evidence_chunk: EvidenceChunk) -> str:
    """Format one evidence chunk as plain text for the LLM prompt."""

    title = evidence_chunk.metadata.get("title", "")
    return (
        f"[{evidence_chunk.evidence_id}]\n"
        f"paper_id: {evidence_chunk.paper_id}\n"
        f"chunk_id: {evidence_chunk.chunk_id}\n"
        f"title: {title}\n"
        f"section: {evidence_chunk.section}\n"
        f"rank: {evidence_chunk.rank}\n"
        f"text:\n{evidence_chunk.text}"
    )


def _truncate_text(text: str, max_chars: int) -> str:
    """Normalize and truncate chunk text without cutting the final word in half."""

    normalized_text = " ".join(text.split())
    if len(normalized_text) <= max_chars:
        return normalized_text
    if max_chars <= 4:
        return normalized_text[:max_chars]
    truncated = normalized_text[: max_chars - 4].rsplit(" ", 1)[0]
    if not truncated:
        truncated = normalized_text[: max_chars - 4]
    return truncated.rstrip() + " ..."
