from __future__ import annotations

from dataclasses import asdict
from typing import Any

from app.agent.state import AgentState
from app.config import get_settings
from app.retrieval.embedding_adapter import ExistingEmbeddingAdapter, ExistingEmbedderInterface
from app.retrieval.evaluation import (
    RetrievalEvalCase,
    evaluate_retrieval_results,
)
from app.retrieval.models import RetrievalFilters, RetrievalRequest
from app.retrieval.retriever import MetadataAwareRetriever
from app.tools.embedding_tools import DEFAULT_BGE_MODEL_NAME, load_bge_embedder, load_chunks_jsonl
from app.vectorstores.metadata import section_group_for
from app.vectorstores.base import VectorStore
from app.vectorstores.chroma_store import ChromaVectorStore


SECTION_GROUP_PRIORITY = (
    "method",
    "experiments",
    "results",
    "abstract",
    "introduction",
    "conclusion",
    "limitations",
    "discussion",
    "background",
)


def evaluate_retrieval_from_selected_chunks(
    state: AgentState,
    *,
    top_k: int | None = None,
    max_cases: int = 5,
    embedder: ExistingEmbedderInterface | None = None,
    vector_store: VectorStore | None = None,
    model_name: str = DEFAULT_BGE_MODEL_NAME,
    embedding_dimension: int = 384,
) -> dict[str, Any]:
    """
    Evaluate semantic retrieval against real chunks from selected papers.
    """
    if not state.selected_papers:
        return {
            "status": "skipped",
            "cases": 0,
            "summary": "No selected papers available for retrieval evaluation.",
        }

    settings = get_settings()
    resolved_top_k = top_k or settings.retrieval_default_top_k
    cases = _build_eval_cases_from_state(state=state, max_cases=max_cases)
    if not cases:
        return {
            "status": "skipped",
            "cases": 0,
            "summary": "No chunk files available for retrieval evaluation.",
        }

    embedder = embedder or ExistingEmbeddingAdapter(
        embedder=load_bge_embedder(model_name=model_name),
        model_name=model_name,
    )
    vector_store = vector_store or ChromaVectorStore(
        embedding_model_id=model_name,
        embedding_dimension=embedding_dimension,
    )
    retriever = MetadataAwareRetriever(embedder=embedder, vector_store=vector_store)

    results_by_case: dict[str, list[str]] = {}
    retrieved_details: dict[str, list[dict[str, Any]]] = {}
    errors: list[dict[str, str]] = []

    for case in cases:
        try:
            request = RetrievalRequest(
                query=case.query,
                top_k=resolved_top_k,
                candidate_k=max(settings.retrieval_default_candidate_k, resolved_top_k),
                filters=RetrievalFilters(
                    paper_ids=(case.paper_id,) if case.paper_id else (),
                ),
                metadata_weight=settings.retrieval_metadata_weight,
            )
            retrieved = retriever.retrieve(request)
            result_key = case.case_id or case.query
            results_by_case[result_key] = [result.chunk_id for result in retrieved]
            retrieved_details[result_key] = [asdict(result) for result in retrieved]
        except Exception as exc:
            result_key = case.case_id or case.query
            results_by_case[result_key] = []
            retrieved_details[result_key] = []
            errors.append({"query": case.query, "error": str(exc)})

    summary = evaluate_retrieval_results(
        cases=cases,
        results_by_query=results_by_case,
        top_k=resolved_top_k,
    )
    eval_results = summary.to_dict()
    eval_results["cases"] = [asdict(case) for case in cases]
    eval_results["retrieved_details"] = retrieved_details
    eval_results["errors"] = errors
    eval_results["retrieval_filter_mode"] = "paper_id_only"
    state.set_eval_results(eval_results)

    status = "success" if not errors else "partial_success"
    if errors and len(errors) == len(cases):
        status = "failed"

    return {
        "status": status,
        "cases": len(cases),
        "top_k": resolved_top_k,
        "hit_rate_at_k": summary.hit_rate_at_k,
        "mean_recall_at_k": summary.mean_recall_at_k,
        "mean_precision_at_k": summary.mean_precision_at_k,
        "mrr": summary.mrr,
        "mean_ndcg_at_k": summary.mean_ndcg_at_k,
        "errors": errors,
        "summary": (
            f"Evaluated retrieval on {len(cases)} real chunk cases: "
            f"Hit@{resolved_top_k}={summary.hit_rate_at_k:.2f}, "
            f"Recall@{resolved_top_k}={summary.mean_recall_at_k:.2f}, "
            f"Precision@{resolved_top_k}={summary.mean_precision_at_k:.2f}, "
            f"MRR={summary.mrr:.2f}, nDCG@{resolved_top_k}={summary.mean_ndcg_at_k:.2f}."
        ),
    }


def _build_eval_cases_from_state(
    state: AgentState,
    max_cases: int,
) -> list[RetrievalEvalCase]:
    if max_cases <= 0:
        raise ValueError("max_cases must be positive.")

    cases: list[RetrievalEvalCase] = []
    for paper in state.selected_papers:
        if not paper.paper_id:
            continue

        chunks_path = state.paper_chunk_paths.get(paper.paper_id)
        if not chunks_path:
            continue

        chunks = load_chunks_jsonl(chunks_path)
        for case in _cases_from_chunks(
            chunks=chunks,
            paper_id=paper.paper_id,
        ):
            cases.append(case)
            if len(cases) >= max_cases:
                return cases

    return cases


def _cases_from_chunks(
    chunks: list[dict[str, Any]],
    paper_id: str,
) -> list[RetrievalEvalCase]:
    cases: list[RetrievalEvalCase] = []
    chunks_by_group = _chunks_by_section_group(chunks)

    for section_group in SECTION_GROUP_PRIORITY:
        relevant_chunks = chunks_by_group.get(section_group, [])
        if not relevant_chunks:
            continue

        cases.append(
            _case_from_section_group(
                paper_id=paper_id,
                section_group=section_group,
                chunks=relevant_chunks,
            )
        )

    if cases:
        return cases

    return [_case_from_section_group(paper_id=paper_id, section_group="other", chunks=chunks[:1])]


def _chunks_by_section_group(
    chunks: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    chunks_by_group: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks:
        section_group = section_group_for(str(chunk.get("section", "")))
        chunks_by_group.setdefault(section_group, []).append(chunk)
    return chunks_by_group


def _case_from_section_group(
    paper_id: str,
    section_group: str,
    chunks: list[dict[str, Any]],
) -> RetrievalEvalCase:
    relevant_chunk_ids = tuple(str(chunk["chunk_id"]) for chunk in chunks)
    return RetrievalEvalCase(
        query=_query_for_section_group(section_group),
        relevant_chunk_ids=relevant_chunk_ids,
        relevance_by_chunk_id={chunk_id: 3 for chunk_id in relevant_chunk_ids},
        case_id=f"{paper_id}::{section_group}",
        paper_id=paper_id,
        gold_section=_gold_section_label(chunks),
        section_groups=(section_group,),
    )


def _query_for_section_group(section_group: str) -> str:
    if section_group == "method":
        return "What method or approach is proposed?"
    if section_group == "experiments":
        return "What experimental setup is used?"
    if section_group == "results":
        return "What are the main reported findings?"
    if section_group == "abstract":
        return "What is the main idea of this paper?"
    if section_group == "introduction":
        return "What problem motivates this paper?"
    if section_group == "conclusion":
        return "What conclusion does this paper make?"
    if section_group == "limitations":
        return "What limitations are discussed?"
    if section_group == "discussion":
        return "What analysis or discussion does the paper provide?"
    if section_group == "background":
        return "What background does the paper provide?"
    return "What evidence is relevant in this paper?"


def _gold_section_label(chunks: list[dict[str, Any]]) -> str:
    sections: list[str] = []
    for chunk in chunks:
        section = str(chunk.get("section", "Full Text"))
        if section not in sections:
            sections.append(section)
    return ", ".join(sections)
