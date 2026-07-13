from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.retrieval.embedding_adapter import ExistingEmbeddingAdapter
from app.retrieval.evaluation import RetrievalEvalCase, evaluate_retrieval_results
from app.retrieval.models import RetrievalFilters, RetrievalRequest
from app.retrieval.retriever import MetadataAwareRetriever
from app.tools.embedding_tools import DEFAULT_BGE_MODEL_NAME, load_bge_embedder, load_chunks_jsonl
from app.vectorstores.chroma_store import ChromaVectorStore
from app.vectorstores.metadata import section_group_for


SECTION_GROUP_PRIORITY = (
    "abstract",
    "introduction",
    "method",
    "experiments",
    "results",
    "limitations",
    "conclusion",
)

BASIC_QUERIES = {
    "abstract": "What is the main idea of this paper?",
    "introduction": "What problem motivates this paper?",
    "method": "What method or approach is proposed?",
    "experiments": "What experimental setup is used?",
    "results": "What are the main reported findings?",
    "limitations": "What limitations are discussed?",
    "conclusion": "What conclusion does this paper make?",
}

EXPANDED_QUERIES = {
    "abstract": "What is the central contribution, scope, and main takeaway of this paper?",
    "introduction": "What research problem, limitation, or gap motivates the proposed work?",
    "method": "What methodology, algorithm, system design, or approach does the paper propose?",
    "experiments": "What experimental setup, datasets, benchmarks, or evaluation protocol is used?",
    "results": "What empirical results, findings, or performance conclusions are reported?",
    "limitations": "What limitations, threats to validity, or open challenges does the paper discuss?",
    "conclusion": "What final conclusions, implications, or future directions does the paper state?",
}


def parse_args() -> argparse.Namespace:
    """Parse CLI options for semantic and hybrid retrieval baseline experiments."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate semantic-only retrieval on already-fetched paper chunks. "
            "This does not search arXiv, fetch PDFs, chunk, embed, or reindex."
        )
    )
    parser.add_argument(
        "--papers-dir",
        default="data/papers",
        help="Directory containing paper folders with chunks.jsonl files.",
    )
    parser.add_argument(
        "--paper-id",
        action="append",
        default=[],
        help="Paper id to evaluate, e.g. arxiv:2505.18906v2. Can be repeated.",
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=5,
        help="Maximum number of existing papers to evaluate when --paper-id is omitted.",
    )
    parser.add_argument(
        "--max-cases-per-paper",
        type=int,
        default=5,
        help="Maximum section-query cases per paper.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-k retrieved chunks used for metrics.",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=None,
        help="Candidate count requested from Chroma before final semantic ranking.",
    )
    parser.add_argument(
        "--backend",
        choices=("local", "chroma"),
        default="local",
        help="Use local in-memory embeddings for ablation, or existing Chroma index.",
    )
    parser.add_argument(
        "--embedding-format",
        choices=("content_only", "section_content", "title_section_content", "compare"),
        default="compare",
        help="Chunk text format for local embedding ablation.",
    )
    parser.add_argument(
        "--ranking-mode",
        choices=("semantic", "hybrid", "compare"),
        default="semantic",
        help="Use semantic-only scoring, hybrid semantic+BM25+metadata, or compare both.",
    )
    parser.add_argument(
        "--semantic-weight",
        type=float,
        default=0.65,
        help="Hybrid weight for dense semantic score.",
    )
    parser.add_argument(
        "--bm25-weight",
        type=float,
        default=0.25,
        help="Hybrid weight for BM25 lexical score.",
    )
    parser.add_argument(
        "--metadata-weight",
        type=float,
        default=0.10,
        help="Hybrid weight for section metadata intent score.",
    )
    parser.add_argument(
        "--exclude-section-group",
        action="append",
        default=["other"],
        help="Section group to exclude from local retrieval candidates. Defaults to 'other'.",
    )
    parser.add_argument(
        "--query-style",
        choices=("basic", "expanded", "both"),
        default="both",
        help="Evaluate short queries, clearer expanded queries, or both.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_BGE_MODEL_NAME,
        help="Embedding model id used by the existing Chroma collection.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print aggregate metrics, not each retrieved chunk.",
    )
    return parser.parse_args()


def main() -> None:
    """Run retrieval evaluation over existing chunk files without fetching papers."""

    args = parse_args()
    settings = get_settings()
    papers_dir = Path(args.papers_dir)
    selected_chunk_files = select_chunk_files(
        papers_dir=papers_dir,
        paper_ids=tuple(args.paper_id),
        max_papers=args.max_papers,
    )
    cases_by_style = build_eval_cases_from_existing_chunks(
        papers_dir=papers_dir,
        paper_ids=tuple(args.paper_id),
        max_papers=args.max_papers,
        max_cases_per_paper=args.max_cases_per_paper,
        query_style=args.query_style,
    )
    if not cases_by_style:
        print(f"No eval cases found under {papers_dir}.")
        return

    embedder = ExistingEmbeddingAdapter(
        embedder=load_bge_embedder(model_name=args.model_name),
        model_name=args.model_name,
    )
    candidate_k = args.candidate_k or settings.retrieval_default_candidate_k

    print("===== SEMANTIC RETRIEVAL BASELINE =====")
    print(f"Papers dir: {papers_dir}")
    print(f"Backend: {args.backend}")
    print("Filters: paper_id only")
    print("metadata_weight: 0.0 for semantic mode")
    print(
        "hybrid weights: "
        f"semantic={args.semantic_weight}, "
        f"bm25={args.bm25_weight}, "
        f"metadata={args.metadata_weight}"
    )
    print("section filters: disabled")
    print("metadata hints: disabled")
    print("BM25/lexical score: disabled")
    print(f"Excluded section groups: {tuple(args.exclude_section_group)}")
    print_word_count_report(selected_chunk_files)

    if args.backend == "chroma":
        vector_store = ChromaVectorStore(
            embedding_model_id=args.model_name,
            embedding_dimension=384,
        )
        retriever = MetadataAwareRetriever(embedder=embedder, vector_store=vector_store)
        print(f"Chroma count: {vector_store.count()}")
        for style, cases in cases_by_style.items():
            summary, details_by_case = evaluate_with_chroma(
                cases=cases,
                retriever=retriever,
                top_k=args.top_k,
                candidate_k=max(candidate_k, args.top_k),
            )
            print_summary(
                label=f"{style.upper()} / CHROMA",
                summary=summary,
                details_by_case=details_by_case,
                summary_only=args.summary_only,
            )
        return

    formats = (
        ("content_only", "section_content", "title_section_content")
        if args.embedding_format == "compare"
        else (args.embedding_format,)
    )
    paper_chunks = load_paper_chunks(selected_chunk_files)
    ranking_modes = (
        ("semantic", "hybrid")
        if args.ranking_mode == "compare"
        else (args.ranking_mode,)
    )
    for embedding_format in formats:
        for style, cases in cases_by_style.items():
            for ranking_mode in ranking_modes:
                summary, details_by_case = evaluate_with_local_embeddings(
                    cases=cases,
                    paper_chunks=paper_chunks,
                    embedder=embedder,
                    embedding_format=embedding_format,
                    top_k=args.top_k,
                    excluded_section_groups=tuple(args.exclude_section_group),
                    ranking_mode=ranking_mode,
                    semantic_weight=args.semantic_weight,
                    bm25_weight=args.bm25_weight,
                    metadata_weight=args.metadata_weight,
                )
                print_summary(
                    label=f"{style.upper()} / {embedding_format} / {ranking_mode}",
                    summary=summary,
                    details_by_case=details_by_case,
                    summary_only=args.summary_only,
                )


def evaluate_with_chroma(
    cases: list[RetrievalEvalCase],
    retriever: MetadataAwareRetriever,
    top_k: int,
    candidate_k: int,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    """Evaluate the Chroma-backed retriever with paper-id filters only."""

    results_by_case: dict[str, list[str]] = {}
    details_by_case: dict[str, list[dict[str, Any]]] = {}

    for case in cases:
        retrieved = retriever.retrieve(
            RetrievalRequest(
                query=case.query,
                top_k=top_k,
                candidate_k=candidate_k,
                filters=RetrievalFilters(
                    paper_ids=(case.paper_id,) if case.paper_id else (),
                ),
                metadata_weight=0.0,
            )
        )
        result_key = case.case_id or case.query
        results_by_case[result_key] = [result.chunk_id for result in retrieved]
        details_by_case[result_key] = [
            {
                "rank": result.rank,
                "chunk_id": result.chunk_id,
                "document": result.document,
                "metadata": result.metadata,
                "semantic_score": result.semantic_score,
                "distance": result.distance,
            }
            for result in retrieved
        ]

    summary = evaluate_retrieval_results(
        cases=cases,
        results_by_query=results_by_case,
        top_k=top_k,
    )
    return summary.to_dict(), details_by_case


def evaluate_with_local_embeddings(
    cases: list[RetrievalEvalCase],
    paper_chunks: dict[str, dict[str, Any]],
    embedder: ExistingEmbeddingAdapter,
    embedding_format: str,
    top_k: int,
    excluded_section_groups: tuple[str, ...] = ("other",),
    ranking_mode: str = "semantic",
    semantic_weight: float = 0.65,
    bm25_weight: float = 0.25,
    metadata_weight: float = 0.10,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    """Evaluate local embeddings with optional semantic-only or hybrid reranking."""

    results_by_case: dict[str, list[str]] = {}
    details_by_case: dict[str, list[dict[str, Any]]] = {}
    embeddings_by_paper = build_local_embeddings(
        paper_chunks=paper_chunks,
        embedder=embedder,
        embedding_format=embedding_format,
        excluded_section_groups=excluded_section_groups,
    )

    for case in cases:
        result_key = case.case_id or case.query
        paper_id = case.paper_id or ""
        local_rows = embeddings_by_paper.get(paper_id, [])
        query_embedding = embedder.embed_query(case.query)
        bm25_scores = bm25_scores_for_query(case.query, local_rows)
        normalized_bm25_scores = normalize_scores(bm25_scores)
        scored_rows = []
        for row in local_rows:
            semantic_score = dot_product(query_embedding, row["embedding"])
            bm25_score = normalized_bm25_scores.get(row["chunk_id"], 0.0)
            metadata_score = metadata_intent_score(case.query, row)
            final_score = final_local_score(
                ranking_mode=ranking_mode,
                semantic_score=semantic_score,
                bm25_score=bm25_score,
                metadata_score=metadata_score,
                semantic_weight=semantic_weight,
                bm25_weight=bm25_weight,
                metadata_weight=metadata_weight,
            )
            scored_rows.append(
                (
                    final_score,
                    {
                        **row,
                        "semantic_score": semantic_score,
                        "bm25_score": bm25_score,
                        "metadata_score": metadata_score,
                    },
                )
            )

        scored_rows.sort(key=lambda item: (-item[0], item[1]["chunk_id"]))
        top_rows = scored_rows[:top_k]
        results_by_case[result_key] = [row["chunk_id"] for _, row in top_rows]
        details_by_case[result_key] = [
            {
                "rank": rank,
                "chunk_id": row["chunk_id"],
                "document": row["chunk"]["text"],
                "metadata": {
                    "section": row["chunk"].get("section"),
                    "section_group": row["section_group"],
                    "word_count": row["word_count"],
                },
                "semantic_score": row["semantic_score"],
                "bm25_score": row["bm25_score"],
                "metadata_score": row["metadata_score"],
                "final_score": score,
                "distance": 1.0 - row["semantic_score"],
            }
            for rank, (score, row) in enumerate(top_rows, start=1)
        ]

    summary = evaluate_retrieval_results(
        cases=cases,
        results_by_query=results_by_case,
        top_k=top_k,
    )
    return summary.to_dict(), details_by_case


def build_local_embeddings(
    paper_chunks: dict[str, dict[str, Any]],
    embedder: ExistingEmbeddingAdapter,
    embedding_format: str,
    excluded_section_groups: tuple[str, ...],
) -> dict[str, list[dict[str, Any]]]:
    """Embed selected chunks in memory for fast format/ranking comparisons."""

    embeddings_by_paper: dict[str, list[dict[str, Any]]] = {}
    for paper_id, payload in paper_chunks.items():
        rows = [
            {
                "chunk": chunk,
                "chunk_id": str(chunk["chunk_id"]),
                "section_group": section_group_for(str(chunk.get("section", ""))),
                "word_count": int(chunk.get("word_count", len(str(chunk.get("text", "")).split()))),
                "embedding_text": embedding_text_for_chunk(
                    chunk=chunk,
                    title=str(payload.get("title", "")),
                    embedding_format=embedding_format,
                )
            }
            for chunk in payload["chunks"]
        ]
        rows = [
            row
            for row in rows
            if row["section_group"] not in excluded_section_groups
        ]
        embeddings = embedder.embed_documents([row["embedding_text"] for row in rows])
        embeddings_by_paper[paper_id] = [
            {**row, "embedding": embedding}
            for row, embedding in zip(rows, embeddings)
        ]
    return embeddings_by_paper


def load_paper_chunks(chunk_files: list[Path]) -> dict[str, dict[str, Any]]:
    """Load chunks grouped by paper id from selected chunks.jsonl files."""

    paper_chunks: dict[str, dict[str, Any]] = {}
    for chunks_path in chunk_files:
        chunks = load_chunks_jsonl(chunks_path)
        if not chunks:
            continue

        paper_id = str(chunks[0]["paper_id"])
        paper_chunks[paper_id] = {
            "title": load_paper_title(chunks_path.parent),
            "chunks": chunks,
        }
    return paper_chunks


def load_paper_title(paper_dir: Path) -> str:
    """Load a paper title from metadata.json or fall back to directory name."""

    metadata_path = paper_dir / "metadata.json"
    if not metadata_path.exists():
        return paper_dir.name.replace("_", " ")

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return paper_dir.name.replace("_", " ")

    return str(metadata.get("title") or paper_dir.name.replace("_", " "))


def embedding_text_for_chunk(
    chunk: dict[str, Any],
    title: str,
    embedding_format: str,
) -> str:
    """Build the exact text variant sent to the embedder for one chunk."""

    section = str(chunk.get("section", ""))
    text = str(chunk.get("text", ""))
    if embedding_format == "content_only":
        return text
    if embedding_format == "section_content":
        return f"Section: {section}\nContent:\n{text}"
    if embedding_format == "title_section_content":
        return f"Title: {title}\nSection: {section}\nContent:\n{text}"
    raise ValueError(
        "embedding_format must be content_only, section_content, or "
        "title_section_content."
    )


def dot_product(left: list[float], right: list[float]) -> float:
    """Compute dot product after checking embedding dimensions match."""

    if len(left) != len(right):
        raise ValueError(
            f"Embedding dimension mismatch: query={len(left)} chunk={len(right)}"
        )
    return sum(left_value * right_value for left_value, right_value in zip(left, right))


def final_local_score(
    ranking_mode: str,
    semantic_score: float,
    bm25_score: float,
    metadata_score: float,
    semantic_weight: float,
    bm25_weight: float,
    metadata_weight: float,
) -> float:
    """Return semantic-only score or weighted hybrid score for local eval."""

    if ranking_mode == "semantic":
        return semantic_score
    if ranking_mode != "hybrid":
        raise ValueError("ranking_mode must be semantic or hybrid.")

    total_weight = semantic_weight + bm25_weight + metadata_weight
    if total_weight <= 0:
        raise ValueError("Hybrid score weights must sum to a positive value.")

    return (
        semantic_weight * semantic_score
        + bm25_weight * bm25_score
        + metadata_weight * metadata_score
    ) / total_weight


def metadata_intent_score(query: str, row: dict[str, Any]) -> float:
    """Score whether a chunk section matches inferred query intent."""

    expected_groups = set(infer_section_groups_from_query(query))
    if not expected_groups:
        return 0.0
    return 1.0 if row["section_group"] in expected_groups else 0.0


def infer_section_groups_from_query(query: str) -> tuple[str, ...]:
    """Infer target section groups from query wording for local hybrid eval."""

    lowered = query.lower()
    if any(term in lowered for term in ("main idea", "central contribution", "main takeaway", "scope")):
        return ("abstract",)
    if any(term in lowered for term in ("motivat", "research problem", "gap")):
        return ("introduction",)
    if any(term in lowered for term in ("method", "methodology", "algorithm", "approach", "system design")):
        return ("method",)
    if any(term in lowered for term in ("experimental setup", "dataset", "benchmark", "evaluation protocol")):
        return ("experiments",)
    if any(term in lowered for term in ("result", "finding", "performance conclusion", "empirical")):
        return ("results",)
    if any(term in lowered for term in ("limitation", "threat", "open challenge")):
        return ("limitations",)
    if any(term in lowered for term in ("conclusion", "future direction", "implication")):
        return ("conclusion",)
    return ()


def bm25_scores_for_query(
    query: str,
    rows: list[dict[str, Any]],
    k1: float = 1.5,
    b: float = 0.75,
) -> dict[str, float]:
    """Compute BM25 lexical scores for local candidate rows."""

    tokenized_docs = [tokenize(row["embedding_text"]) for row in rows]
    query_terms = tokenize(query)
    if not rows or not query_terms:
        return {}

    doc_count = len(tokenized_docs)
    doc_lengths = [len(tokens) for tokens in tokenized_docs]
    avg_doc_length = sum(doc_lengths) / doc_count if doc_count else 0.0
    document_frequency: Counter[str] = Counter()
    for tokens in tokenized_docs:
        document_frequency.update(set(tokens))

    scores: dict[str, float] = {}
    for row, tokens, doc_length in zip(rows, tokenized_docs, doc_lengths):
        term_counts = Counter(tokens)
        score = 0.0
        for term in query_terms:
            frequency = term_counts.get(term, 0)
            if frequency == 0:
                continue

            idf = math.log(
                1.0
                + (doc_count - document_frequency[term] + 0.5)
                / (document_frequency[term] + 0.5)
            )
            denominator = frequency + k1 * (
                1.0 - b + b * doc_length / max(avg_doc_length, 1e-9)
            )
            score += idf * frequency * (k1 + 1.0) / denominator
        scores[row["chunk_id"]] = score

    return scores


def normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    """Normalize positive scores to 0-1 for hybrid weighting."""

    if not scores:
        return {}

    max_score = max(scores.values())
    if max_score <= 0.0:
        return {key: 0.0 for key in scores}

    return {
        key: value / max_score
        for key, value in scores.items()
    }


def tokenize(text: str) -> list[str]:
    """Tokenize text for local BM25 scoring."""

    return re.findall(r"[a-z0-9]+", text.lower())


def print_word_count_report(chunk_files: list[Path]) -> None:
    """Print chunk word-count diagnostics before retrieval evaluation."""

    chunks = []
    for chunks_path in chunk_files:
        chunks.extend(load_chunks_jsonl(chunks_path))

    word_counts = [
        int(chunk.get("word_count", len(str(chunk.get("text", "")).split())))
        for chunk in chunks
    ]
    if not word_counts:
        print("Chunk word counts: no chunks")
        return

    sorted_counts = sorted(word_counts)
    over_700 = sum(1 for count in word_counts if count > 700)
    over_900 = sum(1 for count in word_counts if count > 900)
    print(
        "Chunk word counts: "
        f"count={len(word_counts)}, "
        f"min={sorted_counts[0]}, "
        f"median={percentile(sorted_counts, 0.5)}, "
        f"p90={percentile(sorted_counts, 0.9)}, "
        f"max={sorted_counts[-1]}, "
        f">700={over_700}, "
        f">900={over_900}"
    )


def percentile(sorted_values: list[int], fraction: float) -> int:
    """Return an integer percentile from an already sorted list."""

    if not sorted_values:
        return 0

    index = round((len(sorted_values) - 1) * fraction)
    return sorted_values[index]


def build_eval_cases_from_existing_chunks(
    papers_dir: Path,
    paper_ids: tuple[str, ...] = (),
    max_papers: int = 5,
    max_cases_per_paper: int = 5,
    query_style: str = "both",
) -> dict[str, list[RetrievalEvalCase]]:
    """Build gold section-based eval cases from existing chunk files."""

    if max_papers <= 0:
        raise ValueError("max_papers must be positive.")
    if max_cases_per_paper <= 0:
        raise ValueError("max_cases_per_paper must be positive.")

    selected_chunk_files = select_chunk_files(
        papers_dir=papers_dir,
        paper_ids=paper_ids,
        max_papers=max_papers,
    )
    styles = ("basic", "expanded") if query_style == "both" else (query_style,)
    cases_by_style: dict[str, list[RetrievalEvalCase]] = {style: [] for style in styles}

    for chunks_path in selected_chunk_files:
        chunks = load_chunks_jsonl(chunks_path)
        if not chunks:
            continue

        paper_id = str(chunks[0]["paper_id"])
        for style in styles:
            cases_by_style[style].extend(
                build_eval_cases_for_chunks(
                    chunks=chunks,
                    paper_id=paper_id,
                    query_style=style,
                    max_cases=max_cases_per_paper,
                )
            )

    return {
        style: cases
        for style, cases in cases_by_style.items()
        if cases
    }


def select_chunk_files(
    papers_dir: Path,
    paper_ids: tuple[str, ...] = (),
    max_papers: int = 5,
) -> list[Path]:
    """Select chunks.jsonl files by paper id or by the first max_papers files."""

    chunk_files = sorted(papers_dir.glob("*/chunks.jsonl"))
    if not paper_ids:
        return chunk_files[:max_papers]

    wanted = set(paper_ids)
    selected: list[Path] = []
    for chunks_path in chunk_files:
        chunks = load_chunks_jsonl(chunks_path)
        if chunks and str(chunks[0].get("paper_id")) in wanted:
            selected.append(chunks_path)

    return selected


def build_eval_cases_for_chunks(
    chunks: list[dict[str, Any]],
    paper_id: str,
    query_style: str,
    max_cases: int,
) -> list[RetrievalEvalCase]:
    """Create section-group retrieval cases for one paper's chunks."""

    query_map = query_map_for_style(query_style)
    chunks_by_group = chunks_by_section_group(chunks)
    cases: list[RetrievalEvalCase] = []

    for section_group in SECTION_GROUP_PRIORITY:
        relevant_chunks = chunks_by_group.get(section_group, [])
        if not relevant_chunks:
            continue

        relevant_chunk_ids = tuple(str(chunk["chunk_id"]) for chunk in relevant_chunks)
        cases.append(
            RetrievalEvalCase(
                query=query_map[section_group],
                relevant_chunk_ids=relevant_chunk_ids,
                relevance_by_chunk_id={chunk_id: 3 for chunk_id in relevant_chunk_ids},
                case_id=f"{paper_id}::{query_style}::{section_group}",
                paper_id=paper_id,
                gold_section=gold_section_label(relevant_chunks),
                section_groups=(section_group,),
            )
        )
        if len(cases) >= max_cases:
            break

    return cases


def query_map_for_style(query_style: str) -> dict[str, str]:
    """Return basic or expanded query templates for section-group eval."""

    if query_style == "basic":
        return BASIC_QUERIES
    if query_style == "expanded":
        return EXPANDED_QUERIES
    raise ValueError("query_style must be 'basic' or 'expanded'.")


def chunks_by_section_group(
    chunks: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Group chunks by normalized section group for gold-label generation."""

    chunks_by_group: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks:
        section_group = section_group_for(str(chunk.get("section", "")))
        if section_group not in BASIC_QUERIES:
            continue
        chunks_by_group.setdefault(section_group, []).append(chunk)
    return chunks_by_group


def gold_section_label(chunks: list[dict[str, Any]]) -> str:
    """Build a readable label for the gold sections in one eval case."""

    sections: list[str] = []
    for chunk in chunks:
        section = str(chunk.get("section", "Full Text"))
        if section not in sections:
            sections.append(section)
    return ", ".join(sections)


def print_summary(
    label: str,
    summary: dict[str, Any],
    details_by_case: dict[str, list[dict[str, Any]]],
    summary_only: bool = False,
) -> None:
    """Print aggregate metrics and optional per-query retrieved chunk details."""

    top_k = summary["top_k"]
    print(f"\n===== {label} =====")
    print(f"Cases: {summary['num_cases']}")
    print(f"Hit Rate@{top_k}: {summary['hit_rate_at_k']:.3f}")
    print(f"Recall@{top_k}: {summary['mean_recall_at_k']:.3f}")
    print(f"Precision@{top_k}: {summary['mean_precision_at_k']:.3f}")
    print(f"MRR: {summary['mrr']:.3f}")
    print(f"nDCG@{top_k}: {summary['mean_ndcg_at_k']:.3f}")

    if summary_only:
        return

    for result in summary["results"]:
        result_key = result["case_id"]
        print(f"\nQuery: {result['query']}")
        print(f"Paper: {result['paper_id']}")
        print(f"Gold section group: {result['section_groups']}")
        print(f"Gold section: {result['gold_section']}")
        print(f"Gold chunks: {result['relevant_chunk_ids']}")
        print(f"First relevant rank: {result['first_relevant_rank']}")
        for retrieved in details_by_case.get(result_key, []):
            metadata = retrieved["metadata"]
            preview = " ".join(retrieved["document"].split()[:18])
            print(
                f"  {retrieved['rank']}. {retrieved['chunk_id']} "
                f"section={metadata.get('section')} "
                f"words={metadata.get('word_count', '?')} "
                f"final={retrieved.get('final_score', retrieved['semantic_score']):.3f} "
                f"semantic={retrieved['semantic_score']:.3f} "
                f"bm25={retrieved.get('bm25_score', 0.0):.3f} "
                f"metadata={retrieved.get('metadata_score', 0.0):.3f} "
                f"distance={retrieved['distance']:.3f} "
                f"text={preview}"
            )

if __name__ == "__main__":
    main()
