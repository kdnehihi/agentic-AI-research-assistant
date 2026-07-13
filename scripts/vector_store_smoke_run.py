from __future__ import annotations

import argparse
from pathlib import Path

from app.retrieval.embedding_adapter import DeterministicKeywordEmbedder
from app.retrieval.models import (
    RetrievalFilters,
    RetrievalRequest,
    SemanticMetadataHints,
)
from app.retrieval.retriever import MetadataAwareRetriever
from app.services.chunk_indexing import PaperIndexMetadata, index_chunks
from app.vectorstores.chroma_store import ChromaVectorStore


VOCABULARY = (
    "rag",
    "retrieval",
    "diversity",
    "rlhf",
    "reward",
    "evaluation",
    "agentic",
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the deterministic Chroma smoke test."""

    parser = argparse.ArgumentParser(description="Smoke-test local Chroma RAG retrieval.")
    parser.add_argument(
        "--path",
        default="data/vector_store/chroma_smoke",
        help="Persistent Chroma path for smoke-test data.",
    )
    return parser.parse_args()


def main() -> None:
    """Index deterministic fake chunks into Chroma and run retrieval scenarios."""

    args = parse_args()
    path = Path(args.path)
    embedder = DeterministicKeywordEmbedder(VOCABULARY)
    store = ChromaVectorStore(
        persist_path=path,
        collection_name="smoke_research_paper_chunks_v1",
        embedding_model_id=embedder.model_name,
        embedding_dimension=len(VOCABULARY),
        metadata_schema_version=1,
    )
    store.delete_by_paper("paper_rag")
    store.delete_by_paper("paper_rlhf")

    index_chunks(
        chunks=[
            _chunk("paper_rag::chunk:0", "Abstract", 0, "Agentic RAG uses retrieval for scientific literature."),
            _chunk("paper_rag::chunk:1", "Method", 1, "Diversity retrieval improves RAG context selection."),
            _chunk("paper_rag::chunk:2", "Results", 2, "Evaluation shows better retrieval performance."),
        ],
        paper_metadata=PaperIndexMetadata(
            paper_id="paper_rag",
            title="Agentic RAG for Scientific Literature",
            source="arxiv",
            knowledge_base_ids=("agentic_rag", "vector_database"),
            published_date="2025-04-01",
            topics=("agentic_rag",),
            methods=("diversity_retrieval",),
            tasks=("scientific_literature_search",),
        ),
        embedder=embedder,
        vector_store=store,
    )
    index_chunks(
        chunks=[
            _chunk("paper_rlhf::chunk:0", "Abstract", 0, "RLHF aligns reasoning models with reward feedback."),
            _chunk("paper_rlhf::chunk:1", "Results", 1, "Reward model evaluation improves reasoning quality."),
        ],
        paper_metadata=PaperIndexMetadata(
            paper_id="paper_rlhf",
            title="RLHF Reward Models",
            source="arxiv",
            knowledge_base_ids=("rlvr",),
            published_date="2024-02-15",
            topics=("rlhf",),
            methods=("reward_modeling",),
            tasks=("reasoning",),
        ),
        embedder=embedder,
        vector_store=store,
    )

    retriever = MetadataAwareRetriever(embedder=embedder, vector_store=store)
    scenarios = [
        (
            "semantic-only",
            RetrievalRequest(query="rag retrieval diversity", top_k=3),
        ),
        (
            "kb=rlvr",
            RetrievalRequest(
                query="reward model evaluation",
                top_k=3,
                filters=RetrievalFilters(knowledge_base_ids=("rlvr",)),
            ),
        ),
        (
            "paper=paper_rag",
            RetrievalRequest(
                query="scientific retrieval",
                top_k=3,
                filters=RetrievalFilters(paper_ids=("paper_rag",)),
            ),
        ),
        (
            "method/date",
            RetrievalRequest(
                query="diversity retrieval",
                top_k=3,
                filters=RetrievalFilters(
                    section_groups=("method",),
                    published_from_yyyymmdd=20250101,
                    published_to_yyyymmdd=20251231,
                ),
            ),
        ),
        (
            "soft hints",
            RetrievalRequest(
                query="retrieval",
                top_k=3,
                metadata_hints=SemanticMetadataHints(methods=("diversity_retrieval",)),
                metadata_weight=0.3,
            ),
        ),
    ]

    print(f"Chroma path: {path}")
    print(f"Collection count: {store.count()}")
    for label, request in scenarios:
        print(f"\n== {label} ==")
        for result in retriever.retrieve(request):
            preview = " ".join(result.document.split()[:12])
            print(
                f"{result.rank}. paper={result.paper_id} section={result.metadata['section']} "
                f"distance={result.distance:.4f} semantic={result.semantic_score:.4f} "
                f"metadata={result.metadata_score:.4f} final={result.final_score:.4f} "
                f"text={preview}"
            )


def _chunk(chunk_id: str, section: str, index: int, text: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "paper_id": chunk_id.split("::", 1)[0],
        "section": section,
        "section_index": index,
        "chunk_index": index,
        "section_chunk_index": 0,
        "start_word": 0,
        "end_word": len(text.split()),
        "section_word_count": len(text.split()),
        "word_count": len(text.split()),
        "text": text,
    }


if __name__ == "__main__":
    main()
