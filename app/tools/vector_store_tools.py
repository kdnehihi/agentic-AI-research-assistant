from __future__ import annotations

from typing import Any

from app.agent.state import AgentState, Paper
from app.config import get_settings
from app.retrieval.embedding_adapter import ExistingEmbeddingAdapter, ExistingEmbedderInterface
from app.services.chunk_indexing import PaperIndexMetadata, index_chunks
from app.tools.embedding_tools import DEFAULT_BGE_MODEL_NAME, load_bge_embedder, load_chunks_jsonl
from app.vectorstores.base import VectorStore
from app.vectorstores.chroma_store import ChromaVectorStore


def index_selected_paper_chunks(
    state: AgentState,
    *,
    knowledge_base_ids: tuple[str, ...] = ("default",),
    embedder: ExistingEmbedderInterface | None = None,
    vector_store: VectorStore | None = None,
    model_name: str = DEFAULT_BGE_MODEL_NAME,
    embedding_dimension: int = 384,
) -> dict[str, Any]:
    if not state.selected_papers:
        return {
            "status": "skipped",
            "indexed": 0,
            "failed": 0,
            "summary": "No selected papers available for vector indexing.",
        }

    settings = get_settings()
    embedder = embedder or ExistingEmbeddingAdapter(
        embedder=load_bge_embedder(model_name=model_name),
        model_name=model_name,
    )
    vector_store = vector_store or ChromaVectorStore(
        embedding_model_id=model_name,
        embedding_dimension=embedding_dimension,
    )

    indexed = 0
    failed = 0
    errors: list[dict[str, str]] = []

    for paper in state.selected_papers:
        if not paper.paper_id:
            failed += 1
            errors.append({"paper_id": "", "error": "Paper is missing paper_id."})
            continue

        try:
            chunks_path = state.paper_chunk_paths.get(paper.paper_id)
            if not chunks_path:
                raise FileNotFoundError(
                    f"No chunk path registered for paper_id={paper.paper_id}."
                )
            chunks = load_chunks_jsonl(chunks_path)
            result = index_chunks(
                chunks=chunks,
                paper_metadata=_paper_index_metadata(
                    paper=paper,
                    topic=state.topic,
                    knowledge_base_ids=knowledge_base_ids,
                ),
                embedder=embedder,
                vector_store=vector_store,
                batch_size=settings.vector_upsert_batch_size,
                metadata_schema_version=settings.metadata_schema_version,
            )
            indexed += result.upserted
            failed += result.failed
            if result.errors:
                errors.append(
                    {
                        "paper_id": paper.paper_id,
                        "error": "; ".join(result.errors),
                    }
                )
        except Exception as exc:
            failed += 1
            errors.append({"paper_id": paper.paper_id, "error": str(exc)})

    status = "success" if failed == 0 else "partial_success" if indexed else "failed"
    return {
        "status": status,
        "indexed": indexed,
        "failed": failed,
        "errors": errors,
        "collection_count": vector_store.count(),
        "summary": f"Indexed {indexed} chunks into the vector store. Failed: {failed}.",
    }


def _paper_index_metadata(
    paper: Paper,
    topic: str,
    knowledge_base_ids: tuple[str, ...],
) -> PaperIndexMetadata:
    return PaperIndexMetadata(
        paper_id=paper.paper_id or "",
        title=paper.title,
        source=paper.source,
        knowledge_base_ids=knowledge_base_ids,
        published_date=paper.published_date,
        authors=tuple(paper.authors),
        source_url=paper.url,
        topics=(topic,),
    )
