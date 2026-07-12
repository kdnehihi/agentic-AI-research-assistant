from __future__ import annotations

import argparse

from app.agent.state import AgentState, Paper
from app.storage.paper_store import PaperStore
from app.tools.embedding_tools import (
    DEFAULT_BGE_MODEL_NAME,
    embed_selected_paper_chunks,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed an existing chunks.jsonl file with BAAI BGE."
    )
    parser.add_argument(
        "--paper-id",
        required=True,
        help="Paper id, for example arxiv:2601.17212v1.",
    )
    parser.add_argument(
        "--chunks-path",
        default=None,
        help="Optional path to chunks.jsonl. Defaults to PaperStore convention.",
    )
    parser.add_argument(
        "--title",
        default="Manual embedding run",
        help="Paper title used for temporary AgentState metadata.",
    )
    parser.add_argument(
        "--url",
        default="manual://embedding-run",
        help="Paper URL used for temporary AgentState metadata.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_BGE_MODEL_NAME,
        help="SentenceTransformer model name.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Embedding batch size.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    state = AgentState(topic="manual embedding", max_papers=1)
    paper = Paper(
        paper_id=args.paper_id,
        title=args.title,
        source="local",
        url=args.url,
    )
    state.set_selected_papers([paper])

    if args.chunks_path:
        state.set_paper_chunk_paths({args.paper_id: args.chunks_path})

    store = PaperStore()
    observation = embed_selected_paper_chunks(
        state=state,
        file_store=store,
        model_name=args.model_name,
        batch_size=args.batch_size,
    )

    print("Status:", observation["status"])
    print("Processed:", observation["processed"])
    print("Embedded Chunks:", observation["embedded_chunks"])
    print("Model:", observation["model_name"])
    print("Embeddings Path:", state.paper_embedding_paths.get(args.paper_id))
    if observation.get("errors"):
        print("Errors:", observation["errors"])


if __name__ == "__main__":
    main()
