from __future__ import annotations

import argparse

from app.tools.embedding_tools import (
    DEFAULT_BGE_MODEL_NAME,
    load_bge_embedder,
    load_embeddings_jsonl,
    search_embedded_chunks,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search an embeddings.jsonl file with a BAAI BGE query."
    )
    parser.add_argument(
        "--embeddings-path",
        required=True,
        help="Path to embeddings.jsonl.",
    )
    parser.add_argument(
        "--query",
        required=True,
        help="Retrieval query to test.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to return.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_BGE_MODEL_NAME,
        help="SentenceTransformer model name.",
    )
    parser.add_argument(
        "--include-front-matter",
        action="store_true",
        help="Include title/author metadata chunks in retrieval results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    embedder = load_bge_embedder(model_name=args.model_name)
    embeddings = load_embeddings_jsonl(args.embeddings_path)
    results = search_embedded_chunks(
        query=args.query,
        embeddings=embeddings,
        embedder=embedder,
        top_k=args.top_k,
        excluded_sections=() if args.include_front_matter else ("Front Matter",),
    )

    print(f"Query: {args.query}")
    print(f"Embeddings: {args.embeddings_path}")
    print(f"Model: {args.model_name}")
    print()

    for rank, result in enumerate(results, start=1):
        preview = " ".join(result.text.split()[:80])
        print(f"{rank}. score={result.score:.4f}")
        print(f"   chunk_id={result.chunk_id}")
        print(f"   section={result.section}")
        print(f"   words={result.start_word}:{result.end_word} count={result.word_count}")
        print(f"   text={preview}")
        print()


if __name__ == "__main__":
    main()
