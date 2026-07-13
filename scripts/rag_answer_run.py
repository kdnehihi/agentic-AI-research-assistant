from __future__ import annotations

import argparse

from app.agent.state import AgentState
from app.tools.rag_answer_tools import answer_question_with_retrieval


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for one grounded RAG answer run."""

    parser = argparse.ArgumentParser(
        description="Answer a question using retrieved paper chunks and OpenAI."
    )
    parser.add_argument(
        "--query",
        required=True,
        help="Question to answer from retrieved chunks.",
    )
    parser.add_argument(
        "--paper-id",
        action="append",
        default=[],
        help="Restrict retrieval to a paper id. Can be repeated.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of retrieved chunks to put in the LLM context.",
    )
    return parser.parse_args()


def main() -> None:
    """Run one RAG answer query and print cited evidence chunks."""

    args = parse_args()
    state = AgentState(topic=args.query, max_papers=args.top_k)
    observation = answer_question_with_retrieval(
        state=state,
        query=args.query,
        paper_ids=tuple(args.paper_id),
        top_k=args.top_k,
    )

    print("===== RAG ANSWER =====")
    print(f"Status: {observation['status']}")
    if observation["status"] != "success":
        print(f"Error: {observation.get('error')}")
        return

    print(f"Query: {observation['query']}")
    print()
    print(observation["answer"])
    print("\n===== CITED CHUNKS =====")
    for evidence in observation["evidence_chunks"]:
        cited_marker = "*" if evidence["chunk_id"] in observation["cited_chunk_ids"] else " "
        print(
            f"{cited_marker} {evidence['evidence_id']} "
            f"paper={evidence['paper_id']} "
            f"chunk={evidence['chunk_id']} "
            f"section={evidence['section']} "
            f"rank={evidence['rank']}"
        )


if __name__ == "__main__":
    main()
