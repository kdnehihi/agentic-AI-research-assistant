from __future__ import annotations

import argparse

from app.agent.state import AgentState, Paper
from app.storage.paper_store import PaperStore
from app.tools.chunking_tools import chunk_selected_papers_by_section


def main() -> None:
    """Chunk one manually provided clean_text.txt file and print the output path."""

    parser = argparse.ArgumentParser(
        description="Chunk one cleaned paper text file by detected sections."
    )
    parser.add_argument(
        "--paper-id",
        required=True,
        help="Paper id, for example arxiv:2601.17212v1.",
    )
    parser.add_argument(
        "--clean-text-path",
        required=True,
        help="Path to clean_text.txt.",
    )
    parser.add_argument(
        "--title",
        default="Manual chunking run",
        help="Paper title used only for the temporary AgentState.",
    )
    parser.add_argument(
        "--url",
        default="manual://clean-text",
        help="Paper URL used only for the temporary AgentState.",
    )
    parser.add_argument("--min-words", type=int, default=700)
    parser.add_argument("--target-words", type=int, default=850)
    parser.add_argument("--max-words", type=int, default=900)
    parser.add_argument("--overlap-words", type=int, default=200)
    args = parser.parse_args()

    paper = Paper(
        paper_id=args.paper_id,
        title=args.title,
        source="manual",
        url=args.url,
    )
    state = AgentState(topic="manual chunking", max_papers=1)
    state.set_selected_papers([paper])
    state.set_paper_text_paths({args.paper_id: args.clean_text_path})

    store = PaperStore()
    observation = chunk_selected_papers_by_section(
        state=state,
        file_store=store,
        min_chunk_words=args.min_words,
        target_chunk_words=args.target_words,
        max_chunk_words=args.max_words,
        overlap_words=args.overlap_words,
    )

    print("Status:", observation["status"])
    print("Processed:", observation["processed"])
    print("Failed:", observation["failed"])
    print("Chunks:", observation["chunks"])
    print("Summary:", observation["summary"])
    print("Chunks Path:", state.paper_chunk_paths.get(args.paper_id))

    if observation["errors"]:
        print("\nErrors:")
        for error in observation["errors"]:
            print(f"- {error['paper_id']}: {error['error']}")


if __name__ == "__main__":
    main()
