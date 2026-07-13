from __future__ import annotations

import argparse

from app.agent.state import AgentState
from app.tools.fetch_selected_papers import remove_fetched_papers


def main() -> None:
    """Remove fetched paper directories, optionally as a dry run."""

    parser = argparse.ArgumentParser(
        description="Remove fetched paper files from data/papers."
    )
    parser.add_argument(
        "--output-dir",
        default="data/papers",
        help="Directory containing fetched paper folders.",
    )
    parser.add_argument(
        "--paper-id",
        action="append",
        dest="paper_ids",
        help="Paper id to remove. Can be provided multiple times.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Remove every fetched paper directory.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Actually delete files. Without this flag the script is a dry run.",
    )
    args = parser.parse_args()

    if not args.all and not args.paper_ids:
        parser.error("Use --all or at least one --paper-id.")

    state = AgentState(topic="remove fetched papers", max_papers=0)
    observation = remove_fetched_papers(
        state=state,
        paper_ids=args.paper_ids,
        output_dir=args.output_dir,
        remove_all=args.all,
        dry_run=not args.yes,
    )

    print("Status:", observation["status"])
    print("Dry Run:", observation["dry_run"])
    print("Requested:", observation["requested"])
    print("Matched:", observation["matched"])
    print("Removed:", observation["removed"])
    print("Missing:", observation["missing"])
    print("Summary:", observation["summary"])

    if observation["papers"]:
        print("\nPapers:")
        for paper in observation["papers"]:
            print(f"- {paper.get('paper_id')}: {paper['paper_dir']}")


if __name__ == "__main__":
    main()
