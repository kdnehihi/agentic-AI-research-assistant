from __future__ import annotations

import argparse

from app.agent.runner import ARXIV_SEARCH_AND_FETCH_WORKFLOW, AgentRunner
from app.agent.state import AgentState
from app.tools.registry import ToolRegistry


DEFAULT_TOPIC = (
    "rag"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run arXiv search, select papers, and fetch selected PDFs."
    )
    parser.add_argument(
        "--topic",
        default=DEFAULT_TOPIC,
        help="Research topic to search for.",
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=3,
        help="Number of selected papers to fetch.",
    )
    args = parser.parse_args()

    state = AgentState(
        topic=args.topic,
        max_papers=args.max_papers,
    )
    runner = AgentRunner(
        state=state,
        registry=ToolRegistry(),
    )

    runner.run_workflow(workflow=ARXIV_SEARCH_AND_FETCH_WORKFLOW)

    print("\n===== FETCHED SELECTED PAPERS =====\n")
    if not state.selected_papers:
        print("No selected papers were fetched.")
        return

    for index, paper in enumerate(state.selected_papers, start=1):
        print(f"{index}. {paper.title}")
        print(f"   Paper ID: {paper.paper_id}")
        print(f"   URL: {paper.url}")
        print(f"   Full text path: {paper.full_text_path}")


if __name__ == "__main__":
    main()
