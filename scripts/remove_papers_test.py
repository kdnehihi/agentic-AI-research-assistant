import argparse

from app.agent.state import AgentState
from app.storage.paper_store import PaperStore
from app.tools.knowledge_base_tools import remove_papers_from_kb

SAVED_RAG_PAPER_IDS = [
    "arxiv:2506.10408v1",
    "arxiv:2501.09136v4",
    "arxiv:2603.07379v1",
    "arxiv:2510.13910v2",
    "arxiv:2510.25518v1",
]


def main():
    parser = argparse.ArgumentParser(
        description="Remove test papers from the local SQLite knowledge base."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Remove every paper currently stored in the knowledge base.",
    )
    args = parser.parse_args()

    store = PaperStore()
    paper_ids = store.get_all_paper_ids() if args.all else SAVED_RAG_PAPER_IDS
    state = AgentState(topic="remove papers from knowledge base", max_papers=0)

    observation = remove_papers_from_kb(
        state=state,
        paper_ids=paper_ids,
        store=store,
    )

    print("Paper IDs requested for removal:")
    for paper_id in paper_ids:
        print(f"- {paper_id}")

    print("\nObservation:", observation)


if __name__ == "__main__":
    main()
