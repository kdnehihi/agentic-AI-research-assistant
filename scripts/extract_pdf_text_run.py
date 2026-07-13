from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from app.agent.state import AgentState, Paper
from app.storage.paper_store import PaperStore
from app.tools.pdf_text_tools import extract_pdf_text_for_selected_papers


def main() -> None:
    """Copy one PDF into PaperStore and extract raw/clean text files."""

    parser = argparse.ArgumentParser(
        description="Extract raw and clean text from one fetched paper PDF."
    )
    parser.add_argument(
        "--paper-id",
        required=True,
        help="Paper id, for example arxiv:2501.09136v4.",
    )
    parser.add_argument(
        "--pdf-path",
        required=True,
        help="Path to the source PDF to extract.",
    )
    parser.add_argument(
        "--title",
        default="Manual PDF extraction",
        help="Paper title used only for the temporary AgentState.",
    )
    parser.add_argument(
        "--url",
        default="manual://local-pdf",
        help="Paper URL used only for the temporary AgentState.",
    )
    parser.add_argument(
        "--keep-references",
        action="store_true",
        help="Keep the references section in clean_text.txt.",
    )
    args = parser.parse_args()

    source_pdf_path = Path(args.pdf_path)
    if not source_pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {source_pdf_path}")

    store = PaperStore()
    target_pdf_path = store.pdf_path(args.paper_id)
    shutil.copy2(source_pdf_path, target_pdf_path)

    paper = Paper(
        paper_id=args.paper_id,
        title=args.title,
        source="manual",
        url=args.url,
    )
    state = AgentState(topic="pdf text extraction", max_papers=1)
    state.set_selected_papers([paper])

    observation = extract_pdf_text_for_selected_papers(
        state=state,
        file_store=store,
        remove_references=not args.keep_references,
    )

    print("Status:", observation["status"])
    print("Processed:", observation["processed"])
    print("Failed:", observation["failed"])
    print("Fallback Abstract:", observation["fallback_abstract"])
    print("Summary:", observation["summary"])
    print("PDF Path:", target_pdf_path)
    print("Raw Text Path:", store.raw_text_path(args.paper_id))
    print("Clean Text Path:", store.clean_text_path(args.paper_id))

    if observation["errors"]:
        print("\nErrors:")
        for error in observation["errors"]:
            print(f"- {error['paper_id']}: {error['error']}")


if __name__ == "__main__":
    main()
