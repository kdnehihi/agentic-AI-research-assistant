from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from app.agent.state import AgentState
from app.storage.paper_store import PaperStore


MIN_EXTRACTED_TEXT_CHARS = 1000


def extract_pdf_text_for_selected_papers(
    state: AgentState,
    file_store: PaperStore | None = None,
    remove_references: bool = True,
) -> dict[str, Any]:
    """
    Agent tool.

    Extract text from selected paper PDFs, clean the extracted text,
    save raw_text.txt and clean_text.txt, and update state.paper_text_paths.

    This tool expects each selected paper PDF to already exist in the file store.
    Example expected path:
        data/papers/arxiv_2501_09136v4/paper.pdf
    """
    file_store = file_store or PaperStore()

    if not state.selected_papers:
        return {
            "status": "partial_success",
            "processed": 0,
            "failed": 0,
            "fallback_abstract": 0,
            "summary": "No selected papers available for PDF text extraction.",
        }

    paper_text_paths = dict(getattr(state, "paper_text_paths", {}))

    processed = 0
    failed = 0
    fallback_abstract = 0
    errors: list[dict[str, str]] = []

    for paper in state.selected_papers:
        try:
            pdf_path = file_store.pdf_path(paper.paper_id)

            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")

            raw_text = extract_text_from_pdf(pdf_path)
            clean_text = clean_pdf_text(raw_text)

            if remove_references:
                clean_text = remove_references_section(clean_text)

            if len(clean_text) < MIN_EXTRACTED_TEXT_CHARS:
                if paper.abstract:
                    clean_text = paper.abstract
                    fallback_abstract += 1
                else:
                    raise ValueError(
                        f"Extracted text too short and no abstract fallback available. "
                        f"Length={len(clean_text)}"
                    )

            raw_text_path = file_store.save_raw_text(
                paper_id=paper.paper_id,
                text=raw_text,
            )

            clean_text_path = file_store.save_clean_text(
                paper_id=paper.paper_id,
                text=clean_text,
            )

            paper_text_paths[paper.paper_id] = str(clean_text_path)

            processed += 1

        except Exception as exc:
            failed += 1
            errors.append(
                {
                    "paper_id": paper.paper_id,
                    "title": paper.title,
                    "error": str(exc),
                }
            )

    state.set_paper_text_paths(paper_text_paths)

    status = "success" if failed == 0 else "partial_success"

    return {
        "status": status,
        "processed": processed,
        "failed": failed,
        "fallback_abstract": fallback_abstract,
        "errors": errors,
        "summary": (
            f"Extracted text for {processed} papers. "
            f"Failed: {failed}. "
            f"Fallback abstract: {fallback_abstract}."
        ),
    }


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """
    Helper function.

    Extract raw text from a PDF file using PyMuPDF.

    Returns:
        Raw extracted text as a string.

    This is a helper, not an agent tool, so it returns str instead of dict.
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise ImportError(
            "PyMuPDF is required for PDF text extraction. "
            "Install it with: pip install pymupdf"
        ) from exc

    pages: list[str] = []

    with fitz.open(pdf_path) as doc:
        for page_index, page in enumerate(doc, start=1):
            page_text = page.get_text("text")

            if page_text.strip():
                pages.append(
                    f"\n\n[PAGE {page_index}]\n\n{page_text.strip()}"
                )

    return "\n".join(pages).strip()


def clean_pdf_text(text: str) -> str:
    """
    Clean raw PDF text extracted from PyMuPDF.

    This function handles common PDF extraction issues:
    - null characters
    - hyphenated line breaks
    - broken paragraph line breaks
    - repeated whitespace
    - excessive blank lines
    """
    if not text:
        return ""

    text = text.replace("\x00", " ")

    # "retrieval-\naugmented" -> "retrieval-augmented"
    text = re.sub(r"(\w)-\n(\w)", r"\1-\2", text)

    # Normalize Windows/Mac line endings.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Keep page markers separated.
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Join single newlines inside paragraphs.
    # Keeps paragraph breaks when there are 2+ newlines.
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # Collapse repeated spaces/tabs.
    text = re.sub(r"[ \t]+", " ", text)

    # Clean spaces around newlines.
    text = re.sub(r" *\n *", "\n", text)

    # Collapse excessive blank lines again.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def remove_references_section(text: str) -> str:
    """
    Remove the references/bibliography section from a paper.

    This is useful for RAG because references often create noisy chunks.
    """
    if not text:
        return ""

    patterns = [
        r"\n\s*references\s*\n",
        r"\n\s*bibliography\s*\n",
        r"\n\s*works cited\s*\n",
    ]

    lower_text = text.lower()

    for pattern in patterns:
        match = re.search(pattern, lower_text, flags=re.IGNORECASE)
        if match:
            return text[: match.start()].strip()

    return text.strip()


def extract_clean_text_from_pdf(
    pdf_path: str | Path,
    remove_references: bool = True,
) -> str:
    """
    Convenience helper.

    PDF path -> raw text -> clean text -> optional references removal.
    """
    raw_text = extract_text_from_pdf(pdf_path)
    clean_text = clean_pdf_text(raw_text)

    if remove_references:
        clean_text = remove_references_section(clean_text)

    return clean_text