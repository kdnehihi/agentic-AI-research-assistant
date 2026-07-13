from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from app.agent.state import AgentState
from app.storage.paper_store import PaperStore


MIN_EXTRACTED_TEXT_CHARS = 1000

PRESERVED_HYPHENATED_TERMS = {
    "answer-level",
    "binary-search",
    "chain-of",
    "chunk-free",
    "chunk-length",
    "chunk-set",
    "chunk-sets",
    "cosine-similarity",
    "dataset-level",
    "df-rag",
    "distance-based",
    "diversity-aware",
    "diversity-based",
    "diversity-focused",
    "domain-specific",
    "dual-perspective",
    "few-shot",
    "fine-tuning",
    "fixed-size",
    "follow-up",
    "full-context",
    "in-context",
    "inference-time",
    "intent-aware",
    "large-language",
    "learning-to",
    "llm-based",
    "long-context",
    "long-tail",
    "long-term",
    "multi-hop",
    "multi-qa",
    "multi-step",
    "multi-task",
    "neural-based",
    "non-informative",
    "non-multi",
    "one-shot",
    "open-domain",
    "open-source",
    "optimally-diverse",
    "optimally-tuned",
    "plug-and",
    "point-wise",
    "pre-training",
    "query-adaptive",
    "query-aware",
    "query-level",
    "query-specific",
    "question-answering",
    "rag-based",
    "reasoning-intensive",
    "retrieval-augmented",
    "self-rag",
    "self-reflection",
    "self-taught",
    "single-hop",
    "single-run",
    "task-specific",
    "test-time",
    "tie-breaking",
    "time-consuming",
    "token-processing",
    "trade-off",
    "training-dependent",
    "training-free",
    "tree-organized",
    "vanilla-rag",
    "well-calibrated",
    "well-suited",
    "word-chunks",
    "zero-shot",
}


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
            pdf_path = _pdf_path_for_paper(
                paper=paper,
                file_store=file_store,
            )

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


def _pdf_path_for_paper(paper, file_store: PaperStore) -> Path:
    """
    Prefer the fetched full_text_path, then common fetched/full-store PDF paths.
    """
    if paper.full_text_path:
        full_text_path = Path(paper.full_text_path)
        if full_text_path.suffix.lower() == ".pdf":
            return full_text_path

    fetched_pdf_path = file_store.paper_dir(paper.paper_id) / "full_text.pdf"
    if fetched_pdf_path.exists():
        return fetched_pdf_path

    return file_store.pdf_path(paper.paper_id)


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

    # Normalize Windows/Mac line endings before line-based cleanup.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove page markers inserted during extraction.
    text = re.sub(r"(?m)^\s*\[PAGE\s+\d+\]\s*$", "\n", text)

    # Remove arXiv footer/header and standalone page numbers.
    text = re.sub(
        r"\b\d+\s+arXiv:\d{4}\.\d+(?:v\d+)?\s+\[[^\]]+\]\s+"
        r"\d{1,2}\s+[A-Za-z]+\s+\d{4}\b",
        " ",
        text,
    )
    text = re.sub(r"(?m)^\s*\d{1,3}\s*$", "", text)

    # "retrieval-\naugmented" -> "retrieval-augmented"
    text = re.sub(r"(\w)-\n(\w)", r"\1-\2", text)

    # Keep page markers separated.
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Join single newlines inside paragraphs.
    # Keeps paragraph breaks when there are 2+ newlines.
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # Collapse repeated spaces/tabs.
    text = re.sub(r"[ \t]+", " ", text)

    # Repair in-line PDF hyphenation such as "informa-tion" while preserving
    # meaningful compounds such as "DF-RAG", "query-aware", and "multi-hop".
    text = _repair_inline_hyphenation(text)

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
        r"(?im)(?:^|\n)\s*references\s*(?:\n|$)",
        r"(?im)(?:^|\n)\s*bibliography\s*(?:\n|$)",
        r"(?im)(?:^|\n)\s*works cited\s*(?:\n|$)",
        r"(?m)\bReferences\s+(?=[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?[, ])",
        r"(?m)\bBibliography\s+(?=[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?[, ])",
        r"(?m)\bWorks cited\s+(?=[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?[, ])",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return text[: match.start()].strip()

    return text.strip()


def _repair_inline_hyphenation(text: str) -> str:
    """Repair accidental PDF hyphenation while preserving known compound terms."""

    def replace_match(match: re.Match[str]) -> str:
        """Remove one inline hyphen unless the term is intentionally hyphenated."""

        token = match.group(0)
        if token.lower() in PRESERVED_HYPHENATED_TERMS:
            return token

        return token.replace("-", "")

    return re.sub(r"\b[A-Za-z]{2,}-[A-Za-z]{2,}\b", replace_match, text)


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
