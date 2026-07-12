from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from app.agent.state import AgentState
from app.storage.paper_store import PaperStore


DEFAULT_MIN_CHUNK_WORDS = 700
DEFAULT_TARGET_CHUNK_WORDS = 850
DEFAULT_MAX_CHUNK_WORDS = 900
DEFAULT_OVERLAP_WORDS = 200

SECTION_ALIASES = {
    "abstract": "Abstract",
    "introduction": "Introduction",
    "related work": "Related Work",
    "background": "Background",
    "method": "Method",
    "methods": "Method",
    "methodology": "Methodology",
    "approach": "Method",
    "model": "Method",
    "experiments": "Experiments",
    "experimental setup": "Experimental Setup",
    "experimental details": "Experimental Details",
    "results": "Results",
    "experimental results": "Experimental Results",
    "discussion": "Discussion",
    "discussion and analysis": "Discussion",
    "analysis": "Analysis",
    "limitations": "Limitations",
    "ethical considerations": "Ethical Considerations",
    "conclusion": "Conclusion",
    "appendix": "Appendix",
    "additional experimental results": "Appendix",
    "error analysis": "Appendix",
}

SECTION_CANDIDATES = sorted(
    SECTION_ALIASES,
    key=lambda section: len(section.split()),
    reverse=True,
)


@dataclass(frozen=True)
class Section:
    title: str
    text: str


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    paper_id: str
    section: str
    section_index: int
    chunk_index: int
    section_chunk_index: int
    start_word: int
    end_word: int
    section_word_count: int
    word_count: int
    text: str


def chunk_selected_papers_by_section(
    state: AgentState,
    file_store: PaperStore | None = None,
    min_chunk_words: int = DEFAULT_MIN_CHUNK_WORDS,
    target_chunk_words: int = DEFAULT_TARGET_CHUNK_WORDS,
    max_chunk_words: int = DEFAULT_MAX_CHUNK_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
) -> dict[str, Any]:
    """
    Chunk selected papers section by section and save chunks.jsonl per paper.
    """
    # Agent entrypoint: read clean_text.txt for each selected paper and save
    # section-aware chunks beside that paper's local files.
    file_store = file_store or PaperStore()

    if not state.selected_papers:
        return {
            "status": "skipped",
            "processed": 0,
            "failed": 0,
            "chunks": 0,
            "summary": "No selected papers available for chunking.",
        }

    paper_chunk_paths = dict(state.paper_chunk_paths)
    processed = 0
    failed = 0
    total_chunks = 0
    errors: list[dict[str, str]] = []

    for paper in state.selected_papers:
        paper_id = paper.paper_id
        if not paper_id:
            failed += 1
            errors.append(
                {
                    "paper_id": "",
                    "title": paper.title,
                    "error": "Paper is missing paper_id.",
                }
            )
            continue

        try:
            clean_text_path = _clean_text_path_for_paper(
                state=state,
                file_store=file_store,
                paper_id=paper_id,
            )
            if not clean_text_path.exists():
                raise FileNotFoundError(f"Clean text file not found: {clean_text_path}")

            text = clean_text_path.read_text(encoding="utf-8")
            chunks = chunk_text_by_sections(
                text=text,
                paper_id=paper_id,
                min_chunk_words=min_chunk_words,
                target_chunk_words=target_chunk_words,
                max_chunk_words=max_chunk_words,
                overlap_words=overlap_words,
            )
            chunks_path = save_chunks_jsonl(
                chunks=chunks,
                path=file_store.paper_dir(paper_id) / "chunks.jsonl",
            )

            paper_chunk_paths[paper_id] = str(chunks_path)
            processed += 1
            total_chunks += len(chunks)
        except Exception as exc:
            failed += 1
            errors.append(
                {
                    "paper_id": paper_id,
                    "title": paper.title,
                    "error": str(exc),
                }
            )

    state.set_paper_chunk_paths(paper_chunk_paths)

    if failed == 0:
        status = "success"
    elif processed > 0:
        status = "partial_success"
    else:
        status = "failed"

    return {
        "status": status,
        "processed": processed,
        "failed": failed,
        "chunks": total_chunks,
        "errors": errors,
        "summary": (
            f"Chunked {processed} papers into {total_chunks} chunks. "
            f"Failed: {failed}."
        ),
    }


def detect_sections(text: str) -> list[Section]:
    """
    Detect common research-paper sections from cleaned text.

    The cleaned text often has headings inline with paragraph text, so this
    detects section starts by known heading phrases rather than relying only
    on line breaks.
    """
    # Clean PDF text often has section headings inline, so we normalize
    # whitespace first and then look for known research-paper headings.
    normalized_text = _normalize_spaces(text)
    if not normalized_text:
        return []

    matches = _section_matches(normalized_text)
    if not matches:
        return [Section(title="Full Text", text=normalized_text)]

    sections: list[Section] = []

    first_start = matches[0][0]
    if first_start > 0:
        preface = normalized_text[:first_start].strip()
        if preface:
            sections.append(Section(title="Front Matter", text=preface))

    for index, (start, end, title) in enumerate(matches):
        next_start = matches[index + 1][0] if index + 1 < len(matches) else len(normalized_text)
        section_text = normalized_text[end:next_start].strip()

        if section_text:
            sections.append(Section(title=title, text=section_text))

    return sections


def chunk_text_by_sections(
    text: str,
    paper_id: str,
    min_chunk_words: int = DEFAULT_MIN_CHUNK_WORDS,
    target_chunk_words: int = DEFAULT_TARGET_CHUNK_WORDS,
    max_chunk_words: int = DEFAULT_MAX_CHUNK_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
) -> list[TextChunk]:
    # Validate chunk settings once before section detection and splitting.
    _validate_chunk_settings(
        min_chunk_words=min_chunk_words,
        target_chunk_words=target_chunk_words,
        max_chunk_words=max_chunk_words,
        overlap_words=overlap_words,
    )

    sections = detect_sections(text)
    chunks: list[TextChunk] = []

    for section_index, section in enumerate(sections):
        chunks.extend(
            chunk_section(
                section=section,
                paper_id=paper_id,
                section_index=section_index,
                start_chunk_index=len(chunks),
                min_chunk_words=min_chunk_words,
                target_chunk_words=target_chunk_words,
                max_chunk_words=max_chunk_words,
                overlap_words=overlap_words,
            )
        )

    return chunks


def chunk_section(
    section: Section,
    paper_id: str,
    section_index: int = 0,
    start_chunk_index: int = 0,
    min_chunk_words: int = DEFAULT_MIN_CHUNK_WORDS,
    target_chunk_words: int = DEFAULT_TARGET_CHUNK_WORDS,
    max_chunk_words: int = DEFAULT_MAX_CHUNK_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
) -> list[TextChunk]:
    # Split only inside one section. Short sections are kept intact so chunks
    # do not mix Introduction/Method/Results content.
    words = section.text.split()
    if not words:
        return []

    section_word_count = len(words)

    if len(words) <= max_chunk_words:
        return [
            _build_chunk(
                paper_id=paper_id,
                section=section.title,
                section_index=section_index,
                chunk_index=start_chunk_index,
                section_chunk_index=0,
                words=words,
                start_word=0,
                end_word=len(words),
                section_word_count=section_word_count,
            )
        ]

    chunks: list[TextChunk] = []
    start = 0

    while start < len(words):
        end = min(start + target_chunk_words, len(words))
        remaining = len(words) - end

        if 0 < remaining < min_chunk_words and len(words) - start <= max_chunk_words:
            end = len(words)

        chunks.append(
            _build_chunk(
                paper_id=paper_id,
                section=section.title,
                section_index=section_index,
                chunk_index=start_chunk_index + len(chunks),
                section_chunk_index=len(chunks),
                words=words[start:end],
                start_word=start,
                end_word=end,
                section_word_count=section_word_count,
            )
        )

        if end == len(words):
            break

        start = max(end - overlap_words, start + 1)

    return chunks


def save_chunks_jsonl(chunks: list[TextChunk], path: str | Path) -> Path:
    # Store one JSON object per line. Each chunk carries enough metadata to
    # locate it back to paper, section, global chunk index, and section span.
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")

    return output_path


def _section_matches(text: str) -> list[tuple[int, int, str]]:
    # Collect candidate heading matches from all known heading aliases.
    matches: list[tuple[int, int, str]] = []

    for heading in SECTION_CANDIDATES:
        title = SECTION_ALIASES[heading]
        pattern = _section_heading_pattern(heading)

        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            start = match.start("heading")
            end = match.end("heading")
            matches.append((start, end, title))

    return _deduplicate_section_matches(matches)


def _section_heading_pattern(heading: str) -> str:
    # Accept headings at text start, after sentence punctuation, or after a
    # newline, with optional numeric prefixes such as "3.1 Methodology".
    escaped_heading = re.escape(heading).replace(r"\ ", r"\s+")
    if heading == "abstract":
        return (
            r"(?P<prefix>(?:^|\s+))"
            rf"(?P<heading>{escaped_heading})"
            r"(?=\s+[A-Z0-9(]|\s*$)"
        )

    return (
        r"(?P<prefix>(?:^|(?<=[.!?])\s+|\n+))"
        r"(?:(?:\d+|[A-Z])(?:\.\d+)*\s+)?"
        rf"(?P<heading>{escaped_heading})"
        r"(?=\s+[A-Z0-9(]|\s*$)"
    )


def _deduplicate_section_matches(
    matches: list[tuple[int, int, str]],
) -> list[tuple[int, int, str]]:
    # Prefer the longest heading at a given position so "Related Work" wins
    # over a shorter overlapping candidate.
    matches = sorted(matches, key=lambda item: (item[0], -(item[1] - item[0])))
    deduped: list[tuple[int, int, str]] = []
    occupied_starts: set[int] = set()

    for start, end, title in matches:
        if start in occupied_starts:
            continue

        if deduped and start < deduped[-1][1]:
            continue

        deduped.append((start, end, title))
        occupied_starts.add(start)

    return deduped


def _build_chunk(
    paper_id: str,
    section: str,
    section_index: int,
    chunk_index: int,
    section_chunk_index: int,
    words: list[str],
    start_word: int,
    end_word: int,
    section_word_count: int,
) -> TextChunk:
    # Build a serializable chunk with flat metadata for vector stores and
    # debugging without needing side-channel lookup tables.
    return TextChunk(
        chunk_id=f"{paper_id}::chunk:{chunk_index}",
        paper_id=paper_id,
        section=section,
        section_index=section_index,
        chunk_index=chunk_index,
        section_chunk_index=section_chunk_index,
        start_word=start_word,
        end_word=end_word,
        section_word_count=section_word_count,
        word_count=len(words),
        text=" ".join(words),
    )


def _clean_text_path_for_paper(
    state: AgentState,
    file_store: PaperStore,
    paper_id: str,
) -> Path:
    # Prefer state-provided paths from pdf_text_tools; fall back to the
    # conventional PaperStore path for manual runs.
    if paper_id in state.paper_text_paths:
        return Path(state.paper_text_paths[paper_id])

    return file_store.clean_text_path(paper_id)


def _normalize_spaces(text: str) -> str:
    # Section detection works better when PDF line wrapping has been collapsed.
    return re.sub(r"\s+", " ", text).strip()


def _validate_chunk_settings(
    min_chunk_words: int,
    target_chunk_words: int,
    max_chunk_words: int,
    overlap_words: int,
) -> None:
    # Fail fast on invalid chunking settings so the tool does not silently
    # create pathological chunks.
    if min_chunk_words <= 0:
        raise ValueError("min_chunk_words must be positive.")
    if target_chunk_words <= 0:
        raise ValueError("target_chunk_words must be positive.")
    if max_chunk_words < target_chunk_words:
        raise ValueError("max_chunk_words must be >= target_chunk_words.")
    if target_chunk_words < min_chunk_words:
        raise ValueError("target_chunk_words must be >= min_chunk_words.")
    if overlap_words < 0:
        raise ValueError("overlap_words must be non-negative.")
    if overlap_words >= target_chunk_words:
        raise ValueError("overlap_words must be smaller than target_chunk_words.")
