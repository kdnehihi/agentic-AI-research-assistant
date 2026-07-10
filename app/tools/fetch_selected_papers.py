from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from app.agent.state import AgentState, Paper
from app.tools.knowledge_base_tools import ensure_paper_id


DownloadFn = Callable[[str, float], tuple[bytes, str]]


def fetch_selected_papers(
    state: AgentState,
    output_dir: str | Path = "data/papers",
    timeout: float = 30.0,
    downloader: DownloadFn | None = None,
) -> dict[str, Any]:
    """
    Persist selected papers to disk.

    Each selected paper gets its own folder containing metadata.json,
    abstract.txt when available, and a best-effort full_text file.
    For arXiv papers, the tool downloads the PDF from the paper URL.
    """
    if not state.selected_papers:
        return {
            "status": "skipped",
            "requested": 0,
            "saved": 0,
            "failed": 0,
            "papers": [],
            "summary": "No selected papers to fetch.",
        }

    papers_root = Path(output_dir)
    papers_root.mkdir(parents=True, exist_ok=True)

    downloader = downloader or _download_url

    results = []
    saved_count = 0
    failed_count = 0

    for paper in state.selected_papers:
        ensure_paper_id(paper)
        paper_dir = papers_root / _build_paper_dir_name(paper)
        paper_dir.mkdir(parents=True, exist_ok=True)

        result = _save_one_paper(
            paper=paper,
            paper_dir=paper_dir,
            timeout=timeout,
            downloader=downloader,
        )
        results.append(result)

        if result["status"] in {"success", "partial_success"}:
            saved_count += 1
        else:
            failed_count += 1

    if failed_count == 0 and all(result["status"] == "success" for result in results):
        status = "success"
    elif saved_count > 0:
        status = "partial_success"
    else:
        status = "failed"

    return {
        "status": status,
        "requested": len(state.selected_papers),
        "saved": saved_count,
        "failed": failed_count,
        "output_dir": str(papers_root),
        "papers": results,
        "summary": (
            f"Saved files for {saved_count}/{len(state.selected_papers)} "
            "selected papers."
        ),
    }


def _save_one_paper(
    paper: Paper,
    paper_dir: Path,
    timeout: float,
    downloader: DownloadFn,
) -> dict[str, Any]:
    metadata_path = paper_dir / "metadata.json"
    abstract_path = paper_dir / "abstract.txt"
    full_text_url = _build_full_text_url(paper)

    _write_metadata(
        paper=paper,
        metadata_path=metadata_path,
        paper_dir=paper_dir,
        full_text_url=full_text_url,
    )

    if paper.abstract:
        abstract_path.write_text(paper.abstract.strip() + "\n", encoding="utf-8")

    if not full_text_url:
        fallback_path = _write_abstract_fallback(paper, paper_dir)
        if fallback_path:
            paper.full_text_path = str(fallback_path)
            _write_metadata(
                paper=paper,
                metadata_path=metadata_path,
                paper_dir=paper_dir,
                full_text_url=full_text_url,
            )
        return {
            "status": "partial_success",
            "paper_id": paper.paper_id,
            "title": paper.title,
            "paper_dir": str(paper_dir),
            "metadata_path": str(metadata_path),
            "full_text_path": str(fallback_path) if fallback_path else None,
            "error": "No downloadable full text URL was found.",
        }

    try:
        content, content_type = downloader(full_text_url, timeout)
        full_text_path = _write_full_text_file(
            paper_dir=paper_dir,
            content=content,
            content_type=content_type,
            source_url=full_text_url,
        )
    except Exception as exc:  # pragma: no cover - exact network errors vary.
        fallback_path = _write_abstract_fallback(paper, paper_dir)
        if fallback_path:
            paper.full_text_path = str(fallback_path)
            _write_metadata(
                paper=paper,
                metadata_path=metadata_path,
                paper_dir=paper_dir,
                full_text_url=full_text_url,
            )
        return {
            "status": "partial_success" if fallback_path else "failed",
            "paper_id": paper.paper_id,
            "title": paper.title,
            "paper_dir": str(paper_dir),
            "metadata_path": str(metadata_path),
            "full_text_path": str(fallback_path) if fallback_path else None,
            "error": str(exc),
        }

    paper.full_text_path = str(full_text_path)
    _write_metadata(
        paper=paper,
        metadata_path=metadata_path,
        paper_dir=paper_dir,
        full_text_url=full_text_url,
    )

    return {
        "status": "success",
        "paper_id": paper.paper_id,
        "title": paper.title,
        "paper_dir": str(paper_dir),
        "metadata_path": str(metadata_path),
        "full_text_path": str(full_text_path),
        "full_text_url": full_text_url,
    }


def _write_metadata(
    paper: Paper,
    metadata_path: Path,
    paper_dir: Path,
    full_text_url: str | None,
) -> None:
    metadata = paper.model_dump(mode="json")
    metadata["local_dir"] = str(paper_dir)
    metadata["full_text_url"] = full_text_url

    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _download_url(url: str, timeout: float) -> tuple[bytes, str]:
    request = Request(
        url,
        headers={"User-Agent": "agentic-ai-research-assistant/0.1"},
    )
    with urlopen(request, timeout=timeout) as response:
        content_type = response.headers.get("Content-Type", "")
        return response.read(), content_type


def _build_full_text_url(paper: Paper) -> str | None:
    if "arxiv.org" in paper.url:
        return _build_arxiv_pdf_url(paper.url)

    if paper.url.lower().endswith(".pdf"):
        return paper.url

    return None


def _build_arxiv_pdf_url(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")

    if "/pdf/" in path:
        paper_id = path.rsplit("/", maxsplit=1)[-1]
    elif "/abs/" in path:
        paper_id = path.rsplit("/", maxsplit=1)[-1]
    else:
        paper_id = path.rsplit("/", maxsplit=1)[-1]

    if not paper_id.endswith(".pdf"):
        paper_id = f"{paper_id}.pdf"

    return f"https://arxiv.org/pdf/{paper_id}"


def _write_full_text_file(
    paper_dir: Path,
    content: bytes,
    content_type: str,
    source_url: str,
) -> Path:
    is_pdf = (
        "pdf" in content_type.lower()
        or source_url.lower().endswith(".pdf")
        or content.startswith(b"%PDF")
    )

    if is_pdf:
        full_text_path = paper_dir / "full_text.pdf"
        full_text_path.write_bytes(content)
        return full_text_path

    full_text_path = paper_dir / "full_text.txt"
    full_text_path.write_text(
        content.decode("utf-8", errors="replace"),
        encoding="utf-8",
    )
    return full_text_path


def _write_abstract_fallback(paper: Paper, paper_dir: Path) -> Path | None:
    if not paper.abstract:
        return None

    full_text_path = paper_dir / "full_text.txt"
    full_text_path.write_text(paper.abstract.strip() + "\n", encoding="utf-8")
    return full_text_path


def _build_paper_dir_name(paper: Paper) -> str:
    title_slug = _slugify(paper.title)
    id_slug = _slugify(paper.paper_id or paper.title)

    if id_slug and id_slug not in title_slug:
        return f"{id_slug}_{title_slug}"[:120].rstrip("_")

    return title_slug[:120].rstrip("_")


def _slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "untitled_paper"
