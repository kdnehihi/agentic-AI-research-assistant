import json

from app.agent.state import AgentState, Paper
from app.tools.fetch_selected_papers import (
    _build_arxiv_pdf_url,
    _save_one_paper,
    fetch_selected_papers,
)


def test_build_arxiv_pdf_url_from_abs_url():
    pdf_url = _build_arxiv_pdf_url("http://arxiv.org/abs/2401.12345v1")

    assert pdf_url == "https://arxiv.org/pdf/2401.12345v1.pdf"


def test_fetch_selected_papers_saves_metadata_abstract_and_pdf(tmp_path):
    paper = Paper(
        paper_id="arxiv:2401.12345v1",
        title="RLHF for Reasoning Models",
        source="arxiv",
        url="http://arxiv.org/abs/2401.12345v1",
        abstract="This paper studies RLHF for reasoning models.",
    )
    state = AgentState(topic="RLHF", max_papers=1)
    state.set_selected_papers([paper])
    downloaded_urls = []

    def fake_downloader(url, timeout):
        downloaded_urls.append(url)
        return b"%PDF fake paper content", "application/pdf"

    observation = fetch_selected_papers(
        state=state,
        output_dir=tmp_path,
        downloader=fake_downloader,
    )

    assert observation["status"] == "success"
    assert observation["saved"] == 1
    assert downloaded_urls == ["https://arxiv.org/pdf/2401.12345v1.pdf"]

    paper_result = observation["papers"][0]
    metadata_path = tmp_path / paper_result["paper_dir"] / "metadata.json"
    abstract_path = tmp_path / paper_result["paper_dir"] / "abstract.txt"
    full_text_path = tmp_path / paper_result["paper_dir"] / "full_text.pdf"

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata["paper_id"] == "arxiv:2401.12345v1"
    assert metadata["full_text_url"] == "https://arxiv.org/pdf/2401.12345v1.pdf"
    assert metadata["full_text_path"] == str(full_text_path)
    assert paper.full_text_path == str(full_text_path)
    assert abstract_path.read_text(encoding="utf-8").strip() == paper.abstract
    assert full_text_path.read_bytes() == b"%PDF fake paper content"


def test_save_one_paper_calls_downloader_and_updates_full_text_path(tmp_path):
    paper = Paper(
        paper_id="arxiv:direct-save",
        title="Direct Save Paper",
        source="arxiv",
        url="https://arxiv.org/abs/2501.00001v2",
        abstract="This abstract should also be written.",
    )
    calls = []

    def fake_downloader(url, timeout):
        calls.append({"url": url, "timeout": timeout})
        return b"plain text full content", "text/plain"

    observation = _save_one_paper(
        paper=paper,
        paper_dir=tmp_path,
        timeout=12.5,
        downloader=fake_downloader,
    )

    full_text_path = tmp_path / "full_text.pdf"
    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))

    assert calls == [
        {
            "url": "https://arxiv.org/pdf/2501.00001v2.pdf",
            "timeout": 12.5,
        }
    ]
    assert observation["status"] == "success"
    assert observation["full_text_path"] == str(full_text_path)
    assert paper.full_text_path == str(full_text_path)
    assert metadata["full_text_path"] == str(full_text_path)
    assert full_text_path.read_bytes() == b"plain text full content"


def test_fetch_selected_papers_falls_back_to_abstract_when_download_fails(tmp_path):
    paper = Paper(
        paper_id="arxiv:broken",
        title="Broken Download",
        source="arxiv",
        url="http://arxiv.org/abs/broken",
        abstract="Fallback abstract summary.",
    )
    state = AgentState(topic="fallback", max_papers=1)
    state.set_selected_papers([paper])

    def failing_downloader(url, timeout):
        raise RuntimeError("network unavailable")

    observation = fetch_selected_papers(
        state=state,
        output_dir=tmp_path,
        downloader=failing_downloader,
    )

    assert observation["status"] == "partial_success"
    assert observation["saved"] == 1

    paper_result = observation["papers"][0]
    full_text_path = tmp_path / paper_result["paper_dir"] / "full_text.txt"

    assert paper_result["status"] == "partial_success"
    assert paper_result["error"] == "network unavailable"
    assert paper.full_text_path == str(full_text_path)
    assert full_text_path.read_text(encoding="utf-8").strip() == paper.abstract


def test_fetch_selected_papers_skips_when_no_selected_papers(tmp_path):
    state = AgentState(topic="empty", max_papers=1)

    observation = fetch_selected_papers(state=state, output_dir=tmp_path)

    assert observation["status"] == "skipped"
    assert observation["requested"] == 0
    assert observation["saved"] == 0
