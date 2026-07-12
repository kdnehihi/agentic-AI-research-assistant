import sys
from types import SimpleNamespace

from app.agent.state import AgentState, Paper
from app.storage.paper_store import PaperStore
from app.tools import pdf_text_tools
from app.tools.pdf_text_tools import (
    clean_pdf_text,
    extract_pdf_text_for_selected_papers,
    extract_text_from_pdf,
    remove_references_section,
)


def test_clean_pdf_text_normalizes_common_pdf_artifacts():
    raw_text = "retrieval-\naugmented\x00 generation\nkeeps\nflow\n\nNew paragraph"

    clean_text = clean_pdf_text(raw_text)

    assert clean_text == "retrieval-augmented generation keeps flow\n\nNew paragraph"


def test_remove_references_section_removes_reference_tail():
    text = "Introduction\n\nUseful body text.\n\nReferences\n\n[1] Noisy citation"

    cleaned = remove_references_section(text)

    assert cleaned == "Introduction\n\nUseful body text."


def test_extract_text_from_pdf_reads_pages_with_fake_fitz(tmp_path, monkeypatch):
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF fake")

    class FakePage:
        def __init__(self, text):
            self.text = text

        def get_text(self, mode):
            assert mode == "text"
            return self.text

    class FakeDoc:
        def __enter__(self):
            return [FakePage("Page one text"), FakePage("Page two text")]

        def __exit__(self, exc_type, exc_value, traceback):
            return None

    fake_fitz = SimpleNamespace(open=lambda path: FakeDoc())
    monkeypatch.setitem(sys.modules, "fitz", fake_fitz)

    text = extract_text_from_pdf(pdf_path)

    assert "[PAGE 1]" in text
    assert "Page one text" in text
    assert "[PAGE 2]" in text
    assert "Page two text" in text


def test_extract_pdf_text_for_selected_papers_saves_raw_and_clean_text(
    tmp_path,
    monkeypatch,
):
    store = PaperStore(
        db_path=tmp_path / "metadata" / "papers.sqlite3",
        papers_dir=tmp_path / "papers",
    )
    paper = Paper(
        paper_id="arxiv:2501.09136v4",
        title="Agentic RAG",
        source="arxiv",
        url="https://arxiv.org/abs/2501.09136v4",
    )
    state = AgentState(topic="agentic rag", max_papers=1)
    state.set_selected_papers([paper])
    store.pdf_path(paper.paper_id).write_bytes(b"%PDF fake")

    raw_text = ("Main content about agentic RAG. " * 80) + "\n\nReferences\n\n[1] citation"
    monkeypatch.setattr(pdf_text_tools, "extract_text_from_pdf", lambda path: raw_text)

    observation = extract_pdf_text_for_selected_papers(
        state=state,
        file_store=store,
    )

    raw_text_path = store.raw_text_path(paper.paper_id)
    clean_text_path = store.clean_text_path(paper.paper_id)

    assert observation["status"] == "success"
    assert observation["processed"] == 1
    assert observation["failed"] == 0
    assert raw_text_path.read_text(encoding="utf-8") == raw_text
    assert "References" not in clean_text_path.read_text(encoding="utf-8")
    assert state.paper_text_paths == {paper.paper_id: str(clean_text_path)}


def test_extract_pdf_text_for_selected_papers_falls_back_to_abstract(
    tmp_path,
    monkeypatch,
):
    store = PaperStore(
        db_path=tmp_path / "metadata" / "papers.sqlite3",
        papers_dir=tmp_path / "papers",
    )
    paper = Paper(
        paper_id="arxiv:short",
        title="Short PDF",
        source="arxiv",
        url="https://arxiv.org/abs/short",
        abstract="Fallback abstract text.",
    )
    state = AgentState(topic="fallback", max_papers=1)
    state.set_selected_papers([paper])
    store.pdf_path(paper.paper_id).write_bytes(b"%PDF fake")

    monkeypatch.setattr(pdf_text_tools, "extract_text_from_pdf", lambda path: "short")

    observation = extract_pdf_text_for_selected_papers(
        state=state,
        file_store=store,
    )

    clean_text = store.clean_text_path(paper.paper_id).read_text(encoding="utf-8")

    assert observation["status"] == "success"
    assert observation["fallback_abstract"] == 1
    assert clean_text == paper.abstract
