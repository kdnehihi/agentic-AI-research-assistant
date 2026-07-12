from app.agent.state import Paper
from app.storage.paper_store import PaperStore


def test_paper_store_saves_and_tracks_seen_papers(tmp_path):
    store = PaperStore(db_path=tmp_path / "papers.sqlite3")
    paper = Paper(
        paper_id="arxiv:1234.5678v1",
        title="A Test Paper",
        authors=["Alice"],
        source="arxiv",
        url="https://arxiv.org/abs/1234.5678v1",
        abstract="This is a test abstract.",
        published_date="2026-07-01",
        score=3.5,
    )

    assert store.paper_exists(paper.paper_id) is False

    saved_count = store.save_papers(
        papers=[paper],
        topic="test topic",
        selected=True,
    )

    assert saved_count == 1
    assert store.paper_exists(paper.paper_id) is True
    assert store.get_seen_paper_ids() == {"arxiv:1234.5678v1"}
    assert store.get_all_paper_ids() == ["arxiv:1234.5678v1"]


def test_paper_store_removes_paper(tmp_path):
    store = PaperStore(db_path=tmp_path / "papers.sqlite3")
    paper = Paper(
        paper_id="arxiv:remove-me",
        title="Paper To Remove",
        source="arxiv",
        url="https://arxiv.org/abs/remove-me",
    )
    store.save_paper(paper, topic="test topic", selected=True)

    assert store.remove_paper("arxiv:remove-me") is True
    assert store.paper_exists("arxiv:remove-me") is False
    assert store.remove_paper("arxiv:missing") is False


def test_paper_store_builds_paper_file_paths_and_saves_text(tmp_path):
    store = PaperStore(
        db_path=tmp_path / "metadata" / "papers.sqlite3",
        papers_dir=tmp_path / "papers",
    )
    paper_id = "arxiv:2501.09136v4"

    paper_dir = store.paper_dir(paper_id)
    pdf_path = store.pdf_path(paper_id)
    raw_text_path = store.save_raw_text(paper_id, "Raw text")
    clean_text_path = store.save_clean_text(paper_id, "Clean text")

    assert paper_dir == tmp_path / "papers" / "arxiv_2501_09136v4"
    assert pdf_path == paper_dir / "paper.pdf"
    assert raw_text_path == paper_dir / "raw_text.txt"
    assert clean_text_path == paper_dir / "clean_text.txt"
    assert raw_text_path.read_text(encoding="utf-8") == "Raw text"
    assert clean_text_path.read_text(encoding="utf-8") == "Clean text"
