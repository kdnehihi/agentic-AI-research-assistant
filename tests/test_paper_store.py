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
