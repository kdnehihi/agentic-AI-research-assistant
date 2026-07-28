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
    assert store.get_saved_paper_ids() == {"arxiv:1234.5678v1"}
    assert store.get_all_paper_ids() == ["arxiv:1234.5678v1"]


def test_paper_store_distinguishes_seen_metadata_from_saved_papers(tmp_path):
    store = PaperStore(db_path=tmp_path / "papers.sqlite3")
    metadata_only = Paper(
        paper_id="arxiv:metadata",
        title="Metadata Only",
        source="arxiv",
        url="https://arxiv.org/abs/metadata",
    )
    saved = Paper(
        paper_id="arxiv:saved",
        title="Saved Paper",
        source="arxiv",
        url="https://arxiv.org/abs/saved",
    )

    store.save_paper(metadata_only, topic="displayed result", selected=False)
    store.save_paper(saved, topic="workspace", selected=True)

    assert store.get_seen_paper_ids() == {"arxiv:metadata", "arxiv:saved"}
    assert store.get_saved_paper_ids() == {"arxiv:saved"}


def test_paper_store_preserves_multi_source_metadata(tmp_path):
    store = PaperStore(db_path=tmp_path / "papers.sqlite3")
    paper = Paper(
        paper_id="arxiv:2603.07379",
        title="SoK: Agentic Retrieval-Augmented Generation",
        authors=["Saroj Mishra"],
        source="arxiv",
        url="https://arxiv.org/abs/2603.07379",
        doi="10.1234/example",
        arxiv_id="2603.07379",
        semantic_scholar_id="S2ID",
        external_ids={"ArXiv": "2603.07379", "DOI": "10.1234/example"},
        provenance=[
            {"source": "arxiv", "source_paper_id": "2603.07379"},
            {"source": "semantic_scholar", "source_paper_id": "S2ID"},
        ],
        venue="arXiv",
        citation_count=12,
        open_access_pdf_url="https://arxiv.org/pdf/2603.07379",
    )

    store.save_paper(paper, topic="agentic rag")

    stored_paper = store.get_paper("arxiv:2603.07379")
    stored_record = store.get_paper_record("arxiv:2603.07379")

    assert stored_paper is not None
    assert stored_paper.doi == "10.1234/example"
    assert stored_paper.external_ids["ArXiv"] == "2603.07379"
    assert stored_paper.provenance[1]["source"] == "semantic_scholar"
    assert stored_paper.citation_count == 12
    assert stored_record is not None
    assert stored_record["semantic_scholar_id"] == "S2ID"
    assert stored_record["open_access_pdf_url"] == "https://arxiv.org/pdf/2603.07379"


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
