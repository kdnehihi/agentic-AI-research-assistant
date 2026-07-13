from app.agent.state import AgentState, Paper
from app.storage.paper_store import PaperStore
from app.tools.production.knowledge_base_tools import (
    get_paper_metadata,
    list_papers,
    save_papers_to_kb,
)


def test_save_papers_to_kb_uses_explicit_ids_and_is_idempotent(tmp_path):
    store = PaperStore(db_path=tmp_path / "papers.sqlite3", papers_dir=tmp_path / "papers")
    paper = Paper(
        paper_id="paper:1",
        title="Paper One",
        source="manual",
        url="https://example.com/1",
    )
    state = AgentState(topic="kb", max_papers=1)
    state.set_candidate_papers([paper])

    first = save_papers_to_kb(state, paper_ids=["paper:1"], store=store)
    second = save_papers_to_kb(state, paper_ids=["paper:1"], store=store)

    assert first["inserted_paper_ids"] == ["paper:1"]
    assert second["inserted_paper_ids"] == []
    assert second["already_present_paper_ids"] == ["paper:1"]
    assert store.get_all_paper_ids() == ["paper:1"]


def test_list_papers_is_read_only_and_distinguishes_dates(tmp_path):
    store = PaperStore(db_path=tmp_path / "papers.sqlite3", papers_dir=tmp_path / "papers")
    paper = Paper(
        paper_id="paper:2",
        title="Paper Two",
        source="manual",
        url="https://example.com/2",
        published_date="2026-01-02",
    )
    store.save_paper(paper, topic="kb", selected=True)
    state = AgentState(topic="kb", max_papers=1)

    observation = list_papers(state, store=store)

    assert observation["count"] == 1
    assert observation["papers"][0]["published_date"] == "2026-01-02"
    assert observation["papers"][0]["added_date"] is not None
    assert state.candidate_papers == []


def test_get_paper_metadata_reports_artifact_readiness(tmp_path):
    store = PaperStore(db_path=tmp_path / "papers.sqlite3", papers_dir=tmp_path / "papers")
    paper = Paper(
        paper_id="paper:3",
        title="Paper Three",
        source="manual",
        url="https://example.com/3",
    )
    store.save_paper(paper, topic="kb", selected=True)
    store.save_clean_text("paper:3", "clean text")

    observation = get_paper_metadata(
        AgentState(topic="kb", max_papers=1),
        paper_ids=["paper:3"],
        store=store,
        vector_store=None,
    )

    assert observation["status"] == "success"
    assert observation["papers"][0]["exists_in_kb"] is True
    assert observation["papers"][0]["clean_text_exists"] is True
    assert observation["papers"][0]["indexed"] is False
