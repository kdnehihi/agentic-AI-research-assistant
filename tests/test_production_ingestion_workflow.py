from app.agent.state import AgentState, Paper
from app.storage.paper_store import PaperStore
from app.workflows import paper_ingestion
from app.workflows.paper_ingestion import ensure_papers_retrievable_workflow


class FakeVectorStore:
    def __init__(self, indexed=None):
        self.indexed = set(indexed or [])

    def get_by_paper(self, paper_id):
        return [object()] if paper_id in self.indexed else []


def test_ensure_papers_retrievable_skips_already_indexed(tmp_path, monkeypatch):
    store = PaperStore(db_path=tmp_path / "papers.sqlite3", papers_dir=tmp_path / "papers")
    paper = Paper(paper_id="paper:ready", title="Ready", source="manual", url="https://x")
    store.save_paper(paper, topic="kb", selected=True)
    state = AgentState(topic="kb", max_papers=1)

    monkeypatch.setattr(paper_ingestion, "fetch_selected_papers", lambda **kwargs: (_ for _ in ()).throw(AssertionError("fetch should not run")))

    observation = ensure_papers_retrievable_workflow(
        state,
        paper_ids=["paper:ready"],
        store=store,
        vector_store=FakeVectorStore(indexed={"paper:ready"}),
    )

    assert observation["status"] == "success"
    assert observation["already_ready_paper_ids"] == ["paper:ready"]
    assert observation["newly_indexed_paper_ids"] == []


def test_ensure_papers_retrievable_reports_missing_metadata(tmp_path):
    store = PaperStore(db_path=tmp_path / "papers.sqlite3", papers_dir=tmp_path / "papers")
    state = AgentState(topic="kb", max_papers=1)

    observation = ensure_papers_retrievable_workflow(
        state,
        paper_ids=["missing"],
        store=store,
        vector_store=FakeVectorStore(),
    )

    assert observation["status"] == "failed"
    assert observation["failed"][0]["stage"] == "metadata"


def test_ensure_papers_retrievable_runs_missing_stages_in_order(tmp_path, monkeypatch):
    store = PaperStore(db_path=tmp_path / "papers.sqlite3", papers_dir=tmp_path / "papers")
    paper = Paper(paper_id="paper:new", title="New", source="manual", url="https://x")
    state = AgentState(topic="kb", max_papers=1)
    state.set_candidate_papers([paper])
    calls = []

    def fake_fetch(state):
        calls.append("fetch")
        store.pdf_path("paper:new").write_bytes(b"%PDF fake")
        return {"status": "success"}

    def fake_extract(state, file_store):
        calls.append("extract")
        file_store.save_clean_text("paper:new", "clean text")
        state.set_paper_text_paths({"paper:new": str(file_store.clean_text_path("paper:new"))})
        return {"status": "success"}

    def fake_chunk(state, file_store):
        calls.append("chunk")
        file_store.chunks_path("paper:new").write_text('{"chunk_id":"c1","paper_id":"paper:new","text":"x"}\n')
        state.set_paper_chunk_paths({"paper:new": str(file_store.chunks_path("paper:new"))})
        return {"status": "success"}

    def fake_embed(state, file_store):
        calls.append("embed")
        file_store.embeddings_path("paper:new").write_text("{}\n")
        return {"status": "success"}

    def fake_index(state, vector_store=None):
        calls.append("index")
        return {"status": "success"}

    monkeypatch.setattr(paper_ingestion, "fetch_selected_papers", fake_fetch)
    monkeypatch.setattr(paper_ingestion, "extract_pdf_text_for_selected_papers", fake_extract)
    monkeypatch.setattr(paper_ingestion, "chunk_selected_papers_by_section", fake_chunk)
    monkeypatch.setattr(paper_ingestion, "embed_selected_paper_chunks", fake_embed)
    monkeypatch.setattr(paper_ingestion, "index_selected_paper_chunks", fake_index)

    observation = ensure_papers_retrievable_workflow(
        state,
        paper_ids=["paper:new"],
        store=store,
        vector_store=FakeVectorStore(),
    )

    assert calls == ["fetch", "extract", "chunk", "embed", "index"]
    assert observation["ready_paper_ids"] == ["paper:new"]
    assert observation["newly_indexed_paper_ids"] == ["paper:new"]
