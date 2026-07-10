from app.agent.state import AgentState, Paper
from app.storage.paper_store import PaperStore
from app.tools.knowledge_base_tools import (
    filter_seen_papers,
    remove_papers_from_kb,
    save_candidate_papers_to_kb,
    save_selected_papers_to_kb,
)


def test_filter_seen_papers_removes_papers_already_in_store(tmp_path):
    store = PaperStore(db_path=tmp_path / "papers.sqlite3")
    seen_paper = Paper(
        paper_id="arxiv:seen",
        title="Seen Paper",
        source="arxiv",
        url="https://arxiv.org/abs/seen",
    )
    new_paper = Paper(
        paper_id="arxiv:new",
        title="New Paper",
        source="arxiv",
        url="https://arxiv.org/abs/new",
    )
    store.save_paper(seen_paper, topic="old topic", selected=True)

    state = AgentState(topic="new topic", max_papers=2)
    state.set_candidate_papers([seen_paper, new_paper])

    observation = filter_seen_papers(state=state, store=store)

    assert observation["removed_seen"] == 1
    assert [paper.paper_id for paper in state.candidate_papers] == ["arxiv:new"]


def test_save_candidate_and_selected_papers_to_kb(tmp_path):
    store = PaperStore(db_path=tmp_path / "papers.sqlite3")
    candidate = Paper(
        paper_id=None,
        title="Candidate Paper",
        source="manual",
        url="https://example.com/candidate",
    )
    selected = Paper(
        paper_id="manual:selected",
        title="Selected Paper",
        source="manual",
        url="https://example.com/selected",
    )
    state = AgentState(topic="knowledge base test", max_papers=1)
    state.set_candidate_papers([candidate])
    state.set_selected_papers([selected])

    candidate_observation = save_candidate_papers_to_kb(state=state, store=store)
    selected_observation = save_selected_papers_to_kb(state=state, store=store)

    assert candidate_observation["saved"] == 1
    assert selected_observation["saved"] == 1
    assert candidate.paper_id == "manual:candidate"
    assert store.get_seen_paper_ids() == {"manual:candidate", "manual:selected"}


def test_remove_papers_from_kb_by_ids(tmp_path):
    store = PaperStore(db_path=tmp_path / "papers.sqlite3")
    paper = Paper(
        paper_id="manual:remove",
        title="Remove Me",
        source="manual",
        url="https://example.com/remove",
    )
    store.save_paper(paper, topic="knowledge base test", selected=True)
    state = AgentState(topic="knowledge base test", max_papers=1)

    observation = remove_papers_from_kb(
        state=state,
        paper_ids=["manual:remove", "manual:missing"],
        store=store,
    )

    assert observation["status"] == "success"
    assert observation["requested"] == 2
    assert observation["removed"] == 1
    assert observation["missing"] == 1
    assert store.paper_exists("manual:remove") is False


def test_remove_papers_from_kb_defaults_to_selected_papers(tmp_path):
    store = PaperStore(db_path=tmp_path / "papers.sqlite3")
    paper = Paper(
        paper_id="manual:selected-remove",
        title="Selected Remove",
        source="manual",
        url="https://example.com/selected-remove",
    )
    store.save_paper(paper, topic="knowledge base test", selected=True)
    state = AgentState(topic="knowledge base test", max_papers=1)
    state.set_selected_papers([paper])

    observation = remove_papers_from_kb(state=state, store=store)

    assert observation["removed"] == 1
    assert store.paper_exists("manual:selected-remove") is False
