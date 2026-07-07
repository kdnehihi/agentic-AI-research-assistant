from app.agent.state import AgentState
from app.tools.fake_paper_tools import (
    search_fake_papers,
    deduplicate_papers,
    rank_papers,
    generate_report_from_abstracts,
)


def test_search_fake_papers():
    state = AgentState(topic="RLHF RLVR reasoning models", max_papers=3)

    result = search_fake_papers(
        state=state,
        query=state.topic,
        max_results=10,
    )

    assert result["status"] == "success"
    assert len(state.candidate_papers) > 0
    assert "fake_arxiv" in state.searched_sources


def test_deduplicate_papers():
    state = AgentState(topic="RLHF RLVR reasoning models", max_papers=3)

    search_fake_papers(
        state=state,
        query=state.topic,
        max_results=10,
    )

    before = len(state.candidate_papers)

    result = deduplicate_papers(state)

    after = len(state.candidate_papers)

    assert result["status"] == "success"
    assert after < before
    assert result["removed_duplicates"] == before - after


def test_rank_papers():
    state = AgentState(topic="RLHF RLVR reasoning models", max_papers=3)

    search_fake_papers(state=state, query=state.topic, max_results=10)
    deduplicate_papers(state)

    result = rank_papers(
        state=state,
        topic=state.topic,
        max_papers=state.max_papers,
    )

    assert result["status"] == "success"
    assert len(state.selected_papers) == 3
    assert state.selected_papers[0].score >= state.selected_papers[-1].score


def test_generate_report_from_abstracts():
    state = AgentState(topic="RLHF RLVR reasoning models", max_papers=3)

    search_fake_papers(state=state, query=state.topic, max_results=10)
    deduplicate_papers(state)
    rank_papers(
        state=state,
        topic=state.topic,
        max_papers=state.max_papers,
    )

    result = generate_report_from_abstracts(state)

    assert result["status"] == "success"
    assert state.report is not None
    assert "# Paper Research Report" in state.report
    assert "## Selected Papers" in state.report
