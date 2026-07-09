from app.agent.state import AgentState, Paper
from app.tools.fake_paper_tools import (
    search_fake_papers,
    deduplicate_papers,
    rank_papers,
    generate_fake_report,
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


def test_rank_papers_prioritizes_rl_terms_over_generic_reasoning():
    state = AgentState(topic="RLHF RLVR reasoning models", max_papers=2)
    state.set_candidate_papers(
        [
            Paper(
                title="Generic Reasoning Model",
                abstract="This paper studies reasoning in models.",
                source="test",
                url="https://example.com/generic",
            ),
            Paper(
                title="RLHF for Reasoning Models",
                abstract="This paper studies reinforcement learning from human feedback for reasoning.",
                source="test",
                url="https://example.com/rlhf",
            ),
            Paper(
                title="RLVR and Verifiable Rewards",
                abstract="This paper studies verifiable rewards for reasoning.",
                source="test",
                url="https://example.com/rlvr",
            ),
        ]
    )

    rank_papers(state)

    assert state.selected_papers[0].title in {
        "RLHF for Reasoning Models",
        "RLVR and Verifiable Rewards",
    }
    assert state.selected_papers[-1].score > state.candidate_papers[0].score


def test_generate_fake_report():
    state = AgentState(topic="RLHF RLVR reasoning models", max_papers=3)

    search_fake_papers(state=state, query=state.topic, max_results=10)
    deduplicate_papers(state)
    rank_papers(
        state=state,
        topic=state.topic,
        max_papers=state.max_papers,
    )

    result = generate_fake_report(state)

    assert result["status"] == "success"
    assert state.report is not None
    assert "# Paper Research Report" in state.report
    assert "## Selected Papers" in state.report
