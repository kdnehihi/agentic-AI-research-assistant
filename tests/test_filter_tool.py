#this file test if filter_relevant_papers.py works correctly
from app.tools.filter_relevant_papers import filter_relevant_papers
from app.agent.state import AgentState, Paper

def test_filter_relevant_papers():
    state = AgentState(topic="RLHF RLVR reasoning models", max_papers=3)
    #source and url is requirement and the score to filter
    papers = [
        Paper(title="Paper 1", abstract="This is a paper about RLHF.", source="fake_arxiv", url="http://fake_arxiv.org/paper1", score=0.0),
        Paper(
            title="Paper 2",
            abstract="This is a paper about RLVR.",
            source="fake_arxiv",
            url="http://fake_arxiv.org/paper2",
            score=0.6,
            score_components={"semantic": 0.3},
        ),
        Paper(title="Paper 3", abstract="This is a paper about reasoning models.", source="fake_arxiv", url="http://fake_arxiv.org/paper3", score=2.0),
        Paper(title="Paper 4", abstract="This is a paper about RLHF and RLVR reasoning.", source="fake_arxiv", url="http://fake_arxiv.org/paper4", score=3.0),
    ]
    state.candidate_papers = papers

    result = filter_relevant_papers(state=state)

    assert result["before"] == 4
    assert result["passed_threshold"] == 3
    assert result["after"] == 3
    assert len(state.selected_papers) == 3
    assert state.selected_papers[0].title == "Paper 4"
    assert state.selected_papers[1].title == "Paper 3"
    assert state.selected_papers[2].title == "Paper 2"


def test_filter_relevant_papers_rejects_low_score_without_component_signal():
    state = AgentState(topic="RAG for literature search", max_papers=3)
    papers = [
        Paper(
            title="Weak Candidate",
            source="fake_arxiv",
            url="http://fake_arxiv.org/weak",
            score=0.6,
            score_components={"bm25_lexical": 0.1, "semantic": 0.05},
        ),
        Paper(
            title="Strong Lexical Candidate",
            source="fake_arxiv",
            url="http://fake_arxiv.org/strong",
            score=0.6,
            score_components={"bm25_lexical": 0.45, "semantic": 0.05},
        ),
    ]
    state.set_candidate_papers(papers)

    result = filter_relevant_papers(state=state)

    assert result["passed_threshold"] == 1
    assert state.selected_papers[0].title == "Strong Lexical Candidate"


def test_filter_relevant_papers_caps_candidates_to_max_papers():
    state = AgentState(topic="RLHF RLVR reasoning models", max_papers=2)
    papers = [
        Paper(title="Candidate 1", source="fake_arxiv", url="http://fake_arxiv.org/candidate1", score=3.0),
        Paper(title="Candidate 2", source="fake_arxiv", url="http://fake_arxiv.org/candidate2", score=5.0),
        Paper(title="Candidate 3", source="fake_arxiv", url="http://fake_arxiv.org/candidate3", score=4.0),
    ]
    state.set_candidate_papers(papers)
    state.set_selected_papers(papers[:2])

    result = filter_relevant_papers(state=state)

    assert result["before"] == 3
    assert result["passed_threshold"] == 3
    assert result["after"] == 2
    assert len(state.selected_papers) == 2
    assert state.selected_papers[0].title == "Candidate 2"
    assert state.selected_papers[1].title == "Candidate 3"
