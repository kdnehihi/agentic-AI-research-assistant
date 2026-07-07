#this file test if filter_relevant_papers.py works correctly
from app.tools.filter_relevant_papers import filter_relevant_papers
from app.agent.state import AgentState, Paper

def test_filter_relevant_papers():
    state = AgentState(topic="RLHF RLVR reasoning models", max_papers=3)
    #source and url is requirement and the score to filter
    papers = [
        Paper(title="Paper 1", abstract="This is a paper about RLHF.", source="fake_arxiv", url="http://fake_arxiv.org/paper1", score=0.0),
        Paper(title="Paper 2", abstract="This is a paper about RLVR.", source="fake_arxiv", url="http://fake_arxiv.org/paper2", score=0.6),
        Paper(title="Paper 3", abstract="This is a paper about reasoning models.", source="fake_arxiv", url="http://fake_arxiv.org/paper3", score=0.9),
    ]
    state.candidate_papers = papers

    filter_relevant_papers(state=state)

    assert len(state.selected_papers) == 2
    assert state.selected_papers[0].title == "Paper 2"
    assert state.selected_papers[1].title == "Paper 3"
