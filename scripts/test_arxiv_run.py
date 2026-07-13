from app.agent.state import AgentState
from app.tools.arxiv_tools import search_arxiv_papers

def main():
    """Smoke-test the arXiv search tool with a small fixed query."""

    state = AgentState(topic="RLHF RLVR reasoning models", max_papers=3)

    # Use the search_arxiv_papers tool to search for papers
    observation = search_arxiv_papers(
        state=state,
        query=state.topic,
        max_results=2,
    )

    print("Observation:", observation)
    print("Candidate Papers:", state.candidate_papers)
    print("Searched Sources:", state.searched_sources)
    print("Report:", state.report)

if __name__ == "__main__":
    main()
