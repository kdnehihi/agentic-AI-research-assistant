from app.agent.state import AgentState
from app.tools.arxiv_tools import search_arxiv_papers
from app.tools.scoring_tools import rank_papers_by_similarity


TOPIC = "rag"
MAX_CANDIDATES = 20


def main():
    state = AgentState(topic=TOPIC, max_papers=MAX_CANDIDATES)

    search_observation = search_arxiv_papers(
        state=state,
        query=state.topic,
        max_results=MAX_CANDIDATES,
    )
    rank_papers_by_similarity(
        state=state,
        query=state.topic,
        max_papers=MAX_CANDIDATES,
        title_weight=0.3,
        abstract_weight=0.7,
    )

    print(f"Topic: {state.topic}")
    print(f"arXiv Query: {search_observation.get('search_query')}")
    print("-" * 80)

    for paper in state.candidate_papers:
        print(f"Title: {paper.title}")
        print(f"Score: {paper.score}")
        print(f"Relevant Reasons: {paper.relevant_reasons}")
        print(f"Abstract Preview: {(paper.abstract or '')[:100]}...")
        print("-" * 80)


if __name__ == "__main__":
    main()
