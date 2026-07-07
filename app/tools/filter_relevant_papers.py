# This file filters relevant papers, ignoring papers below a score threshold.

from app.agent.state import AgentState


def filter_relevant_papers(state: AgentState, min_score: float = 0.5) -> dict:
    """
    Filter the candidate papers in the AgentState based on a relevance score threshold.

    Args:
        state (AgentState): The current state of the agent containing candidate papers.
        min_score (float): The minimum score a paper must have to be considered relevant.

    Returns:
        dict: A dictionary containing the filtered papers and related information.
    """
    before = len(state.candidate_papers)
    filtered = [paper for paper in state.candidate_papers if paper.score >= min_score]
    state.set_selected_papers(filtered)
    return {
        "status": "success",
        "before": before,
        "after": len(filtered),
        "summary": f"Filtered selected papers from {before} to {len(filtered)} using min_score={min_score}.",
    }
