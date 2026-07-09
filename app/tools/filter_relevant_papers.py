# This file filters relevant papers, ignoring papers below a score threshold.

from app.agent.state import AgentState


def filter_relevant_papers(state: AgentState, min_score: float = 2.0) -> dict:
    """
    Filter the candidate papers in the AgentState based on a relevance score threshold.

    Args:
        state (AgentState): The current state of the agent containing candidate papers.
        min_score (float): The minimum score a paper must have to be considered relevant.

    Returns:
        dict: A dictionary containing the filtered papers and related information.
    """
    before = len(state.candidate_papers)
    filtered = [
        paper
        for paper in state.candidate_papers
        if paper.score >= min_score
    ]
    ranked_filtered = sorted(
        filtered,
        key=lambda paper: paper.score,
        reverse=True,
    )
    selected = ranked_filtered[:state.max_papers]
    state.set_selected_papers(selected)
    return {
        "status": "success",
        "before": before,
        "passed_threshold": len(filtered),
        "after": len(selected),
        "summary": (
            f"Filtered candidate papers from {before} to {len(selected)} "
            f"selected papers using min_score={min_score}; "
            f"{len(filtered)} passed the threshold."
        ),
    }
