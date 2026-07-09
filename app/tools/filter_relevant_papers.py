# This file filters relevant papers, ignoring papers below a score threshold.

from app.agent.state import AgentState


def filter_relevant_papers(
    state: AgentState,
    min_score: float = 2.0,
    min_lexical_score: float = 0.4,
    min_semantic_score: float = 0.2,
) -> dict:
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
        if _passes_relevance_filter(
            paper=paper,
            min_score=min_score,
            min_lexical_score=min_lexical_score,
            min_semantic_score=min_semantic_score,
        )
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
            f"selected papers using min_score={min_score}, "
            f"min_lexical_score={min_lexical_score}, "
            f"min_semantic_score={min_semantic_score}; "
            f"{len(filtered)} passed the threshold."
        ),
    }


def _passes_relevance_filter(
    paper,
    min_score: float,
    min_lexical_score: float,
    min_semantic_score: float,
) -> bool:
    components = paper.score_components or {}
    lexical_score = components.get("bm25_lexical", 0.0)
    semantic_score = components.get("semantic", 0.0)
    title_match_score = components.get("title_exact_match", 0.0)
    context_match_score = components.get("context_match", 0.0)

    if paper.score >= min_score:
        return True

    return (
        lexical_score >= min_lexical_score
        or semantic_score >= min_semantic_score
        or title_match_score > 0
        or context_match_score > 0
    )
