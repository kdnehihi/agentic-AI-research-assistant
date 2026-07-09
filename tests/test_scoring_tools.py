from app.agent.state import AgentState, Paper
from app.tools.scoring_tools import rank_papers_by_similarity


def test_rank_papers_by_similarity_selects_most_relevant_papers():
    state = AgentState(topic="RLHF RLVR reasoning models", max_papers=2)
    state.set_candidate_papers(
        [
            Paper(
                paper_id="paper:generic",
                title="Vision Dataset Survey",
                source="test",
                url="https://example.com/generic",
                abstract="This paper surveys image datasets.",
            ),
            Paper(
                paper_id="paper:rlhf",
                title="RLHF for Reasoning Models",
                source="test",
                url="https://example.com/rlhf",
                abstract="Reinforcement learning from human feedback improves reasoning models.",
            ),
            Paper(
                paper_id="paper:rlvr",
                title="RLVR and Verifiable Rewards",
                source="test",
                url="https://example.com/rlvr",
                abstract="Verifiable rewards improve mathematical reasoning in language models.",
            ),
        ]
    )

    observation = rank_papers_by_similarity(state)

    assert observation["status"] == "success"
    assert observation["selected"] == 2
    assert len(state.selected_papers) == 2
    assert {paper.paper_id for paper in state.selected_papers} == {
        "paper:rlhf",
        "paper:rlvr",
    }
    assert state.selected_papers[0].score >= state.selected_papers[1].score
    assert state.selected_papers[0].relevant_reasons


def test_rank_papers_by_similarity_handles_empty_candidates():
    state = AgentState(topic="RLHF", max_papers=2)

    observation = rank_papers_by_similarity(state)

    assert observation["status"] == "partial_success"
    assert observation["selected"] == 0
    assert state.selected_papers == []


def test_rank_papers_by_similarity_hard_gates_core_rl_topics():
    state = AgentState(topic="RLHF RLVR reasoning models", max_papers=2)
    state.set_candidate_papers(
        [
            Paper(
                paper_id="paper:reasoning-only",
                title="CAT: Confidence-Adaptive Thinking for Reasoning Models",
                source="test",
                url="https://example.com/reasoning-only",
                abstract=(
                    "This paper studies reasoning models and adaptive thinking "
                    "without reinforcement learning from feedback or rewards."
                ),
                published_date="2026-07-01",
            ),
            Paper(
                paper_id="paper:rlhf",
                title="RLHF for Reasoning Models",
                source="test",
                url="https://example.com/rlhf",
                abstract=(
                    "Reinforcement learning from human feedback improves "
                    "reasoning models."
                ),
                published_date="2026-06-01",
            ),
            Paper(
                paper_id="paper:rlvr",
                title="RLVR and Verifiable Rewards",
                source="test",
                url="https://example.com/rlvr",
                abstract=(
                    "Verifiable rewards improve mathematical reasoning in "
                    "language models."
                ),
                published_date="2026-06-01",
            ),
        ]
    )

    observation = rank_papers_by_similarity(state)

    blocked_paper = next(
        paper
        for paper in state.candidate_papers
        if paper.paper_id == "paper:reasoning-only"
    )
    selected_ids = {paper.paper_id for paper in state.selected_papers}

    assert observation["hard_gate_enabled"] is True
    assert observation["blocked_by_hard_gate"] == 1
    assert blocked_paper.score == 0.0
    assert "Blocked by hard gate" in blocked_paper.relevant_reasons[0]
    assert selected_ids == {"paper:rlhf", "paper:rlvr"}


def test_rank_papers_by_similarity_soft_gates_rag_context_and_title_phrases():
    state = AgentState(
        topic=(
            "agentic retrieval augmented generation systems for scientific "
            "literature search and research paper summarization"
        ),
        max_papers=2,
    )
    state.set_candidate_papers(
        [
            Paper(
                paper_id="paper:generic-rag",
                title="Retrieval-Augmented Generation",
                source="test",
                url="https://example.com/generic-rag",
                abstract="This paper studies retrieval augmented generation.",
            ),
            Paper(
                paper_id="paper:agentic-rag",
                title="Agentic Retrieval-Augmented Generation for Scientific Literature",
                source="test",
                url="https://example.com/agentic-rag",
                abstract=(
                    "This paper studies RAG agents for scientific literature "
                    "search and research paper summarization."
                ),
            ),
            Paper(
                paper_id="paper:unrelated",
                title="Vision-Language Reasoning Models",
                source="test",
                url="https://example.com/unrelated",
                abstract="This paper studies visual reasoning.",
            ),
        ]
    )

    observation = rank_papers_by_similarity(state)

    generic_rag = next(
        paper
        for paper in state.candidate_papers
        if paper.paper_id == "paper:generic-rag"
    )
    unrelated = next(
        paper
        for paper in state.candidate_papers
        if paper.paper_id == "paper:unrelated"
    )

    assert observation["hard_gate_enabled"] is True
    assert unrelated.score == 0.0
    assert generic_rag.score_components["context_match"] == 0.0
    assert generic_rag.score_components["title_exact_match"] > 0.0
    assert state.selected_papers[0].paper_id == "paper:agentic-rag"
    assert state.selected_papers[0].score_components["context_match"] == 1.0
