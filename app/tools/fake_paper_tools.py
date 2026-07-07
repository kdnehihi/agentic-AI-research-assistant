from app.agent.state import AgentState, Paper


def search_fake_papers(
    state: AgentState,
    query: str | None = None,
    max_results: int = 10,
) -> dict:
    """
    Fake paper search tool.

    Purpose:
    - Test whether tools can update AgentState.
    - Avoid using real arXiv API too early.
    """
    query = query or state.topic
    max_results = max_results if max_results is not None else state.max_papers
    papers = [
        Paper(
            paper_id="fake:001",
            title="RLHF for Reasoning-Centric Language Models",
            authors=["Alice Nguyen", "Bob Chen"],
            abstract=(
                "This paper studies reinforcement learning from human feedback "
                "for improving reasoning behavior in language models."
            ),
            source="fake_arxiv",
            url="https://arxiv.org/abs/fake-001",
            published_date="2026-06-20",
        ),
        Paper(
            paper_id="fake:002",
            title="RLVR and Verifiable Rewards for Mathematical Reasoning",
            authors=["Carlos Smith"],
            abstract=(
                "This work explores reinforcement learning with verifiable rewards "
                "for mathematical reasoning tasks."
            ),
            source="fake_arxiv",
            url="https://arxiv.org/abs/fake-002",
            published_date="2026-06-12",
        ),
        Paper(
            paper_id="fake:003",
            title="Preference Optimization for Long Chain-of-Thought Models",
            authors=["Diana Lee", "Eva Tran"],
            abstract=(
                "The paper proposes a preference optimization method for long "
                "chain-of-thought reasoning models."
            ),
            source="fake_arxiv",
            url="https://arxiv.org/abs/fake-003",
            published_date="2026-05-30",
        ),
        Paper(
            paper_id="fake:001",
            title="RLHF for Reasoning-Centric Language Models",
            authors=["Alice Nguyen", "Bob Chen"],
            abstract="Duplicate record of the same RLHF reasoning paper.",
            source="fake_arxiv",
            url="https://arxiv.org/abs/fake-001",
            published_date="2026-06-20",
        ),
        Paper(
            paper_id="fake:004",
            title="A Survey of Vision-Language Models",
            authors=["Frank Wilson"],
            abstract="This survey reviews recent progress in vision-language models.",
            source="fake_arxiv",
            url="https://arxiv.org/abs/fake-004",
            published_date="2026-04-11",
        ),
    ]

    selected = papers[:max_results]

    state.set_candidate_papers(selected)
    state.add_searched_source("fake_arxiv")

    return {
        "status": "success",
        "num_results": len(selected),
        "summary": f"Found {len(selected)} fake papers for query: {query}",
    }


def deduplicate_papers(state: AgentState) -> dict:
    """
    Remove duplicate papers by paper_id first, then by normalized title.
    """

    seen_keys = set()
    unique_papers = []

    for paper in state.candidate_papers:
        if paper.paper_id:
            key = paper.paper_id.lower().strip()
        else:
            key = paper.title.lower().strip()

        if key not in seen_keys:
            seen_keys.add(key)
            unique_papers.append(paper)

    removed = len(state.candidate_papers) - len(unique_papers)

    state.set_candidate_papers(unique_papers)

    return {
        "status": "success",
        "removed_duplicates": removed,
        "remaining": len(unique_papers),
        "summary": f"Removed {removed} duplicate papers.",
    }


def rank_papers(
    state: AgentState,
    topic: str | None = None,
    max_papers: int | None = None,
) -> dict:
    """
    Simple keyword-based ranker.

    Later this can be replaced by:
    - embedding similarity
    - LLM relevance judge
    - hybrid scoring
    """

    keywords = [
        "rlhf",
        "rlvr",
        "preference",
        "reasoning",
        "reward",
        "verifiable",
        "chain-of-thought",
    ]

    topic_text = topic.lower() if topic else state.topic.lower()
    if max_papers is None:
        max_papers = state.max_papers

    for paper in state.candidate_papers:
        paper_text = f"{paper.title} {paper.abstract or ''}".lower()

        keyword_score = sum(1 for kw in keywords if kw in paper_text)
        topic_score = sum(1 for word in topic_text.split() if word in paper_text)

        paper.score = float(keyword_score + topic_score)

        reasons = []
        if "rlhf" in paper_text:
            reasons.append("Mentions RLHF")
        if "rlvr" in paper_text or "verifiable" in paper_text:
            reasons.append("Mentions RLVR or verifiable rewards")
        if "reasoning" in paper_text:
            reasons.append("Focuses on reasoning")

        paper.relevant_reasons = reasons

    ranked = sorted(
        state.candidate_papers,
        key=lambda paper: paper.score,
        reverse=True,
    )

    selected = ranked[:max_papers]
    state.set_selected_papers(selected)

    return {
        "status": "success",
        "selected": len(selected),
        "summary": f"Selected top {len(selected)} papers.",
    }


def generate_fake_report(state: AgentState) -> dict:
    """
    Generate a simple markdown report without LLM.

    Later this should move to reports/writer.py or use an LLM report writer.
    """

    if not state.selected_papers:
        state.set_report("No relevant papers were selected.")

        return {
            "status": "partial_success",
            "summary": "No selected papers available for report.",
        }

    lines = [
        f"# Paper Research Report: {state.topic}",
        "",
        "## Selected Papers",
        "",
    ]

    for idx, paper in enumerate(state.selected_papers, start=1):
        lines.extend(
            [
                f"### {idx}. {paper.title}",
                f"- Authors: {', '.join(paper.authors)}",
                f"- Source: {paper.source}",
                f"- URL: {paper.url}",
                f"- Published Date: {paper.published_date or 'Unknown'}",
                f"- Score: {paper.score}",
                f"- Relevant Reasons: {', '.join(paper.relevant_reasons) or 'N/A'}",
                f"- Summary: {paper.abstract or 'No abstract available.'}",
                "",
            ]
        )

    report = "\n".join(lines)
    state.set_report(report)

    return {
        "status": "success",
        "summary": f"Generated report with {len(state.selected_papers)} papers.",
    }
