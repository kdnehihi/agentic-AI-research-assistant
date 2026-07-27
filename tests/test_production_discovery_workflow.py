from app.agent.state import AgentState, Paper
from app.workflows.paper_discovery import discover_papers_workflow
from app.tools.filter_relevant_papers import filter_relevant_papers
from app.tools.scoring_tools import rank_papers_by_similarity


def test_discover_papers_runs_internal_steps_in_order():
    state = AgentState(topic="old topic", max_papers=3)
    calls = []

    def plan_step(state):
        calls.append("plan")
        state.set_search_plan(
            {
                "original_query": state.topic,
                "core_terms": ["rag"],
                "context_terms": ["science"],
                "categories": ["cs.CL"],
                "arxiv_query": "planned query",
                "planner": "rule_based",
            }
        )
        return {"status": "success", "search_query": "planned query"}

    def search_step(state, query, max_results):
        calls.append(("search", query, max_results))
        state.set_candidate_papers(
            [
                Paper(paper_id="p1", title="Paper 1", source="arxiv", url="https://x/p1"),
                Paper(paper_id="p2", title="Paper 2", source="arxiv", url="https://x/p2"),
            ]
        )
        return {"status": "success"}

    def filter_seen_step(state):
        calls.append("filter_seen")
        state.set_candidate_papers(state.candidate_papers[:1])
        return {"status": "success", "removed_seen": 1}

    def dedupe_step(state):
        calls.append("dedupe")
        return {"status": "success"}

    def rank_step(state, query, max_papers):
        calls.append(("rank", query, max_papers))
        state.set_selected_papers(state.candidate_papers[:max_papers])
        return {"status": "success"}

    def relevance_step(state):
        calls.append("relevance")
        return {"status": "success"}

    observation = discover_papers_workflow(
        state=state,
        user_query="agentic rag",
        max_results=7,
        max_selected=1,
        exclude_seen=True,
        plan_step=plan_step,
        search_step=search_step,
        filter_seen_step=filter_seen_step,
        dedupe_step=dedupe_step,
        rank_step=rank_step,
        relevance_step=relevance_step,
    )

    assert calls == [
        "plan",
        ("search", "agentic rag", 7),
        "filter_seen",
        "dedupe",
        ("rank", "agentic rag", 1),
        "relevance",
    ]
    assert observation["planned_query"] == "planned query"
    assert observation["candidate_paper_ids"] == ["p1"]
    assert observation["selected_paper_ids"] == ["p1"]
    assert observation["excluded_seen_count"] == 1
    assert state.topic == "old topic"


def test_discover_papers_stops_when_search_fails():
    state = AgentState(topic="old topic", max_papers=3)
    calls = []

    def plan_step(state):
        calls.append("plan")
        return {"status": "success", "planner": "rule_based"}

    def search_step(state, query, max_results):
        calls.append("search")
        return {
            "status": "failed",
            "summary": "arXiv timed out",
            "error": "timeout",
            "search_query": "bad query",
        }

    def filter_seen_step(state):
        calls.append("filter_seen")
        return {"status": "success"}

    observation = discover_papers_workflow(
        state=state,
        user_query="agent memory",
        plan_step=plan_step,
        search_step=search_step,
        filter_seen_step=filter_seen_step,
    )

    assert calls == ["plan", "search"]
    assert observation["status"] == "failed"
    assert observation["failed_step"] == "search"
    assert observation["error"] == "timeout"
    assert "paper search failed" in observation["summary"]
    assert state.topic == "old topic"


def test_discover_papers_retries_llm_search_with_rule_based_query():
    state = AgentState(topic="old topic", max_papers=3)
    calls = []

    def plan_step(state):
        calls.append("plan")
        state.set_search_plan(
            {
                "original_query": state.topic,
                "core_terms": ["agent memory"],
                "context_terms": ["research assistant"],
                "categories": ["cs.AI"],
                "arxiv_query": "llm query",
                "planner": "llm",
            }
        )
        return {
            "status": "success",
            "planner": "llm",
            "search_query": "llm query",
        }

    def search_step(state, query, max_results):
        calls.append(("search", state.search_plan.arxiv_query if state.search_plan else None))
        if state.search_plan is not None:
            return {"status": "failed", "summary": "timeout", "num_results": 0}
        state.set_candidate_papers(
            [Paper(paper_id="p1", title="Agent Memory", source="arxiv", url="https://x/p1")]
        )
        return {"status": "success", "summary": "ok", "num_results": 1}

    def filter_seen_step(state):
        calls.append("filter_seen")
        return {"status": "success", "removed_seen": 0}

    def dedupe_step(state):
        calls.append("dedupe")
        return {"status": "success"}

    def rank_step(state, query, max_papers):
        calls.append("rank")
        state.set_selected_papers(state.candidate_papers)
        return {"status": "success"}

    def relevance_step(state):
        calls.append("relevance")
        return {"status": "success"}

    observation = discover_papers_workflow(
        state=state,
        user_query="agent memory",
        max_selected=1,
        plan_step=plan_step,
        search_step=search_step,
        filter_seen_step=filter_seen_step,
        dedupe_step=dedupe_step,
        rank_step=rank_step,
        relevance_step=relevance_step,
    )

    assert calls == [
        "plan",
        ("search", "llm query"),
        ("search", None),
        "filter_seen",
        "dedupe",
        "rank",
        "relevance",
    ]
    assert observation["status"] == "success"
    assert observation["candidate_paper_ids"] == ["p1"]
    assert observation["selected_paper_ids"] == ["p1"]
    assert observation["steps"]["search_fallback"]["status"] == "success"


def test_discover_papers_passes_requested_sources_to_search_step():
    state = AgentState(topic="old topic", max_papers=3)
    calls = []

    def plan_step(state):
        return {"status": "skipped", "planner": "rule_based"}

    def search_step(state, query, max_results, sources):
        calls.append((query, max_results, sources))
        state.set_candidate_papers(
            [
                Paper(
                    paper_id="semantic_scholar:s2",
                    title="Semantic Scholar Paper",
                    source="semantic_scholar",
                    url="https://semanticscholar.org/paper/s2",
                )
            ]
        )
        return {
            "status": "success",
            "num_results": 1,
            "sources": sources,
        }

    def rank_step(state, query, max_papers):
        state.set_selected_papers(state.candidate_papers)
        return {"status": "success"}

    def passthrough_step(state):
        return {"status": "success"}

    observation = discover_papers_workflow(
        state=state,
        user_query="agentic rag",
        max_results=4,
        sources=["semantic_scholar"],
        exclude_seen=False,
        plan_step=plan_step,
        search_step=search_step,
        dedupe_step=passthrough_step,
        rank_step=rank_step,
        relevance_step=passthrough_step,
    )

    assert calls == [("agentic rag", 4, ["semantic_scholar"])]
    assert observation["sources"] == ["semantic_scholar"]
    assert observation["selected_paper_ids"] == ["semantic_scholar:s2"]


def test_discover_papers_filters_off_domain_transformer_results_end_to_end():
    state = AgentState(topic="old topic", max_papers=5)

    def plan_step(state):
        state.search_plan = None
        return {"status": "skipped", "planner": "rule_based"}

    def search_step(state, query, max_results):
        assert query == "find 5 latest paper about transformer"
        assert max_results == 50
        state.set_candidate_papers(
            [
                Paper(
                    paper_id="semantic_scholar:power",
                    title=(
                        "Analysis of Two Kinds of UHV Transformer Regulation "
                        "Method and Voltage Regulation Compensation"
                    ),
                    source="semantic_scholar",
                    url="https://example.com/power",
                    abstract=(
                        "This paper studies electrical voltage regulation for "
                        "power transformers in UHV grids."
                    ),
                    published_date="2013-01-01",
                ),
                Paper(
                    paper_id="arxiv:teller",
                    title=(
                        "Teller: Real-Time Streaming Audio-Driven Portrait "
                        "Animation with Autoregressive Motion Generation"
                    ),
                    source="arxiv",
                    url="https://example.com/teller",
                    abstract="This paper studies portrait animation.",
                    published_date="2025-03-24",
                ),
                Paper(
                    paper_id="arxiv:transformer-ai",
                    title="Efficient Transformer Language Models with Linear Attention",
                    source="arxiv",
                    url="https://example.com/transformer-ai",
                    abstract=(
                        "This paper studies transformer neural networks, "
                        "attention mechanisms, and large language models."
                    ),
                    published_date="2026-02-01",
                ),
            ]
        )
        return {
            "status": "success",
            "num_results": 3,
            "sources": ["arxiv", "semantic_scholar"],
            "summary": "ok",
        }

    def dedupe_step(state):
        return {"status": "success"}

    observation = discover_papers_workflow(
        state=state,
        user_query="find 5 latest paper about transformer",
        max_selected=5,
        exclude_seen=False,
        plan_step=plan_step,
        search_step=search_step,
        dedupe_step=dedupe_step,
        rank_step=rank_papers_by_similarity,
        relevance_step=filter_relevant_papers,
    )

    assert observation["status"] == "success"
    assert observation["candidate_paper_ids"] == [
        "semantic_scholar:power",
        "arxiv:teller",
        "arxiv:transformer-ai",
    ]
    assert observation["selected_paper_ids"] == ["arxiv:transformer-ai"]
    assert state.selected_papers[0].published_date == "2026-02-01"
