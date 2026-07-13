from app.agent.state import AgentState, Paper
from app.workflows.paper_discovery import discover_papers_workflow


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
