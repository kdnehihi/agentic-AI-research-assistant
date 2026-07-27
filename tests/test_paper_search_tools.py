from app.agent.state import AgentState, Paper
from app.tools import paper_search_tools


def test_search_papers_updates_runtime_state_with_multi_source_results(monkeypatch):
    state = AgentState(topic="agentic rag", max_papers=2)
    state.set_search_plan(
        {
            "original_query": "agentic rag",
            "core_terms": ["agentic rag"],
            "context_terms": [],
            "categories": ["cs.AI"],
            "arxiv_query": "all:agentic AND all:rag",
            "planner": "llm",
        }
    )
    paper = Paper(
        paper_id="arxiv:2603.07379",
        title="SoK: Agentic Retrieval-Augmented Generation",
        source="arxiv",
        url="https://arxiv.org/abs/2603.07379",
        semantic_scholar_id="S2ID",
    )
    captured = {}

    def fake_search_paper_sources(**kwargs):
        captured.update(kwargs)
        return {
            "status": "success",
            "query": kwargs["query"],
            "sources": ["arxiv", "semantic_scholar"],
            "source_results": [
                {"source": "arxiv", "status": "success"},
                {"source": "semantic_scholar", "status": "success"},
            ],
            "papers": [paper],
            "raw_candidate_count": 2,
            "summary": "ok",
        }

    monkeypatch.setattr(
        paper_search_tools,
        "search_paper_sources",
        fake_search_paper_sources,
    )

    observation = paper_search_tools.search_papers(
        state,
        query="agentic rag",
        max_results=7,
        sources=["arxiv", "semantic_scholar"],
    )

    assert captured == {
        "query": "agentic rag",
        "max_results": 7,
        "sources": ["arxiv", "semantic_scholar"],
        "arxiv_query": "all:agentic AND all:rag",
    }
    assert state.candidate_papers == [paper]
    assert state.searched_sources == ["arxiv", "semantic_scholar"]
    assert observation["status"] == "success"
    assert observation["candidate_paper_ids"] == ["arxiv:2603.07379"]
    assert observation["raw_candidate_count"] == 2


def test_search_papers_enriches_ambiguous_transformer_query(monkeypatch):
    state = AgentState(topic="find 5 latest paper about transformer", max_papers=5)
    captured = {}

    def fake_search_paper_sources(**kwargs):
        captured.update(kwargs)
        return {
            "status": "success",
            "query": kwargs["query"],
            "sources": ["arxiv", "semantic_scholar"],
            "source_results": [],
            "papers": [],
            "raw_candidate_count": 0,
            "summary": "ok",
        }

    monkeypatch.setattr(
        paper_search_tools,
        "search_paper_sources",
        fake_search_paper_sources,
    )

    observation = paper_search_tools.search_papers(
        state,
        query="find 5 latest paper about transformer",
    )

    assert captured["query"].startswith("find 5 latest paper about transformer")
    assert "attention" in captured["query"]
    assert "neural network" in captured["query"]
    assert observation["search_query"] == "find 5 latest paper about transformer"
    assert observation["source_query"] == captured["query"]
