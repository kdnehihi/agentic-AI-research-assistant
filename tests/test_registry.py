import pytest

from app.agent.state import AgentState
from app.tools.registry import ToolRegistry


def test_registry_lists_tools():
    registry = ToolRegistry()

    tools = registry.list_tools()

    assert "search_fake_papers" in tools
    assert "deduplicate_papers" in tools
    assert "rank_papers" in tools
    assert "generate_fake_report" in tools
    assert "generate_report_from_abstracts" in tools
    assert "summarize_papers_from_abstracts" in tools
    assert "summarize_papers_with_llm" in tools
    assert "search_arxiv_papers" in tools
    assert "filter_relevant_papers" in tools


def test_registry_has_tool():
    registry = ToolRegistry()

    assert registry.has_tool("search_fake_papers") is True
    assert registry.has_tool("UNKNOWN_TOOL") is False


def test_registry_execute_search_fake_papers():
    registry = ToolRegistry()
    state = AgentState(topic="RLHF RLVR reasoning models", max_papers=3)

    observation = registry.execute(
        tool_name="search_fake_papers",
        state=state,
        query=state.topic,
        max_results=10,
    )

    assert observation["status"] == "success"
    assert len(state.candidate_papers) > 0
    assert "fake_arxiv" in state.searched_sources


def test_registry_execute_full_fake_tool_flow():
    registry = ToolRegistry()
    state = AgentState(topic="RLHF RLVR reasoning models", max_papers=3)

    search_obs = registry.execute(
        tool_name="search_fake_papers",
        state=state,
        query=state.topic,
        max_results=10,
    )

    dedup_obs = registry.execute(
        tool_name="deduplicate_papers",
        state=state,
    )

    rank_obs = registry.execute(
        tool_name="rank_papers",
        state=state,
        topic=state.topic,
        max_papers=state.max_papers,
    )

    report_obs = registry.execute(
        tool_name="generate_fake_report",
        state=state,
    )

    assert search_obs["status"] == "success"
    assert dedup_obs["status"] == "success"
    assert rank_obs["status"] == "success"
    assert report_obs["status"] == "success"

    assert len(state.candidate_papers) > 0
    assert len(state.selected_papers) == state.max_papers
    assert state.report is not None
    assert "# Paper Research Report" in state.report


def test_registry_rejects_unknown_tool():
    registry = ToolRegistry()
    state = AgentState(topic="RLHF")

    with pytest.raises(ValueError):
        registry.execute(
            tool_name="UNKNOWN_TOOL",
            state=state,
        )
