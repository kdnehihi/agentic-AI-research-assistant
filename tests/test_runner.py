#test for runner.py
from app.agent.runner import AgentRunner, LLM_SUMMARY_WORKFLOW
from app.agent.state import AgentState
from app.tools.fake_paper_tools import search_fake_papers
from app.tools.registry import ToolRegistry

def test_runner_execute_workflow():
    state = AgentState(topic="RLHF RLVR reasoning models", max_papers=3)
    registry = ToolRegistry()
    runner = AgentRunner(state=state, registry=registry)

    runner.run_workflow()

    # Check that tool logs have been added for each tool in the default workflow
    assert len(state.tool_logs) == 4
    assert state.tool_logs[0].tool_name == "search_fake_papers"
    assert state.tool_logs[1].tool_name == "deduplicate_papers"
    assert state.tool_logs[2].tool_name == "rank_papers"
    assert state.tool_logs[3].tool_name == "generate_fake_report"
    assert len(state.candidate_papers) > 0
    assert len(state.selected_papers) == state.max_papers
    assert state.report is not None


def test_runner_execute_llm_summary_workflow_without_network():
    state = AgentState(topic="RLHF RLVR reasoning models", max_papers=3)
    registry = ToolRegistry()
    registry.tools["search_arxiv_papers"] = search_fake_papers
    registry.tools["filter_seen_papers"] = lambda state: {
        "status": "success",
        "summary": "Skipped seen-paper filter in test.",
    }
    registry.tools["save_selected_papers_to_kb"] = lambda state: {
        "status": "success",
        "summary": "Skipped paper store save in test.",
    }
    registry.tools["fetch_selected_papers"] = lambda state: {
        "status": "success",
        "summary": "Skipped full text fetch in test.",
    }
    runner = AgentRunner(state=state, registry=registry)

    runner.run_workflow(workflow=LLM_SUMMARY_WORKFLOW)

    expected_tools = [
        "search_arxiv_papers",
        "filter_seen_papers",
        "deduplicate_papers",
        "rank_papers_by_similarity",
        "filter_relevant_papers",
        "fetch_selected_papers",
        "summarize_papers_with_llm",
        "generate_report_from_abstracts",
        "save_selected_papers_to_kb",
    ]

    assert [tool_log.tool_name for tool_log in state.tool_logs] == expected_tools
    assert len(state.candidate_papers) > 0
    assert len(state.selected_papers) > 0
    assert len(state.paper_summaries) == len(state.selected_papers)
    assert state.report is not None
    assert "reinforcement learning" in state.report.lower()
