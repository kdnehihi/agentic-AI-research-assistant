#test for runner.py
from app.agent.runner import AgentRunner
from app.agent.state import AgentState
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
