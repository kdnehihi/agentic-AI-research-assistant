import pytest

from app.agent.state import AgentState
from app.agent.tool_spec import RetrieveEvidenceArgs
from app.tools.registry import ToolRegistry


def test_production_catalog_excludes_fake_eval_and_admin_tools():
    registry = ToolRegistry()

    production_tools = registry.list_tools(category="production")
    development_tools = registry.list_tools(category="development")
    admin_tools = registry.list_tools(category="admin")

    assert "discover_papers" in production_tools
    assert "retrieve_evidence" in production_tools
    assert "search_fake_papers" not in production_tools
    assert "evaluate_retrieval_from_selected_chunks" not in production_tools
    assert "remove_fetched_papers" not in production_tools

    assert "search_fake_papers" in development_tools
    assert "evaluate_retrieval_from_selected_chunks" in development_tools
    assert "remove_fetched_papers" in admin_tools
    assert "remove_papers_from_kb" in admin_tools


def test_production_tool_schema_rejects_invalid_options():
    with pytest.raises(ValueError):
        RetrieveEvidenceArgs(query="", top_k=0)


def test_registry_validates_production_tool_arguments():
    registry = ToolRegistry()
    state = AgentState(topic="test")

    with pytest.raises(ValueError):
        registry.execute("retrieve_evidence", state=state, query="", top_k=0)
