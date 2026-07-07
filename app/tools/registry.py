from __future__ import annotations
from typing import Any, Callable

from app.agent.state import AgentState
from app.tools.fake_paper_tools import (
    search_fake_papers,
    deduplicate_papers,
    rank_papers,
    generate_report_from_abstracts,
)
from app.tools.arxiv_tools import search_arxiv_papers
from app.tools.filter_relevant_papers import filter_relevant_papers

ToolFunction = Callable[..., dict[str, Any]]


class ToolRegistry:
    """
    A registry for tools that can be used by the agent.
    """

    def __init__(self):
        self.tools: dict[str, ToolFunction] = {
            "search_fake_papers": search_fake_papers,
            "deduplicate_papers": deduplicate_papers,
            "rank_papers": rank_papers,
            "generate_report_from_abstracts": generate_report_from_abstracts,
            "search_arxiv_papers": search_arxiv_papers,
            "filter_relevant_papers": filter_relevant_papers,
        }

    def has_tool(self, tool_name: str) -> bool:
        """Check if a tool is registered."""
        return tool_name in self.tools

    def list_tools(self) -> list[str]:
        """List all registered tools."""
        return list(self.tools.keys())

    def execute(self, tool_name: str, state: AgentState, **kwargs) -> dict[str, Any]:
        """Execute a registered tool with the given state and arguments."""
        if not self.has_tool(tool_name):
            raise ValueError(f"Tool '{tool_name}' is not registered.")

        tool_function = self.tools[tool_name]
        observation = tool_function(state, **kwargs)
        if not isinstance(observation, dict):
            raise ValueError(f"Tool '{tool_name}' did not return a dictionary.")
        return observation
