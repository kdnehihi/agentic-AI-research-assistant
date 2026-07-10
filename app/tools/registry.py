from __future__ import annotations
from typing import Any, Callable

from app.agent.state import AgentState
from app.tools.fake_paper_tools import (
    search_fake_papers,
    deduplicate_papers,
    rank_papers,
    generate_fake_report,
)
from app.tools.arxiv_tools import search_arxiv_papers
from app.tools.filter_relevant_papers import filter_relevant_papers
from app.tools.report_tools import (
    generate_report_from_abstracts,
    summarize_papers_from_abstracts,
)
from app.tools.llm_summary_tools import summarize_papers_with_llm
from app.tools.llm_query_planner_tools import plan_arxiv_search_query_with_llm
from app.tools.scoring_tools import rank_papers_by_similarity
from app.tools.knowledge_base_tools import (
    filter_seen_papers,
    remove_papers_from_kb,
    save_candidate_papers_to_kb,
    save_selected_papers_to_kb,
)
from app.tools.fetch_selected_papers import fetch_selected_papers

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
            "rank_papers_by_similarity": rank_papers_by_similarity,
            "generate_fake_report": generate_fake_report,
            "generate_report_from_abstracts": generate_report_from_abstracts,
            "summarize_papers_from_abstracts": summarize_papers_from_abstracts,
            "summarize_papers_with_llm": summarize_papers_with_llm,
            "plan_arxiv_search_query_with_llm": plan_arxiv_search_query_with_llm,
            "search_arxiv_papers": search_arxiv_papers,
            "filter_relevant_papers": filter_relevant_papers,
            "filter_seen_papers": filter_seen_papers,
            "fetch_selected_papers": fetch_selected_papers,
            "save_candidate_papers_to_kb": save_candidate_papers_to_kb,
            "save_selected_papers_to_kb": save_selected_papers_to_kb,
            "remove_papers_from_kb": remove_papers_from_kb,
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
