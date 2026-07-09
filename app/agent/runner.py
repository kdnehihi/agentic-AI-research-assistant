from app.agent.state import AgentState, ToolLog
from app.tools.registry import ToolRegistry

DEFAULT_WORKFLOW = [
    "search_fake_papers",
    "deduplicate_papers",
    "rank_papers",
    "generate_fake_report",
]

ARXIV_WORKFLOW = [
    "search_arxiv_papers",
    "deduplicate_papers",
    "rank_papers_by_similarity",
    "filter_relevant_papers",
    "generate_report_from_abstracts",
]

LLM_SUMMARY_WORKFLOW = [
    "search_arxiv_papers",
    "deduplicate_papers",
    "rank_papers_by_similarity",
    "filter_relevant_papers",
    "summarize_papers_with_llm",
    "generate_report_from_abstracts",
]

LLM_QUERY_ARXIV_WORKFLOW = [
    "plan_arxiv_search_query_with_llm",
    "search_arxiv_papers",
    "deduplicate_papers",
    "rank_papers_by_similarity",
    "filter_relevant_papers",
    "generate_report_from_abstracts",
]


class AgentRunner:
    """
    A class to run the agent's workflow using the ToolRegistry.
    """

    def __init__(self, state: AgentState, registry: ToolRegistry):
        self.state = state
        self.registry = registry

    def run_workflow(self, workflow: list[str] | None = None) -> None:
        """
        Run the specified workflow of tools. If no workflow is provided,
        use the default workflow.
        """
        if workflow is None:
            workflow = DEFAULT_WORKFLOW

        for tool_name in workflow:
            if not self.registry.has_tool(tool_name):
                raise ValueError(f"Tool '{tool_name}' is not registered.")
            observation = self.registry.execute(tool_name, self.state)
            print(f"Executed {tool_name}: {observation}")
            self.state.add_tool_log(
                ToolLog(
                    tool_name=tool_name,
                    status=observation.get("status", "unknown"),
                    output_summary=observation.get("summary", ""),
                    error=observation.get("error"),
                )
            )
