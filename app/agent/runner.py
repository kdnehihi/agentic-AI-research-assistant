from app.agent.state import AgentState, ToolLog, ToolStatus
from app.tools.registry import ToolRegistry

DEFAULT_WORKFLOW = [
    "search_fake_papers",
    "deduplicate_papers",
    "rank_papers",
    "generate_fake_report",
]

ARXIV_SEARCH_AND_FETCH_WORKFLOW = [
    "search_arxiv_papers",
    "filter_seen_papers",
    "deduplicate_papers",
    "rank_papers_by_similarity",
    "filter_relevant_papers",
    "fetch_selected_papers",
]

ARXIV_FULL_TEXT_EMBED_WORKFLOW = [
    *ARXIV_SEARCH_AND_FETCH_WORKFLOW,
    "extract_pdf_text_for_selected_papers",
    "chunk_selected_papers_by_section",
    "embed_selected_paper_chunks",
]

ARXIV_RAG_INDEX_WORKFLOW = [
    *ARXIV_FULL_TEXT_EMBED_WORKFLOW,
    "index_selected_paper_chunks",
]

ARXIV_RAG_RETRIEVAL_WORKFLOW = [
    *ARXIV_RAG_INDEX_WORKFLOW,
    "retrieve_chunks_from_papers",
]

ARXIV_RAG_EVAL_WORKFLOW = [
    *ARXIV_RAG_INDEX_WORKFLOW,
    "evaluate_retrieval_from_selected_chunks",
]

ARXIV_WORKFLOW = [
    *ARXIV_SEARCH_AND_FETCH_WORKFLOW,
    "generate_report_from_abstracts",
    "save_selected_papers_to_kb",
]

LLM_SUMMARY_WORKFLOW = [
    *ARXIV_SEARCH_AND_FETCH_WORKFLOW,
    "summarize_papers_with_llm",
    "generate_report_from_abstracts",
    "save_selected_papers_to_kb",
]

LLM_QUERY_ARXIV_WORKFLOW = [
    "plan_arxiv_search_query_with_llm",
    *ARXIV_SEARCH_AND_FETCH_WORKFLOW,
    "generate_report_from_abstracts",
    "save_selected_papers_to_kb",
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
                    status=_normalize_tool_status(observation.get("status")),
                    output_summary=observation.get("summary", ""),
                    error=observation.get("error"),
                )
            )


def _normalize_tool_status(status: object) -> ToolStatus:
    if status in {"not_started", "success", "partial_success", "failed", "skipped"}:
        return status

    if status in {"error", "failure"}:
        return "failed"

    return "failed"
