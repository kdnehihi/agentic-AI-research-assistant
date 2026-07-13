from __future__ import annotations
from typing import Any, Callable

from app.agent.state import AgentState
from app.agent.tool_catalog import (
    ADMIN_TOOLS,
    DEVELOPMENT_TOOLS,
    PRODUCTION_TOOLS,
    build_tool_specs,
)
from app.agent.tool_spec import ToolCategory, ToolSpec
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
from app.tools.fetch_selected_papers import fetch_selected_papers, remove_fetched_papers
from app.tools.pdf_text_tools import extract_pdf_text_for_selected_papers
from app.tools.chunking_tools import chunk_selected_papers_by_section
from app.tools.embedding_tools import embed_selected_paper_chunks
from app.tools.vector_store_tools import index_selected_paper_chunks
from app.tools.retrieval_tools import (
    retrieve_chunks_from_knowledge_base,
    retrieve_chunks_from_papers,
)
from app.tools.retrieval_eval_tools import evaluate_retrieval_from_selected_chunks
from app.tools.rag_answer_tools import answer_question_with_retrieval

ToolFunction = Callable[..., dict[str, Any]]


class ToolRegistry:
    """
    A registry for tools that can be used by the agent.
    """

    def __init__(self):
        self.tools: dict[str, ToolFunction] = {
            **PRODUCTION_TOOLS,
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
            "extract_pdf_text_for_selected_papers": extract_pdf_text_for_selected_papers,
            "chunk_selected_papers_by_section": chunk_selected_papers_by_section,
            "embed_selected_paper_chunks": embed_selected_paper_chunks,
            "index_selected_paper_chunks": index_selected_paper_chunks,
            "retrieve_chunks_from_knowledge_base": retrieve_chunks_from_knowledge_base,
            "retrieve_chunks_from_papers": retrieve_chunks_from_papers,
            "evaluate_retrieval_from_selected_chunks": evaluate_retrieval_from_selected_chunks,
            "answer_question_with_retrieval": answer_question_with_retrieval,
            "remove_fetched_papers": remove_fetched_papers,
            "save_candidate_papers_to_kb": save_candidate_papers_to_kb,
            "save_selected_papers_to_kb": save_selected_papers_to_kb,
            "remove_papers_from_kb": remove_papers_from_kb,
        }
        self.tools.update(DEVELOPMENT_TOOLS)
        self.tools.update(ADMIN_TOOLS)
        self.specs: dict[str, ToolSpec] = build_tool_specs()

    def has_tool(self, tool_name: str) -> bool:
        """Check if a tool is registered."""
        return tool_name in self.tools

    def list_tools(self, category: ToolCategory | None = None) -> list[str]:
        """List registered tools, optionally filtered by catalog category."""
        if category is not None:
            return [
                name
                for name, spec in self.specs.items()
                if spec.category == category and name in self.tools
            ]
        return list(self.tools.keys())

    def get_tool_spec(self, tool_name: str) -> ToolSpec:
        """Return planner-facing metadata for a cataloged tool."""
        if tool_name not in self.specs:
            raise ValueError(f"Tool '{tool_name}' does not have a catalog spec.")
        return self.specs[tool_name]

    def execute(self, tool_name: str, state: AgentState, **kwargs) -> dict[str, Any]:
        """Execute a registered tool with the given state and arguments."""
        if not self.has_tool(tool_name):
            raise ValueError(f"Tool '{tool_name}' is not registered.")

        kwargs = self._validated_kwargs(tool_name, kwargs)
        tool_function = self.tools[tool_name]
        observation = tool_function(state, **kwargs)
        if not isinstance(observation, dict):
            raise ValueError(f"Tool '{tool_name}' did not return a dictionary.")
        return observation

    def _validated_kwargs(self, tool_name: str, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Validate production tool kwargs with their catalog schema."""

        spec = self.specs.get(tool_name)
        if spec is None or spec.category != "production":
            return kwargs
        args = spec.args_schema(**kwargs)
        return args.model_dump(exclude_unset=True)
