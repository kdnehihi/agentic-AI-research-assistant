from __future__ import annotations

from collections.abc import Callable
from typing import Any

from app.agent.tool_spec import (
    DiscoverPapersArgs,
    EmptyArgs,
    EnsurePapersRetrievableArgs,
    GeneratePaperReportArgs,
    GetPaperMetadataArgs,
    ListPapersArgs,
    RetrieveEvidenceArgs,
    SavePapersToKbArgs,
    SummarizePapersArgs,
    ToolCategory,
    ToolSpec,
)
from app.tools.fake_paper_tools import (
    deduplicate_papers,
    generate_fake_report,
    rank_papers,
    search_fake_papers,
)
from app.tools.production.discovery_tools import discover_papers
from app.tools.production.generation_tools import generate_paper_report, summarize_papers
from app.tools.production.ingestion_tools import ensure_papers_retrievable
from app.tools.production.knowledge_base_tools import (
    get_paper_metadata,
    list_papers,
    save_papers_to_kb,
)
from app.tools.production.retrieval_tools import retrieve_evidence
from app.tools.retrieval_eval_tools import evaluate_retrieval_from_selected_chunks
from app.tools.fetch_selected_papers import remove_fetched_papers
from app.tools.knowledge_base_tools import remove_papers_from_kb


ToolFunction = Callable[..., dict[str, Any]]


PRODUCTION_TOOLS: dict[str, ToolFunction] = {
    "discover_papers": discover_papers,
    "list_papers": list_papers,
    "get_paper_metadata": get_paper_metadata,
    "save_papers_to_kb": save_papers_to_kb,
    "ensure_papers_retrievable": ensure_papers_retrievable,
    "retrieve_evidence": retrieve_evidence,
    "summarize_papers": summarize_papers,
    "generate_paper_report": generate_paper_report,
}

DEVELOPMENT_TOOLS: dict[str, ToolFunction] = {
    "search_fake_papers": search_fake_papers,
    "deduplicate_papers": deduplicate_papers,
    "rank_papers": rank_papers,
    "generate_fake_report": generate_fake_report,
    "evaluate_retrieval_from_selected_chunks": evaluate_retrieval_from_selected_chunks,
}

ADMIN_TOOLS: dict[str, ToolFunction] = {
    "remove_fetched_papers": remove_fetched_papers,
    "remove_papers_from_kb": remove_papers_from_kb,
}


def build_tool_specs() -> dict[str, ToolSpec]:
    """Build the complete tool spec map for catalog filtering."""

    specs = {
        "discover_papers": ToolSpec(
            name="discover_papers",
            description="Discover and rank papers for a user research query.",
            args_schema=DiscoverPapersArgs,
            read_only=False,
            runtime_state_mutation=True,
            category="production",
            prerequisites=["user query"],
            output_shape={"selected_paper_ids": "list[str]", "candidate_count": "int"},
        ),
        "list_papers": ToolSpec(
            name="list_papers",
            description="List compact paper metadata from the knowledge base.",
            args_schema=ListPapersArgs,
            read_only=True,
            category="production",
            prerequisites=["SQLite knowledge base"],
        ),
        "get_paper_metadata": ToolSpec(
            name="get_paper_metadata",
            description="Get compact metadata and artifact readiness for paper ids.",
            args_schema=GetPaperMetadataArgs,
            read_only=True,
            category="production",
            prerequisites=["paper ids"],
        ),
        "save_papers_to_kb": ToolSpec(
            name="save_papers_to_kb",
            description="Persist explicit paper ids to the SQLite knowledge base.",
            args_schema=SavePapersToKbArgs,
            read_only=False,
            persistent_side_effect=True,
            category="production",
            prerequisites=["paper metadata exists in state or SQLite"],
        ),
        "ensure_papers_retrievable": ToolSpec(
            name="ensure_papers_retrievable",
            description="Ensure papers are available for semantic retrieval.",
            args_schema=EnsurePapersRetrievableArgs,
            read_only=False,
            runtime_state_mutation=True,
            persistent_side_effect=True,
            category="production",
            prerequisites=["paper metadata exists"],
        ),
        "retrieve_evidence": ToolSpec(
            name="retrieve_evidence",
            description=(
                "Retrieve evidence chunks from indexed papers or knowledge bases. "
                "Omit paper_ids to search all indexed knowledge-base chunks."
            ),
            args_schema=RetrieveEvidenceArgs,
            read_only=True,
            category="production",
            prerequisites=["papers are indexed for retrieval"],
        ),
        "summarize_papers": ToolSpec(
            name="summarize_papers",
            description="Summarize explicit papers using abstract or available evidence.",
            args_schema=SummarizePapersArgs,
            read_only=False,
            runtime_state_mutation=True,
            category="production",
            prerequisites=["paper metadata exists"],
        ),
        "generate_paper_report": ToolSpec(
            name="generate_paper_report",
            description="Generate a report for explicit paper ids.",
            args_schema=GeneratePaperReportArgs,
            read_only=False,
            runtime_state_mutation=True,
            category="production",
            prerequisites=["paper metadata exists"],
        ),
    }

    for name in DEVELOPMENT_TOOLS:
        specs[name] = ToolSpec(
            name=name,
            description="Development or evaluation helper; not planner-facing.",
            args_schema=EmptyArgs,
            read_only=False,
            category="development",
        )
    for name in ADMIN_TOOLS:
        specs[name] = ToolSpec(
            name=name,
            description="Administrative cleanup helper; not planner-facing.",
            args_schema=EmptyArgs,
            read_only=False,
            persistent_side_effect=True,
            destructive=True,
            requires_confirmation=True,
            category="admin",
        )
    return specs


def list_tool_names(category: ToolCategory | None = None) -> list[str]:
    """List tool names by catalog category."""

    specs = build_tool_specs()
    return [
        name
        for name, spec in specs.items()
        if category is None or spec.category == category
    ]
