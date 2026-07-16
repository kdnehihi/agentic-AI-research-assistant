from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


ToolCategory = Literal["production", "development", "admin"]
SortBy = Literal["published_date", "added_date", "relevance"]
SummaryMode = Literal["abstract", "full_paper", "method", "contributions", "limitations"]
ReportType = Literal["digest", "comparison", "literature_review", "brief"]


class ToolSpec(BaseModel):
    """Planner-facing metadata for one registered capability."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    args_schema: type[BaseModel]
    read_only: bool
    runtime_state_mutation: bool = False
    persistent_side_effect: bool = False
    destructive: bool = False
    requires_confirmation: bool = False
    category: ToolCategory
    prerequisites: list[str] = Field(default_factory=list)
    output_shape: dict[str, Any] = Field(default_factory=dict)

class DiscoverPapersArgs(BaseModel):
    """Arguments for discovering and ranking papers from a user query."""

    user_query: str = Field(min_length=1)
    max_results: int | None = Field(default=None, ge=1, le=100)
    max_selected: int | None = Field(default=None, ge=1, le=20)
    exclude_seen: bool = True
    use_llm_query_planner: bool = False


class ListPapersArgs(BaseModel):
    """Arguments for read-only paper metadata listing."""

    knowledge_base_id: str | None = None
    paper_ids: list[str] | None = None
    published_after: str | None = None
    published_before: str | None = None
    added_after: str | None = None
    limit: int = Field(default=10, ge=1, le=100)
    sort_by: SortBy = "published_date"
    descending: bool = True


class GetPaperMetadataArgs(BaseModel):
    """Arguments for retrieving compact paper metadata."""

    paper_ids: list[str] = Field(min_length=1)


class SavePapersToKbArgs(BaseModel):
    """Arguments for idempotently saving explicit papers to SQLite."""

    paper_ids: list[str] = Field(min_length=1)
    knowledge_base_id: str = Field(default="default", min_length=1)


class EnsurePapersRetrievableArgs(BaseModel):
    """Arguments for ensuring papers are semantically retrievable."""

    paper_ids: list[str] = Field(min_length=1)
    force_reindex: bool = False


class RetrieveEvidenceArgs(BaseModel):
    """Arguments for semantic/hybrid evidence retrieval."""

    query: str = Field(min_length=1)
    paper_ids: list[str] | None = None
    knowledge_base_ids: list[str] | None = None
    section_groups: list[str] | None = None
    published_after: str | None = None
    published_before: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=50)
    candidate_k: int | None = Field(default=None, ge=1, le=200)


class SummarizePapersArgs(BaseModel):
    """Arguments for summarizing explicit papers."""

    paper_ids: list[str] = Field(min_length=1)
    summary_mode: SummaryMode = "abstract"


class GeneratePaperReportArgs(BaseModel):
    """Arguments for generating a report from explicit papers."""

    paper_ids: list[str] = Field(min_length=1)
    report_type: ReportType = "digest"
    user_focus: str | None = None


class EmptyArgs(BaseModel):
    """Placeholder schema for legacy tools whose args are not planner-facing."""

    pass
