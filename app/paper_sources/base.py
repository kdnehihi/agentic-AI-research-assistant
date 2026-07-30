from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from app.agent.state import Paper


class PaperSourceSearchRequest(BaseModel):
    """Normalized request passed to every paper-source adapter."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    max_results: int = Field(default=20, ge=1, le=100)
    arxiv_query: str | None = None
    publication_years: list[int] = Field(default_factory=list)


class PaperSourceProvenance(BaseModel):
    """Source-level provenance for one candidate paper."""

    model_config = ConfigDict(extra="forbid")

    source: str
    source_paper_id: str | None = None
    url: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class PaperCandidate(BaseModel):
    """Canonical paper candidate emitted by all source adapters."""

    model_config = ConfigDict(extra="forbid")

    title: str
    paper_id: str | None = None
    authors: list[str] = Field(default_factory=list)
    source: str
    url: str
    abstract: str | None = None
    published_date: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    semantic_scholar_id: str | None = None
    external_ids: dict[str, str] = Field(default_factory=dict)
    venue: str | None = None
    citation_count: int | None = None
    open_access_pdf_url: str | None = None
    provenance: list[PaperSourceProvenance] = Field(default_factory=list)

    def to_paper(self) -> Paper:
        """Convert a source candidate into the agent's Paper model."""

        return Paper(
            title=self.title,
            paper_id=self.paper_id,
            authors=self.authors,
            source=self.source,
            url=self.url,
            abstract=self.abstract,
            published_date=self.published_date,
            doi=self.doi,
            arxiv_id=self.arxiv_id,
            semantic_scholar_id=self.semantic_scholar_id,
            external_ids=dict(self.external_ids),
            provenance=[
                provenance.model_dump(mode="json")
                for provenance in self.provenance
            ],
            venue=self.venue,
            citation_count=self.citation_count,
            open_access_pdf_url=self.open_access_pdf_url,
        )


class PaperSourceResult(BaseModel):
    """Result returned by one source adapter."""

    model_config = ConfigDict(extra="forbid")

    source: str
    status: str
    candidates: list[PaperCandidate] = Field(default_factory=list)
    query: str | None = None
    error: str | None = None
    summary: str = ""


class PaperSource(Protocol):
    """Common adapter interface for academic paper sources."""

    name: str

    def search(self, request: PaperSourceSearchRequest) -> PaperSourceResult:
        """Search one source and return normalized candidates."""
