from app.paper_sources.base import (
    PaperCandidate,
    PaperSourceProvenance,
    PaperSourceResult,
    PaperSourceSearchRequest,
)
from app.paper_sources.multi_source import (
    merge_and_deduplicate_candidates,
    search_paper_sources,
    select_source_names,
)
from app.paper_sources.semantic_scholar import _parse_semantic_scholar_response


def test_semantic_scholar_parser_normalizes_candidate_metadata():
    payload = {
        "data": [
            {
                "paperId": "S2ID",
                "title": "  Agentic Retrieval Augmented Generation  ",
                "abstract": "  A survey of agentic RAG. ",
                "authors": [{"name": "Saroj Mishra"}, {"name": "Suman Niroula"}],
                "publicationDate": "2026-03-07",
                "url": "https://www.semanticscholar.org/paper/S2ID",
                "venue": "arXiv",
                "citationCount": 12,
                "externalIds": {
                    "ArXiv": "2603.07379",
                    "DOI": "10.1234/example",
                },
                "openAccessPdf": {"url": "https://arxiv.org/pdf/2603.07379"},
            }
        ]
    }

    candidates = _parse_semantic_scholar_response(payload)

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.paper_id == "arxiv:2603.07379"
    assert candidate.source == "semantic_scholar"
    assert candidate.authors == ["Saroj Mishra", "Suman Niroula"]
    assert candidate.doi == "10.1234/example"
    assert candidate.arxiv_id == "2603.07379"
    assert candidate.semantic_scholar_id == "S2ID"
    assert candidate.citation_count == 12
    assert candidate.provenance[0].source == "semantic_scholar"


def test_merge_and_deduplicate_candidates_merges_cross_source_metadata():
    arxiv_candidate = PaperCandidate(
        title="SoK: Agentic Retrieval-Augmented Generation",
        paper_id="arxiv:2603.07379v1",
        authors=["Saroj Mishra"],
        source="arxiv",
        url="https://arxiv.org/abs/2603.07379v1",
        abstract="Short abstract.",
        arxiv_id="2603.07379v1",
        external_ids={"ArXiv": "2603.07379v1"},
        provenance=[
            PaperSourceProvenance(
                source="arxiv",
                source_paper_id="2603.07379v1",
                url="https://arxiv.org/abs/2603.07379v1",
            )
        ],
    )
    semantic_candidate = PaperCandidate(
        title="SoK: Agentic Retrieval-Augmented Generation",
        paper_id="arxiv:2603.07379",
        authors=["Saroj Mishra", "Suman Niroula"],
        source="semantic_scholar",
        url="https://www.semanticscholar.org/paper/S2ID",
        abstract="Longer abstract from Semantic Scholar.",
        arxiv_id="2603.07379",
        semantic_scholar_id="S2ID",
        external_ids={"ArXiv": "2603.07379", "DOI": "10.1234/example"},
        citation_count=12,
        provenance=[
            PaperSourceProvenance(
                source="semantic_scholar",
                source_paper_id="S2ID",
                url="https://www.semanticscholar.org/paper/S2ID",
            )
        ],
    )

    merged = merge_and_deduplicate_candidates([arxiv_candidate, semantic_candidate])

    assert len(merged) == 1
    candidate = merged[0]
    assert candidate.source == "arxiv"
    assert candidate.paper_id == "arxiv:2603.07379v1"
    assert candidate.semantic_scholar_id == "S2ID"
    assert candidate.external_ids["DOI"] == "10.1234/example"
    assert candidate.citation_count == 12
    assert len(candidate.provenance) == 2


def test_search_paper_sources_runs_adapters_and_returns_deduplicated_papers():
    first = _FakeSource(
        source="arxiv",
        candidates=[
            PaperCandidate(
                title="Paper A",
                paper_id="arxiv:1",
                source="arxiv",
                url="https://arxiv.org/abs/1",
                arxiv_id="1",
                external_ids={"ArXiv": "1"},
            )
        ],
    )
    second = _FakeSource(
        source="semantic_scholar",
        candidates=[
            PaperCandidate(
                title="Paper A",
                paper_id="semantic_scholar:s2",
                source="semantic_scholar",
                url="https://semanticscholar.org/paper/s2",
                arxiv_id="1",
                semantic_scholar_id="s2",
                citation_count=5,
            )
        ],
    )

    result = search_paper_sources(
        query="agentic rag",
        max_results=10,
        adapters=[first, second],
    )

    assert result["status"] == "success"
    assert result["sources"] == ["arxiv", "semantic_scholar"]
    assert result["candidate_count"] == 1
    assert result["raw_candidate_count"] == 2
    assert result["papers"][0].paper_id == "arxiv:1"
    assert result["papers"][0].semantic_scholar_id == "s2"


def test_search_paper_sources_reports_partial_success_for_failed_adapter():
    success_source = _FakeSource(
        source="arxiv",
        candidates=[
            PaperCandidate(
                title="Paper A",
                paper_id="arxiv:1",
                source="arxiv",
                url="https://arxiv.org/abs/1",
            )
        ],
    )
    failed_source = _FakeSource(source="semantic_scholar", raises=True)

    result = search_paper_sources(
        query="agentic rag",
        adapters=[success_source, failed_source],
    )

    assert result["status"] == "partial_success"
    assert result["candidate_count"] == 1
    assert result["source_results"][1]["status"] == "failed"


def test_search_paper_sources_prioritizes_recent_candidates_for_latest_queries():
    source = _FakeSource(
        source="semantic_scholar",
        candidates=[
            PaperCandidate(
                title="Older Highly Cited Transformer Paper",
                paper_id="semantic_scholar:old",
                source="semantic_scholar",
                url="https://semanticscholar.org/paper/old",
                published_date="2013-01-01",
                citation_count=10000,
            ),
            PaperCandidate(
                title="New Transformer Attention Paper",
                paper_id="semantic_scholar:new",
                source="semantic_scholar",
                url="https://semanticscholar.org/paper/new",
                published_date="2026-02-01",
                citation_count=2,
            ),
        ],
    )

    result = search_paper_sources(
        query="find latest papers about transformer",
        max_results=1,
        adapters=[source],
    )

    assert result["papers"][0].paper_id == "semantic_scholar:new"


def test_search_paper_sources_filters_to_explicit_publication_year():
    source = _FakeSource(
        source="semantic_scholar",
        candidates=[
            PaperCandidate(
                title="Older Transformer Paper",
                paper_id="semantic_scholar:old",
                source="semantic_scholar",
                url="https://semanticscholar.org/paper/old",
                published_date="2025-12-31",
            ),
            PaperCandidate(
                title="New Transformer Attention Paper",
                paper_id="semantic_scholar:new",
                source="semantic_scholar",
                url="https://semanticscholar.org/paper/new",
                published_date="2026-02-01",
            ),
        ],
    )

    result = search_paper_sources(
        query="find papers in 2026 about transformer language models",
        max_results=10,
        adapters=[source],
    )

    assert result["requested_publication_years"] == [2026]
    assert [paper.paper_id for paper in result["papers"]] == ["semantic_scholar:new"]


def test_select_source_names_filters_future_unimplemented_sources():
    assert select_source_names(query="nlp openreview clinical rag") == [
        "arxiv",
        "semantic_scholar",
    ]
    assert select_source_names(
        query="agentic rag",
        requested_sources=["semantic-scholar", "acl", "arxiv"],
    ) == ["semantic_scholar", "arxiv"]


class _FakeSource:
    def __init__(
        self,
        *,
        source: str,
        candidates: list[PaperCandidate] | None = None,
        raises: bool = False,
    ) -> None:
        self.name = source
        self.candidates = candidates or []
        self.raises = raises

    def search(self, request: PaperSourceSearchRequest) -> PaperSourceResult:
        if self.raises:
            raise RuntimeError("adapter exploded")
        return PaperSourceResult(
            source=self.name,
            status="success",
            query=request.query,
            candidates=self.candidates[: request.max_results],
            summary=f"{self.name} ok",
        )
