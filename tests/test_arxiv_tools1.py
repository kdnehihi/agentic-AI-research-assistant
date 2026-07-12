from app.tools.arxiv_tools import search_arxiv_papers
from tests.test_arxiv_tools0 import FAKE_ARXIV_XML
from app.agent.state import AgentState, SearchPlan
from app.tools import arxiv_tools
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse


class FakeHTTPResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def read(self):
        return FAKE_ARXIV_XML


def fake_urlopen(request, timeout):
    params = parse_qs(urlparse(request.full_url).query)
    assert "search_query" in params
    assert "RLHF" in params["search_query"][0]
    assert "verifiable rewards" in params["search_query"][0]
    assert "all:reasoning" not in params["search_query"][0]
    assert params["max_results"] == ["20"]
    assert params["sortBy"] == ["relevance"]
    assert timeout == arxiv_tools.ARXIV_TIMEOUT_SECONDS
    assert request.headers["User-agent"] == arxiv_tools.ARXIV_USER_AGENT
    return FakeHTTPResponse()


def test_search_arxiv_papers_updates_candidate_papers(monkeypatch):
    monkeypatch.setattr(arxiv_tools, "urlopen", fake_urlopen)

    state = AgentState(
        topic="RLHF reasoning models",
        max_papers=2,
    )

    observation = search_arxiv_papers(state)

    assert observation["status"] == "success"
    assert observation["num_results"] == 2
    assert len(state.candidate_papers) == 2

    assert state.candidate_papers[0].title == "RLHF for Reasoning Models"
    assert state.candidate_papers[1].paper_id == "arxiv:2401.67890v1"

def test_search_arxiv_papers_handles_fetch_error(monkeypatch):
    def fake_failed_urlopen(request, timeout):
        raise TimeoutError("network timeout")

    monkeypatch.setattr(arxiv_tools, "urlopen", fake_failed_urlopen)

    state = AgentState(
        topic="RLHF reasoning models",
        max_papers=2,
    )

    observation = search_arxiv_papers(state)

    assert observation["status"] == "failed"
    assert observation["num_results"] == 0
    assert "Failed to fetch" in observation["summary"]
    assert "network timeout" in observation["error"]
    assert "search_query" in observation


def test_search_arxiv_papers_handles_rate_limit(monkeypatch):
    def fake_rate_limited_urlopen(request, timeout):
        raise HTTPError(
            url=request.full_url,
            code=429,
            msg="Unknown Error",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr(arxiv_tools, "urlopen", fake_rate_limited_urlopen)

    state = AgentState(
        topic="RAG reasoning",
        max_papers=1,
    )

    observation = search_arxiv_papers(state)

    assert observation["status"] == "failed"
    assert observation["num_results"] == 0
    assert "rate-limited" in observation["summary"]
    assert "HTTP Error 429" in observation["error"]


def test_search_arxiv_papers_uses_existing_search_plan(monkeypatch):
    planned_query = "(ti:RLHF OR abs:RLHF) AND (cat:cs.CL)"

    def fake_urlopen_with_plan(request, timeout):
        params = parse_qs(urlparse(request.full_url).query)
        assert params["search_query"] == [planned_query]
        return FakeHTTPResponse()

    monkeypatch.setattr(arxiv_tools, "urlopen", fake_urlopen_with_plan)

    state = AgentState(
        topic="RLHF reasoning models",
        max_papers=2,
    )
    state.set_search_plan(
        SearchPlan(
            original_query=state.topic,
            core_terms=["RLHF"],
            context_terms=["reasoning"],
            categories=["cs.CL"],
            arxiv_query=planned_query,
            planner="llm",
        )
    )

    observation = search_arxiv_papers(state)

    assert observation["status"] == "success"
    assert observation["search_query"] == planned_query
