from app.agent.state import AgentState
from app.tools import arxiv_tools
from app.tools.arxiv_tools import (
    _build_arxiv_search_query,
    _parse_arxiv_response,
    search_arxiv_papers,
)


FAKE_ARXIV_XML = b"""
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2401.12345v1</id>
    <title>
      RLHF for Reasoning Models
    </title>
    <summary>
      This paper studies reinforcement learning from human feedback
      for reasoning-centric language models.
    </summary>
    <published>2026-06-20T12:00:00Z</published>
    <author>
      <name>Alice Nguyen</name>
    </author>
    <author>
      <name>Bob Chen</name>
    </author>
  </entry>

  <entry>
    <id>http://arxiv.org/abs/2401.67890v1</id>
    <title>RLVR and Verifiable Rewards</title>
    <summary>
      This paper explores verifiable rewards for mathematical reasoning.
    </summary>
    <published>2026-06-21T09:30:00Z</published>
    <author>
      <name>Carlos Smith</name>
    </author>
  </entry>
</feed>
"""


def test_parse_arxiv_response_converts_xml_to_papers():
    papers = _parse_arxiv_response(FAKE_ARXIV_XML)

    assert len(papers) == 2

    first = papers[0]
    assert first.title == "RLHF for Reasoning Models"
    assert first.paper_id == "arxiv:2401.12345v1"
    assert first.authors == ["Alice Nguyen", "Bob Chen"]
    assert first.source == "arxiv"
    assert first.url == "http://arxiv.org/abs/2401.12345v1"
    assert first.published_date == "2026-06-20"
    assert "human feedback" in first.abstract


def test_build_arxiv_search_query_expands_rlhf_and_rlvr_terms():
    search_query = _build_arxiv_search_query("RLHF RLVR reasoning models")

    assert "ti:RLHF OR abs:RLHF" in search_query
    assert "ti:RLVR OR abs:RLVR" in search_query
    assert 'ti:"reinforcement learning from human feedback"' in search_query
    assert 'abs:"verifiable rewards"' in search_query
    assert "ti:reasoning OR abs:reasoning" in search_query
    assert "cat:cs.CL" in search_query
    assert "all:reasoning" not in search_query


def test_build_arxiv_search_query_compacts_long_user_prompt():
    user_prompt = (
        "I want papers about RAG systems that reduce hallucination and include "
        "evaluation methods for retrieval augmented generation in language models."
    )

    search_query = _build_arxiv_search_query(user_prompt)

    assert "RAG" in search_query
    assert "retrieval augmented generation" in search_query
    assert "hallucination" in search_query
    assert "evaluation" in search_query
    assert "ti:" in search_query
    assert "abs:" in search_query
    assert "cat:cs.CL" in search_query
    assert user_prompt not in search_query


def test_build_arxiv_search_query_disambiguates_transformer_as_ai_topic():
    search_query = _build_arxiv_search_query("find 5 latest paper about transformer")

    assert "ti:transformer OR abs:transformer" in search_query
    assert 'ti:"transformer model"' in search_query
    assert "ti:attention OR abs:attention" in search_query
    assert 'ti:"neural network"' in search_query
    assert "cat:cs.CL" in search_query


def test_build_arxiv_search_query_uses_recency_terms_only_for_sorting_context():
    search_query = _build_arxiv_search_query("give me 3 newest papers about RAG")

    assert "newest" not in search_query
    assert "recent" not in search_query
    assert "ti:RAG OR abs:RAG" in search_query
    assert "retrieval augmented generation" in search_query


def test_build_arxiv_search_query_adds_submitted_date_for_explicit_year():
    search_query = _build_arxiv_search_query(
        "find 5 latest papers in 2026 about transformer language models"
    )

    assert "submittedDate:[202601010000 TO 202612312359]" in search_query


def test_arxiv_sort_by_uses_submitted_date_for_newest_queries():
    assert arxiv_tools._arxiv_sort_by("give me 3 newest papers about RAG") == (
        "submittedDate"
    )
    assert arxiv_tools._arxiv_sort_by("papers about RAG") == "relevance"


def test_arxiv_timeout_seconds_uses_environment_override(monkeypatch):
    monkeypatch.setenv("ARXIV_TIMEOUT_SECONDS", "3")

    assert arxiv_tools._arxiv_timeout_seconds() == 3
