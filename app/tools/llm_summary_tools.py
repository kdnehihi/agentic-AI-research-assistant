from __future__ import annotations

from typing import Any

from app.agent.state import AgentState, PaperSummary
from app.llm.client import LLMClient
from app.llm.fake_llm import FakeLLMClient


def summarize_papers_with_llm(
    state: AgentState,
    llm_client: LLMClient | None = None,
) -> dict[str, Any]:
    """
    Generate one LLM-based summary for each selected paper using its abstract.

    This tool currently summarizes abstracts only.
    It does not read full paper PDFs or HTML.
    """
    llm_client = llm_client or FakeLLMClient()

    papers = state.selected_papers or state.candidate_papers

    if not papers:
        state.set_paper_summaries([])
        return {
            "status": "partial_success",
            "num_summaries": 0,
            "summary": "No papers available to summarize.",
        }

    summaries: list[PaperSummary] = []

    for paper in papers:
        prompt = _build_abstract_summary_prompt(
            title=paper.title,
            abstract=paper.abstract or "",
            topic=state.topic,
        )

        llm_output = llm_client.generate(prompt).strip()

        summaries.append(
            PaperSummary(
                paper_id=paper.paper_id,
                title=paper.title,
                one_sentence_summary=llm_output,
                detailed_summary=llm_output,
                based_on="abstract_only",
            )
        )

    state.set_paper_summaries(summaries)

    return {
        "status": "success",
        "num_summaries": len(summaries),
        "summary": f"Generated {len(summaries)} LLM summaries from abstracts.",
    }


def _build_abstract_summary_prompt(
    title: str,
    abstract: str,
    topic: str,
) -> str:
    return f"""
You are an AI research assistant.

The user is researching this topic:
{topic}

Summarize the following paper based only on its title and abstract.

Title:
{title}

Abstract:
{abstract}

Write a concise summary in 2-3 sentences.
Focus on:
- the main problem
- the proposed method or idea
- why it is relevant to the user's topic

Do not invent details that are not present in the abstract.
""".strip()
