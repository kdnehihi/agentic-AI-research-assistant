from __future__ import annotations

from typing import Any

from app.agent.state import AgentState, PaperSummary
from app.llm.client import LLMClient, create_default_llm_client
from app.tools.report_tools import first_sentences


def summarize_papers_with_llm(
    state: AgentState,
    llm_client: LLMClient | None = None,
) -> dict[str, Any]:
    """
    Generate one LLM-based summary for each selected paper using its abstract.

    This tool currently summarizes abstracts only.
    It does not read full paper PDFs or HTML.
    """
    llm_client = llm_client or create_default_llm_client()

    papers = state.selected_papers[: state.max_papers]

    if not papers:
        state.set_paper_summaries([])
        return {
            "status": "partial_success",
            "num_summaries": 0,
            "summary": "No papers available to summarize.",
        }

    summaries: list[PaperSummary] = []
    fallback_count = 0
    fallback_errors: list[str] = []

    for paper in papers:
        prompt = _build_abstract_summary_prompt(
            title=paper.title,
            abstract=paper.abstract or "",
            topic=state.topic,
        )

        try:
            llm_output = llm_client.generate(prompt).strip()
        except Exception as exc:
            fallback_count += 1
            fallback_errors.append(str(exc))
            llm_output = first_sentences(paper.abstract, sentence_count=2)

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
        "status": "partial_success" if fallback_count else "success",
        "num_summaries": len(summaries),
        "fallback_count": fallback_count,
        "fallback_errors": fallback_errors,
        "summary": (
            f"Generated {len(summaries)} summaries from abstracts "
            f"with {fallback_count} fallback summaries."
        ),
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
