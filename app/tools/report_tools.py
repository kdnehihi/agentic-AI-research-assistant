from __future__ import annotations

import re

from app.agent.state import AgentState, Paper, PaperSummary


def summarize_papers_from_abstracts(state: AgentState) -> dict:
    """
    Create paper summaries from the first two sentences of each abstract.
    """
    papers = state.selected_papers or state.candidate_papers
    summaries: list[PaperSummary] = []
    if not papers:
        state.set_paper_summaries([])
        return {
            "status": "partial_success",
            "num_summaries": 0,
            "summary": "No papers available to summarize.",
        }
    for paper in papers:
        abstract_summary = _first_sentences(paper.abstract, sentence_count=2)
        summaries.append(
            PaperSummary(
                paper_id=paper.paper_id,
                title=paper.title,
                one_sentence_summary=_first_sentences(
                    paper.abstract,
                    sentence_count=1,
                ),
                detailed_summary=abstract_summary,
                based_on="abstract_only",
            )
        )

    state.set_paper_summaries(summaries)

    return {
        "status": "success",
        "num_summaries": len(summaries),
        "summary": f"Generated {len(summaries)} summaries from abstracts.",
    }


def generate_report_from_abstracts(state: AgentState) -> dict:
    """
    Generate a markdown report using selected papers and abstract summaries.
    """
    if not state.selected_papers:
        state.set_report("No relevant papers were selected.")

        return {
            "status": "partial_success",
            "summary": "No selected papers available for report.",
        }

    if not state.paper_summaries:
        summarize_papers_from_abstracts(state)

    summaries_by_id = {
        _paper_key(summary): summary
        for summary in state.paper_summaries
    }

    lines = [
        f"# Paper Research Report: {state.topic}",
        "",
        "## Selected Papers",
        "",
    ]

    for idx, paper in enumerate(state.selected_papers, start=1):
        paper_summary = summaries_by_id.get(_paper_key(paper))
        summary_text = (
            paper_summary.detailed_summary
            if paper_summary and paper_summary.detailed_summary
            else _first_sentences(paper.abstract, sentence_count=2)
        )

        lines.extend(
            [
                f"### {idx}. {paper.title}",
                f"- Authors: {', '.join(paper.authors)}",
                f"- Source: {paper.source}",
                f"- URL: {paper.url}",
                f"- Published Date: {paper.published_date or 'Unknown'}",
                f"- Score: {paper.score}",
                f"- Relevant Reasons: {', '.join(paper.relevant_reasons) or 'N/A'}",
                f"- Summary: {summary_text or 'No abstract available.'}",
                "",
            ]
        )

    report = "\n".join(lines)
    state.set_report(report)

    return {
        "status": "success",
        "summary": f"Generated report with {len(state.selected_papers)} papers.",
    }


def _first_sentences(text: str | None, sentence_count: int) -> str:
    if not text:
        return ""

    sentences = re.findall(r"[^.!?]+[.!?]?", text.strip())
    cleaned = [sentence.strip() for sentence in sentences if sentence.strip()]
    return " ".join(cleaned[:sentence_count])


def _paper_key(paper: Paper | PaperSummary) -> str:
    return paper.paper_id or paper.title
