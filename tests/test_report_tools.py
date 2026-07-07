from app.agent.state import AgentState, Paper
from app.tools.report_tools import (
    generate_report_from_abstracts,
    summarize_papers_from_abstracts,
)


def test_summarize_papers_from_abstracts_uses_first_two_sentences():
    state = AgentState(topic="RLHF reasoning", max_papers=1)
    paper = Paper(
        paper_id="arxiv:1",
        title="Reasoning Paper",
        authors=["Alice"],
        source="arxiv",
        url="https://arxiv.org/abs/1",
        abstract=(
            "First sentence. Second sentence! Third sentence should not appear."
        ),
    )
    state.set_selected_papers([paper])

    observation = summarize_papers_from_abstracts(state)

    assert observation["status"] == "success"
    assert len(state.paper_summaries) == 1
    assert state.paper_summaries[0].one_sentence_summary == "First sentence."
    assert (
        state.paper_summaries[0].detailed_summary
        == "First sentence. Second sentence!"
    )


def test_generate_report_from_abstracts_uses_abstract_summary():
    state = AgentState(topic="RLHF reasoning", max_papers=1)
    paper = Paper(
        paper_id="arxiv:1",
        title="Reasoning Paper",
        authors=["Alice"],
        source="arxiv",
        url="https://arxiv.org/abs/1",
        abstract="Sentence one. Sentence two. Sentence three.",
    )
    state.set_selected_papers([paper])

    observation = generate_report_from_abstracts(state)

    assert observation["status"] == "success"
    assert state.report is not None
    assert "Sentence one. Sentence two." in state.report
    assert "Sentence three." not in state.report
