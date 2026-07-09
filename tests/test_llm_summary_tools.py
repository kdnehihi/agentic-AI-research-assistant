from app.agent.state import AgentState, Paper
from app.llm.fake_llm import FakeLLMClient
from app.tools.llm_summary_tools import summarize_papers_with_llm


def test_summarize_papers_with_llm_updates_paper_summaries():
    state = AgentState(topic="RLHF reasoning", max_papers=1)
    paper = Paper(
        paper_id="arxiv:1",
        title="Reasoning Paper",
        authors=["Alice"],
        source="arxiv",
        url="https://arxiv.org/abs/1",
        abstract="This paper studies RLHF for reasoning models.",
    )
    state.set_selected_papers([paper])

    observation = summarize_papers_with_llm(
        state=state,
        llm_client=FakeLLMClient(),
    )

    assert observation["status"] == "success"
    assert observation["num_summaries"] == 1
    assert len(state.paper_summaries) == 1
    assert state.paper_summaries[0].paper_id == "arxiv:1"
    assert state.paper_summaries[0].based_on == "abstract_only"
    assert (
        state.paper_summaries[0].detailed_summary
        == "This is a fake summary of the papers."
    )
