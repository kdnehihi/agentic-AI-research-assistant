from app.agent.state import AgentState, Paper
from app.llm.fake_llm import FakeLLMClient
from app.tools.llm_summary_tools import summarize_papers_with_llm


class FailingLLMClient:
    def generate(self, prompt: str, **kwargs) -> str:
        raise RuntimeError("quota exceeded")


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


def test_summarize_papers_with_llm_falls_back_to_abstract_summary():
    state = AgentState(topic="RLHF reasoning", max_papers=1)
    paper = Paper(
        paper_id="arxiv:1",
        title="Reasoning Paper",
        authors=["Alice"],
        source="arxiv",
        url="https://arxiv.org/abs/1",
        abstract="First sentence. Second sentence. Third sentence.",
    )
    state.set_selected_papers([paper])

    observation = summarize_papers_with_llm(
        state=state,
        llm_client=FailingLLMClient(),
    )

    assert observation["status"] == "partial_success"
    assert observation["fallback_count"] == 1
    assert len(state.paper_summaries) == 1
    assert state.paper_summaries[0].detailed_summary == (
        "First sentence. Second sentence."
    )


def test_summarize_papers_with_llm_only_uses_selected_papers():
    state = AgentState(topic="RLHF reasoning", max_papers=1)
    selected = Paper(
        paper_id="arxiv:selected",
        title="Selected Paper",
        source="arxiv",
        url="https://arxiv.org/abs/selected",
        abstract="Selected abstract.",
    )
    candidate = Paper(
        paper_id="arxiv:candidate",
        title="Candidate Paper",
        source="arxiv",
        url="https://arxiv.org/abs/candidate",
        abstract="Candidate abstract.",
    )
    state.set_candidate_papers([selected, candidate])
    state.set_selected_papers([selected])

    observation = summarize_papers_with_llm(
        state=state,
        llm_client=FakeLLMClient(),
    )

    assert observation["status"] == "success"
    assert len(state.paper_summaries) == 1
    assert state.paper_summaries[0].paper_id == "arxiv:selected"
