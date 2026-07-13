from app.agent.state import AgentState, Paper
from app.llm.client import OpenAILLMClient
from app.tools.llm_summary_tools import summarize_papers_with_llm


def main():
    state = AgentState(topic="RLHF RLVR reasoning models", max_papers=1)
    state.set_selected_papers(
        [
            Paper(
                paper_id="demo:1",
                title="RLHF for Reasoning Models",
                source="demo",
                url="https://example.com/demo",
                abstract=(
                    "This paper studies reinforcement learning from human "
                    "feedback for improving reasoning behavior in language "
                    "models. It analyzes how preference optimization and reward "
                    "models affect multi-step reasoning performance."
                ),
            )
        ]
    )

    observation = summarize_papers_with_llm(
        state=state,
        llm_client=OpenAILLMClient(),
    )

    print("Status:", observation["status"])
    print("Fallback Count:", observation["fallback_count"])
    if observation.get("fallback_errors"):
        print("Fallback Errors:", observation["fallback_errors"])
    print("Summary:", state.paper_summaries[0].detailed_summary)


if __name__ == "__main__":
    main()
