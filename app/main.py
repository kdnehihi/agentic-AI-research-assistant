from app.agent.runner import AgentRunner
from app.agent.state import AgentState
from app.llm.client import OpenAILLMClient
from app.tools.report_tools import generate_report_from_abstracts
from app.tools.llm_summary_tools import summarize_papers_with_llm
from app.tools.registry import ToolRegistry

SEARCH_AND_FILTER_WORKFLOW = [
    "search_arxiv_papers",
    "deduplicate_papers",
    "rank_papers_by_similarity",
    "filter_relevant_papers",
]

TOPIC = "Temporal analysis of mental health self-disclosure on Reddit: title-body framing, diagnosis uncertainty, and early linguistic signals before disorder disclosure."
MAX_PAPERS = 3


def main():
    state = AgentState(
        topic=TOPIC,
        max_papers=MAX_PAPERS,
    )

    registry = ToolRegistry()
    runner = AgentRunner(state=state, registry=registry)

    runner.run_workflow(workflow=SEARCH_AND_FILTER_WORKFLOW)
    summarize_papers_with_llm(
        state=state,
        llm_client=OpenAILLMClient(),
    )
    generate_report_from_abstracts(state)

    print("\n===== FINAL REPORT =====\n")
    print(state.report)


if __name__ == "__main__":
    main()
