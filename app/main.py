from app.agent.runner import AgentRunner, ARXIV_SEARCH_AND_FETCH_WORKFLOW
from app.agent.state import AgentState
from app.llm.client import OpenAILLMClient
from app.tools.report_tools import generate_report_from_abstracts
from app.tools.llm_summary_tools import summarize_papers_with_llm
from app.tools.knowledge_base_tools import save_selected_papers_to_kb
from app.tools.registry import ToolRegistry

SEARCH_AND_FILTER_WORKFLOW = ARXIV_SEARCH_AND_FETCH_WORKFLOW


TOPIC = "rag"
MAX_PAPERS = 5


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
    save_observation = save_selected_papers_to_kb(state)

    print("\n===== FINAL REPORT =====\n")
    print(state.report)
    print("\n===== KNOWLEDGE BASE SAVE REPORT =====\n")
    print(save_observation)


if __name__ == "__main__":
    main()
