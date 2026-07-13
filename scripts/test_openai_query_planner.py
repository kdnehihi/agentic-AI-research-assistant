from app.agent.state import AgentState
from app.llm.client import OpenAILLMClient
from app.tools.llm_query_planner_tools import plan_arxiv_search_query_with_llm

TOPIC = "RLHF RLVR reasoning models"


def main():
    """Smoke-test OpenAI query planning for the default topic."""

    state = AgentState(topic=TOPIC, max_papers=3)
    llm_client = OpenAILLMClient()

    observation = plan_arxiv_search_query_with_llm(
        state=state,
        llm_client=llm_client,
    )

    print("Status:", observation["status"])
    print("Planner:", observation["planner"])
    print("Core Terms:", observation["core_terms"])
    print("Context Terms:", observation["context_terms"])
    print("Categories:", observation["categories"])
    print("arXiv Query:", observation["search_query"])
    if observation.get("error"):
        print("Error:", observation["error"])


if __name__ == "__main__":
    main()
