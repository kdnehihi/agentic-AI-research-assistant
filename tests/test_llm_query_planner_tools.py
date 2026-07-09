from app.agent.state import AgentState
from app.tools.llm_query_planner_tools import plan_arxiv_search_query_with_llm


class JsonPlanLLM:
    def generate(self, prompt: str, **kwargs):
        return """
        {
          "core_terms": ["RLHF", "RLVR", "verifiable rewards"],
          "context_terms": ["reasoning", "large language model"],
          "categories": ["cs.CL", "cs.AI", "stat.ML"]
        }
        """


class BrokenLLM:
    def generate(self, prompt: str, **kwargs):
        raise RuntimeError("llm unavailable")


def test_plan_arxiv_search_query_with_llm_sets_search_plan():
    state = AgentState(topic="RLHF RLVR reasoning models", max_papers=3)

    observation = plan_arxiv_search_query_with_llm(
        state=state,
        llm_client=JsonPlanLLM(),
    )

    assert observation["status"] == "success"
    assert observation["planner"] == "llm"
    assert state.search_plan is not None
    assert state.search_plan.core_terms == ["RLHF", "RLVR", "verifiable rewards"]
    assert "ti:RLHF OR abs:RLHF" in state.search_plan.arxiv_query
    assert "ti:reasoning OR abs:reasoning" in state.search_plan.arxiv_query
    assert "cat:cs.CL" in state.search_plan.arxiv_query


def test_plan_arxiv_search_query_with_llm_falls_back_when_llm_fails():
    state = AgentState(topic="RLHF RLVR reasoning models", max_papers=3)

    observation = plan_arxiv_search_query_with_llm(
        state=state,
        llm_client=BrokenLLM(),
    )

    assert observation["status"] == "partial_success"
    assert observation["planner"] == "rule_based"
    assert state.search_plan is not None
    assert state.search_plan.core_terms == [state.topic]
    assert "cat:cs.CL" in state.search_plan.arxiv_query
    assert observation["error"] == "llm unavailable"
