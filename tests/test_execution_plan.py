from app.agent.execution_plan import (
    ExecutionPlan,
    LLMExecutionPlanGenerator,
    PlanStep,
    parse_execution_plan,
)
from app.agent.request_intent import RequestIntent


class FakePlanLLM:
    def __init__(self, response):
        self.response = response
        self.prompts = []

    def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return self.response


def test_parse_execution_plan_accepts_argument_sources():
    plan = parse_execution_plan(
        """
        ```json
        {
          "goal": "Answer from newly discovered papers.",
          "strategy": "Find, prepare, retrieve, then finish.",
          "steps": [
            {
              "step_id": "discover",
              "kind": "tool",
              "tool_name": "discover_papers",
              "arguments": {"user_query": "neural theorem proving"},
              "success_condition": "selected_paper_ids is not empty",
              "rationale": "Need candidate papers."
            },
            {
              "step_id": "prepare",
              "kind": "tool",
              "tool_name": "ensure_papers_retrievable",
              "argument_sources": {"paper_ids": "known_paper_ids"},
              "success_condition": "ready_paper_ids is not empty",
              "rationale": "Need indexed chunks."
            },
            {
              "step_id": "finish",
              "kind": "finish",
              "answer_task": "Answer the user.",
              "success_condition": "final answer returned",
              "rationale": "Evidence is available."
            }
          ]
        }
        ```
        """
    )

    assert plan.steps[1].argument_sources == {"paper_ids": "known_paper_ids"}
    assert plan.steps[0].status == "pending"


def test_llm_execution_plan_generator_includes_intent_and_tools():
    llm = FakePlanLLM(
        """
        {
          "goal": "Find papers.",
          "strategy": "Discover and finish.",
          "steps": [
            {
              "step_id": "discover",
              "kind": "tool",
              "tool_name": "discover_papers",
              "arguments": {"user_query": "neural theorem proving"},
              "success_condition": "selected papers exist",
              "rationale": "Need paper metadata."
            }
          ]
        }
        """
    )
    generator = LLMExecutionPlanGenerator(llm)
    intent = RequestIntent(
        task_type="discovery_only",
        topic="neural theorem proving",
        finish_condition="paper_metadata",
    )

    plan = generator.generate_plan(
        user_request="Find papers about neural theorem proving.",
        request_intent=intent,
        tool_specs=[],
    )

    assert isinstance(plan, ExecutionPlan)
    assert plan.steps == [
        PlanStep(
            step_id="discover",
            kind="tool",
            tool_name="discover_papers",
            arguments={"user_query": "neural theorem proving"},
            success_condition="selected papers exist",
            rationale="Need paper metadata.",
        )
    ]
    assert "neural theorem proving" in llm.prompts[0]
