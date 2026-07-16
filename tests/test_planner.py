from app.agent.planner import parse_planner_decision
from app.agent.planner_models import CallToolAction


def test_parse_planner_decision_accepts_tool_name_action_shorthand():
    decision = parse_planner_decision(
        """
        {
          "action": "list_papers",
          "arguments": {"limit": 5},
          "decision_summary": "Check stored papers first."
        }
        """
    )

    assert isinstance(decision, CallToolAction)
    assert decision.tool_name == "list_papers"
    assert decision.arguments == {"limit": 5}
