from app.agent.runner import AgentRunner, ARXIV_WORKFLOW
from app.agent.state import AgentState
from app.tools.registry import ToolRegistry


def main():
    state = AgentState(
        topic="RLHF RLVR reasoning models",
        max_papers=3,
    )

    registry = ToolRegistry()
    runner = AgentRunner(state=state, registry=registry)

    runner.run_workflow(workflow=ARXIV_WORKFLOW)

    print("\n===== FINAL REPORT =====\n")
    print(state.report)


if __name__ == "__main__":
    main()