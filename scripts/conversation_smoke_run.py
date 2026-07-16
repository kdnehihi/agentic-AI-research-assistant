from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.agent.executor import ToolExecutor
from app.agent.langgraph_runner import LangGraphAgentRunner
from app.agent.planner_eval import EvalAnswerService, EvalRegistry, ScriptedEvalPlanner
from app.agent.planner_models import CallToolAction, FinishAction
from app.conversations.service import ConversationAgentService
from app.conversations.sqlite_repository import SQLiteConversationRepository


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local conversation smoke test.")
    parser.add_argument(
        "--db-path",
        default="data/metadata/conversations.sqlite3",
        help="SQLite DB path for conversation history.",
    )
    args = parser.parse_args()

    repository = SQLiteConversationRepository(Path(args.db_path))
    planner = ScriptedEvalPlanner(
        [
            CallToolAction(
                tool_name="discover_papers",
                arguments={"user_query": "Agentic RAG", "max_results": 2, "max_selected": 2},
                decision_summary="Find Agentic RAG papers.",
            ),
            FinishAction(
                answer_task="Return discovered Agentic RAG papers.",
                decision_summary="Papers were found.",
            ),
            CallToolAction(
                tool_name="retrieve_evidence",
                arguments={
                    "query": "Compare the first one with the second one.",
                    "paper_ids": ["p1", "p2"],
                    "top_k": 2,
                },
                decision_summary="Use the active papers from the previous turn.",
            ),
            FinishAction(
                answer_task="Compare the previously discussed papers.",
                decision_summary="The KB probe retrieved comparison evidence.",
            ),
        ]
    )
    registry = EvalRegistry(
        {
            "discover_papers": [
                {
                    "status": "success",
                    "candidate_paper_ids": ["p1", "p2"],
                    "selected_paper_ids": ["p1", "p2"],
                    "candidate_count": 2,
                    "selected_count": 2,
                    "summary": "Discovered 2 candidate papers and selected 2 papers.",
                }
            ],
            "retrieve_evidence": [
                {
                    "status": "success",
                    "query": "Compare the first one with the second one.",
                    "retrieved": 2,
                    "evidence": [
                        {"chunk_id": "c1", "paper_id": "p1", "text": "p1 evidence"},
                        {"chunk_id": "c2", "paper_id": "p2", "text": "p2 evidence"},
                    ],
                    "summary": "Retrieved 2 evidence chunks.",
                }
            ],
        }
    )
    runner = LangGraphAgentRunner(
        planner=planner,
        executor=ToolExecutor(registry=registry),
        answer_service=EvalAnswerService(),
    )
    service = ConversationAgentService(
        conversation_repository=repository,
        run_repository=repository,
        runner=runner,
        summary_trigger_messages=100,
    )

    first = service.run_turn(user_content="Find papers about Agentic RAG.")
    second = service.run_turn(
        thread_id=first.thread.thread_id,
        user_content="Compare the first one with the second one.",
    )
    reopened = repository.get_thread(first.thread.thread_id)
    messages = repository.list_messages(first.thread.thread_id)

    print(
        json.dumps(
            {
                "thread_id": first.thread.thread_id,
                "reopened": reopened is not None,
                "message_count": len(messages),
                "first_run_id": first.run_id,
                "second_run_id": second.run_id,
                "second_active_paper_ids": second.planner_state.active_paper_ids,
                "messages": [
                    {
                        "role": message.role,
                        "sequence_number": message.sequence_number,
                        "metadata": message.metadata_json,
                    }
                    for message in messages
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
