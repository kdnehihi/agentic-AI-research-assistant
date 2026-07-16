from __future__ import annotations

import argparse
import json
import time

from app.agent.dynamic_runner import DynamicAgentRunner
from app.agent.executor import ToolExecutor
from app.agent.grounded_answer import GroundedAnswerService
from app.agent.planner import Planner
from app.llm.client import OpenAILLMClient
from scripts.dynamic_planner_smoke_run import _compact_final_answer


def run_live_dynamic_smoke(
    *,
    scenario_name: str,
    default_request: str,
    default_max_steps: int = 10,
) -> None:
    """Run a live dynamic planner scenario with compact trace output."""

    parser = argparse.ArgumentParser(description=f"Smoke test: {scenario_name}")
    parser.add_argument(
        "request",
        nargs="?",
        default=default_request,
        help="Optional override for the default smoke-test request.",
    )
    parser.add_argument("--max-steps", type=int, default=default_max_steps)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print the full final payload instead of a compact summary.",
    )
    args = parser.parse_args()

    llm = OpenAILLMClient()
    runner = DynamicAgentRunner(
        planner=Planner(llm),
        executor=ToolExecutor(),
        answer_service=GroundedAnswerService(llm),
    )

    print(f"scenario={scenario_name}")
    print(f"request={args.request}")
    print()

    started_at = time.perf_counter()
    state = runner.run(user_request=args.request, max_steps=args.max_steps)
    elapsed_seconds = time.perf_counter() - started_at

    for record in state.tool_history:
        print(f"step={record.step}")
        print(f"tool={record.decision.tool_name}")
        if record.latency_ms is not None:
            print(f"latency_ms={record.latency_ms:.1f}")
        print(f"arguments={record.decision.arguments}")
        print(f"observation={record.observation.summary}")
        print()

    print(f"final_status={state.status}")
    print(f"elapsed_seconds={elapsed_seconds:.2f}")
    print(f"last_error={state.last_error}")
    print("final_answer=")
    payload = state.final_answer if args.verbose else _compact_final_answer(state.final_answer)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
