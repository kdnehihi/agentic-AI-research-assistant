from __future__ import annotations

import argparse
import json
import sys

from app.agent.planner_eval import evaluate_planner_cases, summarize_eval_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run deterministic dynamic-planner contract evals."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full machine-readable eval summary.",
    )
    args = parser.parse_args()

    summary = summarize_eval_results(evaluate_planner_cases())
    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print(
            "planner_eval "
            f"passed={summary['passed']} failed={summary['failed']} "
            f"total={summary['total']} pass_rate={summary['pass_rate']:.3f}"
        )
        for result in summary["results"]:
            mark = "PASS" if result["passed"] else "FAIL"
            print(
                f"{mark} {result['name']} "
                f"status={result['status']} tools={result['tool_sequence']}"
            )
            for failure in result["failures"]:
                print(f"  - {failure}")

    if summary["failed"]:
        raise SystemExit(1)
    raise SystemExit(0)


if __name__ == "__main__":
    main()
