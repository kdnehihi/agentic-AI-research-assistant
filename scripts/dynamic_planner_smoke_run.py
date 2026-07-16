from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from app.agent.dynamic_runner import DynamicAgentRunner
from app.agent.executor import ToolExecutor
from app.agent.grounded_answer import GroundedAnswerService
from app.agent.planner import Planner
from app.agent.planner_models import CallToolAction, FinishAction
from app.llm.client import OpenAILLMClient
from app.tools.registry import ToolRegistry


class ScriptedPlanner:
    """Small no-LLM planner for local smoke tests when API quota is unavailable."""

    def __init__(self, decisions):
        self.decisions = list(decisions)

    def decide(self, state, tool_specs):
        if not self.decisions:
            return FinishAction(
                answer_task=state.user_request,
                decision_summary="Scripted planner has no more tool calls.",
            )
        return self.decisions.pop(0)


class EchoAnswerService(GroundedAnswerService):
    """Final answer service that avoids LLM calls in fake smoke mode."""

    def __init__(self):
        pass

    def generate(self, *, state, answer_task):
        return {
            "answer_task": answer_task,
            "source": "fake_smoke",
            "known_paper_ids": state.known_paper_ids,
            "saved_paper_ids": state.saved_paper_ids,
            "retrievable_paper_ids": state.retrievable_paper_ids,
            "retrieved_evidence_ids": state.retrieved_evidence_ids,
            "latest_observation": (
                _compact_observation(state.latest_observation.model_dump(mode="json"))
                if state.latest_observation
                else None
            ),
        }


class LocalRetrievalRegistry(ToolRegistry):
    """Registry shim that serves retrieve_evidence from local chunk files."""

    def execute(self, tool_name, state, **kwargs):
        if tool_name != "retrieve_evidence":
            return super().execute(tool_name, state, **kwargs)

        paper_ids = kwargs.get("paper_ids") or []
        top_k = kwargs.get("top_k") or 3
        evidence = []
        missing = []
        for paper_id in paper_ids:
            chunk_path = _chunk_path_for(paper_id)
            if not chunk_path.exists():
                missing.append(paper_id)
                continue
            evidence.extend(_read_local_evidence(chunk_path, limit=top_k - len(evidence)))
            if len(evidence) >= top_k:
                break

        if missing and not evidence:
            return {
                "status": "failed",
                "error_type": "paper_not_retrievable",
                "missing_paper_ids": missing,
                "evidence": [],
                "summary": "No local chunks were found for requested paper ids.",
            }

        return {
            "status": "partial_success" if missing else "success",
            "query": kwargs.get("query"),
            "retrieved": len(evidence),
            "evidence": evidence,
            "missing_paper_ids": missing,
            "summary": f"Retrieved {len(evidence)} local evidence chunks.",
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the dynamic planner smoke test.")
    parser.add_argument("request", help="User research request")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument(
        "--fake-plan",
        choices=["discovery", "list", "retrieve"],
        help="Run a scripted no-LLM planner path.",
    )
    parser.add_argument(
        "--paper-id",
        action="append",
        dest="paper_ids",
        default=[],
        help="Paper id for fake retrieve mode. Can be passed more than once.",
    )
    parser.add_argument(
        "--local-retrieval",
        action="store_true",
        help="For --fake-plan retrieve, read evidence from local chunks.jsonl files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print the full final payload instead of a compact summary.",
    )
    args = parser.parse_args()

    executor = None
    if args.fake_plan:
        planner = ScriptedPlanner(_fake_decisions(args.fake_plan, args))
        answer_service = EchoAnswerService()
        if args.local_retrieval:
            executor = ToolExecutor(registry=LocalRetrievalRegistry())
    else:
        llm = OpenAILLMClient()
        planner = Planner(llm)
        answer_service = GroundedAnswerService(llm)

    runner = DynamicAgentRunner(
        planner=planner,
        executor=executor,
        answer_service=answer_service,
    )
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
    if args.verbose:
        print(json.dumps(state.final_answer, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(_compact_final_answer(state.final_answer), indent=2, ensure_ascii=False))


def _fake_decisions(fake_plan: str, args) -> list:
    if fake_plan == "list":
        return [
            CallToolAction(
                tool_name="list_papers",
                arguments={"limit": 3},
                decision_summary="List stored papers without calling an LLM.",
            ),
            FinishAction(
                answer_task="Return the listed stored papers.",
                decision_summary="Listing observation is enough for this smoke test.",
            ),
        ]
    if fake_plan == "retrieve":
        if not args.paper_ids:
            raise SystemExit("--fake-plan retrieve requires at least one --paper-id.")
        return [
            CallToolAction(
                tool_name="retrieve_evidence",
                arguments={
                    "query": args.request,
                    "paper_ids": args.paper_ids,
                    "top_k": 3,
                },
                decision_summary="Retrieve evidence for explicit paper ids.",
            ),
            FinishAction(
                answer_task=args.request,
                decision_summary="Retrieved evidence is enough for this smoke test.",
            ),
        ]
    return [
        CallToolAction(
            tool_name="discover_papers",
            arguments={
                "user_query": args.request,
                "max_results": 10,
                "max_selected": 3,
                "exclude_seen": True,
            },
            decision_summary="Discover papers without using an LLM planner.",
        ),
        FinishAction(
            answer_task="Return the discovered paper ids.",
            decision_summary="Discovery results are enough for this smoke test.",
        ),
    ]


def _chunk_path_for(paper_id: str) -> Path:
    paper_dir = paper_id.replace(":", "_").replace(".", "_")
    return Path("data") / "papers" / paper_dir / "chunks.jsonl"


def _read_local_evidence(chunk_path: Path, *, limit: int) -> list[dict]:
    evidence = []
    if limit <= 0:
        return evidence
    with chunk_path.open() as handle:
        for line in handle:
            record = json.loads(line)
            evidence.append(
                {
                    "chunk_id": record.get("chunk_id"),
                    "paper_id": record.get("paper_id"),
                    "title": record.get("title"),
                    "section": record.get("section"),
                    "section_group": record.get("section_group"),
                    "page": record.get("page"),
                    "text": record.get("text"),
                    "semantic_score": 0.0,
                    "lexical_score": None,
                    "metadata_score": 0.0,
                    "final_score": 0.0,
                    "rank": len(evidence) + 1,
                }
            )
            if len(evidence) >= limit:
                break
    return evidence


def _compact_final_answer(final_answer):
    if not isinstance(final_answer, dict):
        return final_answer
    compact = dict(final_answer)
    latest = compact.get("latest_observation")
    if isinstance(latest, dict):
        compact["latest_observation"] = _compact_observation(latest)
    evidence_chunks = compact.get("evidence_chunks")
    if isinstance(evidence_chunks, list):
        compact["evidence_chunks"] = [
            _compact_evidence_chunk(item)
            for item in evidence_chunks
            if isinstance(item, dict)
        ]
    return compact


def _compact_evidence_chunk(item: dict) -> dict:
    return {
        "evidence_id": item.get("evidence_id"),
        "chunk_id": item.get("chunk_id"),
        "paper_id": item.get("paper_id"),
        "title": item.get("title"),
        "section": item.get("section"),
        "section_group": item.get("section_group"),
        "rank": item.get("rank"),
        "final_score": item.get("final_score"),
    }


def _compact_observation(observation: dict) -> dict:
    result = observation.get("result") or {}
    evidence = result.get("evidence") or []
    compact_result = dict(result)
    if evidence:
        compact_result["evidence"] = [
            {
                "chunk_id": item.get("chunk_id"),
                "paper_id": item.get("paper_id"),
                "title": item.get("title"),
                "section": item.get("section"),
                "section_group": item.get("section_group"),
                "rank": item.get("rank"),
                "final_score": item.get("final_score"),
            }
            for item in evidence
            if isinstance(item, dict)
        ]

    state_changes = observation.get("state_changes") or {}
    compact_state_changes = {
        key: value
        for key, value in state_changes.items()
        if key != "retrieved_evidence_added"
    }

    return {
        "tool_name": observation.get("tool_name"),
        "status": observation.get("status"),
        "summary": observation.get("summary"),
        "result": compact_result,
        "state_changes": compact_state_changes,
        "error_type": observation.get("error_type"),
        "retryable": observation.get("retryable"),
    }


if __name__ == "__main__":
    main()
