from __future__ import annotations

from typing import Any

from app.agent.planner_models import ObservationStatus, ToolObservation


MAX_TEXT_CHARS = 1200
MAX_LIST_ITEMS = 10
SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "environment",
    "env",
    "password",
    "secret",
    "stack_trace",
    "traceback",
}


class ObservationFactory:
    """Convert raw domain tool results into compact planner observations."""

    def from_tool_result(
        self,
        *,
        tool_name: str,
        raw_result: dict[str, Any],
    ) -> ToolObservation:
        """Normalize status, compact payload, and extract state changes."""

        status = _normalize_status(raw_result)
        error_type = _error_type(raw_result)
        if status == "tool_error" and _is_prerequisite_error(raw_result):
            status = "prerequisite_missing"

        return ToolObservation(
            tool_name=tool_name,
            status=status,
            summary=_summary(tool_name, raw_result, status),
            result=_compact_result(tool_name, raw_result),
            state_changes=_state_changes(tool_name, raw_result),
            error_type=error_type,
            retryable=_retryable(raw_result, status),
        )

    def from_error(
        self,
        *,
        tool_name: str,
        status: ObservationStatus,
        summary: str,
        error_type: str,
        retryable: bool = False,
    ) -> ToolObservation:
        """Build an observation for executor-side validation or runtime errors."""

        return ToolObservation(
            tool_name=tool_name,
            status=status,
            summary=summary,
            error_type=error_type,
            retryable=retryable,
        )


def _normalize_status(raw_result: dict[str, Any]) -> str:
    raw_status = raw_result.get("status")
    if raw_status == "success":
        return "success"
    if raw_status == "partial_success":
        return "partial_success"
    if raw_status == "skipped":
        return "success"
    if raw_status in {"failed", "error", "failure"}:
        return "tool_error"
    return "tool_error"


def _is_prerequisite_error(raw_result: dict[str, Any]) -> bool:
    error_type = _error_type(raw_result) or ""
    if error_type in {
        "paper_not_retrievable",
        "missing_paper_metadata",
        "missing_prerequisite",
    }:
        return True
    if raw_result.get("missing_paper_ids"):
        return True
    failed = raw_result.get("failed") or []
    return any((item or {}).get("error_type") == "missing_prerequisite" for item in failed)


def _error_type(raw_result: dict[str, Any]) -> str | None:
    error_type = raw_result.get("error_type")
    if isinstance(error_type, str):
        return error_type
    failed = raw_result.get("failed") or []
    if failed and isinstance(failed[0], dict):
        nested = failed[0].get("error_type")
        if isinstance(nested, str):
            return nested
    return None


def _retryable(raw_result: dict[str, Any], status: str) -> bool:
    if status == "prerequisite_missing":
        return True
    failed = raw_result.get("failed") or []
    if any(bool((item or {}).get("retryable")) for item in failed if isinstance(item, dict)):
        return True
    return bool(raw_result.get("retryable", False))


def _summary(tool_name: str, raw_result: dict[str, Any], status: str) -> str:
    raw_summary = raw_result.get("summary") or raw_result.get("message")
    if isinstance(raw_summary, str) and raw_summary.strip():
        return _truncate(raw_summary.strip(), 500)
    return f"{tool_name} returned planner status {status}."


def _compact_result(tool_name: str, raw_result: dict[str, Any]) -> dict[str, Any]:
    allowed_by_tool = {
        "discover_papers": [
            "planned_query",
            "candidate_paper_ids",
            "selected_paper_ids",
            "candidate_count",
            "selected_count",
            "excluded_seen_count",
        ],
        "list_papers": ["knowledge_base_id", "count", "papers"],
        "get_paper_metadata": ["papers", "missing_paper_ids"],
        "save_papers_to_kb": [
            "knowledge_base_id",
            "inserted_paper_ids",
            "updated_paper_ids",
            "already_present_paper_ids",
            "failed",
        ],
        "ensure_papers_retrievable": [
            "ready_paper_ids",
            "already_ready_paper_ids",
            "newly_fetched_paper_ids",
            "newly_extracted_paper_ids",
            "newly_chunked_paper_ids",
            "newly_embedded_paper_ids",
            "newly_indexed_paper_ids",
            "failed",
        ],
        "retrieve_evidence": [
            "query",
            "retrieved",
            "evidence",
            "missing_paper_ids",
            "message",
        ],
        "summarize_papers": [
            "paper_ids",
            "missing_paper_ids",
            "summary_mode",
            "summaries",
        ],
        "generate_paper_report": [
            "paper_ids",
            "missing_paper_ids",
            "report_type",
            "report",
        ],
    }
    keys = allowed_by_tool.get(tool_name, [])
    return {
        key: _compact_value(raw_result.get(key), key=key)
        for key in keys
        if key in raw_result
    }


def _state_changes(tool_name: str, raw_result: dict[str, Any]) -> dict[str, Any]:
    if tool_name == "discover_papers":
        return {
            "known_paper_ids_added": _ids(
                raw_result.get("selected_paper_ids"),
                raw_result.get("candidate_paper_ids"),
            )
        }
    if tool_name == "list_papers":
        paper_ids = [
            paper.get("paper_id")
            for paper in raw_result.get("papers", [])
            if isinstance(paper, dict) and paper.get("paper_id")
        ]
        return {
            "known_paper_ids_added": paper_ids,
            "saved_paper_ids_added": paper_ids,
        }
    if tool_name == "get_paper_metadata":
        papers = [paper for paper in raw_result.get("papers", []) if isinstance(paper, dict)]
        return {
            "saved_paper_ids_added": [
                paper["paper_id"]
                for paper in papers
                if paper.get("paper_id") and paper.get("exists_in_kb")
            ],
            "retrievable_paper_ids_added": [
                paper["paper_id"]
                for paper in papers
                if paper.get("paper_id") and paper.get("indexed")
            ],
        }
    if tool_name == "save_papers_to_kb":
        return {
            "saved_paper_ids_added": _ids(
                raw_result.get("inserted_paper_ids"),
                raw_result.get("updated_paper_ids"),
                raw_result.get("already_present_paper_ids"),
            )
        }
    if tool_name == "ensure_papers_retrievable":
        return {
            "retrievable_paper_ids_added": _ids(
                raw_result.get("ready_paper_ids"),
                raw_result.get("already_ready_paper_ids"),
            )
        }
    if tool_name == "retrieve_evidence":
        evidence = [item for item in raw_result.get("evidence", []) if isinstance(item, dict)]
        return {
            "retrieved_evidence_ids_added": [
                item["chunk_id"] for item in evidence if item.get("chunk_id")
            ],
            "retrieved_evidence_added": evidence,
        }
    if tool_name == "summarize_papers":
        return {"summary_paper_ids_added": list(raw_result.get("paper_ids") or [])}
    if tool_name == "generate_paper_report":
        return {
            "summary_paper_ids_added": list(raw_result.get("paper_ids") or []),
            "report_available": bool(raw_result.get("report")),
        }
    return {}


def _compact_value(value: Any, *, key: str) -> Any:
    if key == "raw":
        return None
    if _is_sensitive_key(key):
        return None
    if isinstance(value, str):
        return _truncate(value, MAX_TEXT_CHARS)
    if isinstance(value, list):
        return [_compact_value(item, key=key) for item in value[:MAX_LIST_ITEMS]]
    if isinstance(value, dict):
        return {
            item_key: _compact_value(item_value, key=item_key)
            for item_key, item_value in value.items()
            if not _is_sensitive_key(item_key) and item_key != "raw"
        }
    return value


def _is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    return any(sensitive in lowered for sensitive in SENSITIVE_KEYS)


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + " ..."


def _ids(*values: Any) -> list[str]:
    result: list[str] = []
    for value in values:
        if not value:
            continue
        for item in value:
            if isinstance(item, str) and item not in result:
                result.append(item)
    return result
