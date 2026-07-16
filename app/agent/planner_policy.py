from __future__ import annotations

from app.agent.finish_policy import FACTUAL_CUES
from app.agent.planner_models import CallToolAction
from app.agent.planner_state import PlannerState


def choose_policy_action(state: PlannerState) -> CallToolAction | None:
    """Return a deterministic first action for high-confidence planner cases."""

    if state.step_count > 0:
        return None
    if _should_probe_existing_kb(state.user_request):
        return CallToolAction(
            tool_name="retrieve_evidence",
            arguments={"query": state.user_request, "top_k": 5},
            decision_summary=(
                "Probe indexed knowledge-base evidence before discovering or "
                "preparing new papers."
            ),
        )
    return None


def _should_probe_existing_kb(user_request: str) -> bool:
    text = user_request.lower()
    if _is_listing_request(text):
        return False
    if not any(cue in text for cue in FACTUAL_CUES):
        return False
    if any(cue in text for cue in ("knowledge base", "already stored", "existing kb")):
        return True
    if "if" in text and _is_discovery_request(text):
        return True
    return not _is_discovery_request(text)


def _is_listing_request(text: str) -> bool:
    stripped = text.strip()
    return stripped.startswith(("list ", "show ", "browse ")) or any(
        cue in stripped
        for cue in (
            "list papers",
            "show papers",
            "browse papers",
            "recently added",
        )
    )


def _is_discovery_request(text: str) -> bool:
    stripped = text.strip()
    if stripped.startswith(("find ", "search ", "discover ")):
        return True
    return any(
        cue in stripped
        for cue in (
            "find papers",
            "search papers",
            "discover papers",
            "new papers",
            "papers about",
        )
    )
