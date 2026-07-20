from __future__ import annotations

from app.agent.finish_policy import FACTUAL_CUES
from app.agent.planner_models import CallToolAction, FinishAction, PlannerDecision
from app.agent.planner_state import PlannerState


def choose_policy_action(state: PlannerState) -> PlannerDecision | None:
    """Return a deterministic action for high-confidence planner cases."""

    finish = _finish_when_discovery_artifacts_are_enough(state)
    if finish is not None:
        return finish

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


def _finish_when_discovery_artifacts_are_enough(
    state: PlannerState,
) -> FinishAction | None:
    """Stop discovery-only requests once paper artifacts are available."""

    if not _is_discovery_only_request(state.user_request):
        return None
    paper_ids = state.known_paper_ids or state.saved_paper_ids or state.retrievable_paper_ids
    if not paper_ids:
        return None
    return FinishAction(
        answer_task=state.user_request,
        decision_summary=(
            "The request only asks to find papers, and discovered paper metadata "
            "is already available."
        ),
    )


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


def _is_discovery_only_request(user_request: str) -> bool:
    text = user_request.lower()
    if not _is_discovery_request(text):
        return False
    factual_cues_without_discovery = tuple(
        cue for cue in FACTUAL_CUES if cue not in {"what"}
    )
    return not any(cue in text for cue in factual_cues_without_discovery)
