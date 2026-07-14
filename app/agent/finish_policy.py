from __future__ import annotations

from app.agent.planner_models import FinishAction
from app.agent.planner_state import PlannerState


FACTUAL_CUES = (
    "answer",
    "compare",
    "explain",
    "limitations",
    "method",
    "methods",
    "why",
    "what",
    "how",
    "summarize",
)
DISCOVERY_CUES = ("discover", "find", "search", "new papers", "papers about")
LISTING_CUES = ("list", "recently added", "stored", "knowledge base")


def validate_finish(state: PlannerState, decision: FinishAction) -> tuple[bool, str | None]:
    """Return whether a finish action has enough artifacts for v1."""

    request = state.user_request.lower()
    task = decision.answer_task.lower()
    text = f"{request} {task}"

    if state.retrieved_evidence_ids:
        return True, None
    if state.report_available or state.runtime_state.report:
        return True, None
    if state.summary_paper_ids or state.runtime_state.paper_summaries:
        return True, None

    if any(cue in text for cue in LISTING_CUES) and state.saved_paper_ids:
        return True, None
    if any(cue in text for cue in DISCOVERY_CUES) and state.known_paper_ids:
        return True, None

    if any(cue in text for cue in FACTUAL_CUES):
        return (
            False,
            "Finish requires retrieved evidence, summaries, or a generated report.",
        )
    return False, "Finish requires at least one usable artifact."

