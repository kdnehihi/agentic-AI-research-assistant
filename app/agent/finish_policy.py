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

    if state.request_intent is not None:
        return _validate_intent_finish(state)

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


def _validate_intent_finish(state: PlannerState) -> tuple[bool, str | None]:
    intent = state.request_intent
    if intent is None:
        return False, "Finish requires a classified request intent."

    if state.retrieved_evidence_ids:
        return True, None
    if state.report_available or state.runtime_state.report:
        return True, None
    if state.summary_paper_ids or state.runtime_state.paper_summaries:
        return True, None

    if intent.finish_condition == "paper_metadata" and (
        state.known_paper_ids or state.saved_paper_ids or state.retrievable_paper_ids
    ):
        return True, None
    if intent.finish_condition == "stored_metadata" and (
        state.saved_paper_ids or state.known_paper_ids
    ):
        return True, None

    if intent.finish_condition == "paper_metadata":
        return False, "Finish requires discovered or stored paper metadata."
    if intent.finish_condition == "stored_metadata":
        return False, "Finish requires stored paper metadata."
    if intent.finish_condition == "retrieved_evidence":
        return (
            False,
            "Finish requires retrieved evidence, summaries, or a generated report.",
        )
    if intent.finish_condition == "paper_summary":
        return False, "Finish requires paper summaries."
    if intent.finish_condition == "report":
        return False, "Finish requires a generated report."
    return False, "Finish requires at least one usable artifact."
