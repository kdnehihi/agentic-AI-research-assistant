from __future__ import annotations

from scripts.dynamic_smoke_utils import run_live_dynamic_smoke


DEFAULT_REQUEST = (
    "Find a recent paper about long-term agent memory for research assistants. "
    "If the paper is not already in the knowledge base, discover it, prepare it "
    "for semantic retrieval, then answer: what did the paper discover?"
)


def main() -> None:
    run_live_dynamic_smoke(
        scenario_name="discover_prepare_answer",
        default_request=DEFAULT_REQUEST,
        default_max_steps=10,
    )


if __name__ == "__main__":
    main()
