from __future__ import annotations

from scripts.dynamic_smoke_utils import run_live_dynamic_smoke


DEFAULT_REQUEST = (
    "Use papers already stored in the knowledge base to answer this question: "
    "what are the main limitations or open research directions for xMemory? "
    "Prefer retrieval from indexed papers. Only discover new papers if the "
    "knowledge base does not contain anything useful."
)


def main() -> None:
    run_live_dynamic_smoke(
        scenario_name="existing_kb_answer",
        default_request=DEFAULT_REQUEST,
        default_max_steps=8,
    )


if __name__ == "__main__":
    main()
