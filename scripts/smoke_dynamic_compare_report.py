from __future__ import annotations

from scripts.dynamic_smoke_utils import run_live_dynamic_smoke


DEFAULT_REQUEST = (
    "Compare recent papers about agentic retrieval augmented generation, "
    "especially xMemory, ARAG, and multi-agent RAG filtering. Retrieve evidence "
    "when possible and produce a concise comparison of methods, results, and "
    "limitations."
)


def main() -> None:
    run_live_dynamic_smoke(
        scenario_name="compare_report",
        default_request=DEFAULT_REQUEST,
        default_max_steps=10,
    )


if __name__ == "__main__":
    main()
