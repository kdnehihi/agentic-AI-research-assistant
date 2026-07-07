# Agentic AI Research Assistant

A small Python foundation for an agentic research assistant that can search,
deduplicate, rank, and report on academic papers. The current implementation
uses deterministic fake paper tools so the agent state and tool registry can be
tested before wiring in live APIs.

## What is included

- `AgentState` and related Pydantic models for tracking a research run
- normalized `Paper`, `PaperSummary`, and `ToolLog` objects
- a `ToolRegistry` for listing, checking, and executing tools
- fake paper tools for search, deduplication, ranking, and markdown report generation
- pytest coverage for state transitions, tool behavior, and registry execution

## Project layout

```text
app/
  agent/
    state.py              # Runtime state and data models
  tools/
    fake_paper_tools.py   # Deterministic fake research tools
    registry.py           # Tool registry and execution wrapper
tests/
  test_state.py
  test_fake_paper_tools.py
  test_registry.py
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run tests

```bash
pytest
```

To run only the registry tests:

```bash
pytest tests/test_registry.py
```

## Current workflow

The fake tool flow is:

1. `search_fake_papers`
2. `deduplicate_papers`
3. `rank_papers`
4. `generate_report_from_abstracts`

Each tool mutates `AgentState` and returns a small observation dictionary. This
keeps the control surface simple while the agent loop and real paper search
integrations are still being built.

## Next steps

- Add a real paper search tool backed by arXiv or another source
- Add an agent loop that records `ToolLog` entries for each tool call
- Replace the fake report writer with an LLM-assisted report generator
- Add evaluation checks for final report quality and source coverage
