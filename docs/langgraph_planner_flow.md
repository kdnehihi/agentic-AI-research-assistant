# LangGraph Planner Flow

The dynamic research planner now has a LangGraph orchestration runner in
`app/agent/langgraph_runner.py`.

Entrypoints that use the LangGraph runner:

- `scripts/dynamic_planner_smoke_run.py`
- `scripts/dynamic_smoke_utils.py`
- `scripts/evaluate_dynamic_planner.py` through `app/agent/planner_eval.py`

`LangGraphAgentRunner` is the only dynamic planner runner. The older imperative
runner was removed after its behavior coverage was moved to LangGraph tests.

## Graph State

The graph passes a `LangGraphRunnerState` dictionary:

- `planner_state`: the existing `PlannerState` object that owns runtime state,
  tool history, pending decision, retrieved evidence, status, and final answer.
- `tool_specs`: production tool specs from `ToolExecutor.production_tool_specs()`.

The graph intentionally reuses the current planner, executor, observation
factory, policy, and answer service. The conversion changes orchestration, not
domain logic.

`PlannerState` also carries two supervisor fields:

- `execution_strategy`: the current high-level branch, such as
  `knowledge_only`, `discovery_only`, or `discover_then_answer`.
- `knowledge_coverage`: a deterministic verdict on whether retrieved evidence
  is sufficient, partial, or insufficient for the request.

## Nodes

`route_request`

- Classifies the request intent.
- Chooses an initial execution strategy for confident intents.
- Builds a deterministic execution plan when the strategy is clear.
- Falls back to the LLM execution planner when deterministic routing is not
  confident enough.

`decide`

- Runs `choose_policy_action()` first when policy is enabled.
- If no policy action applies, calls `Planner.decide()`.
- Stores the result in `planner_state.pending_decision`.
- Planner errors set `planner_state.status = "failed"`.

`execute_tool`

- Requires `pending_decision` to be a `CallToolAction`.
- Calls `ToolExecutor.execute()`.
- The executor validates args, invokes the production tool, normalizes the
  observation, updates `PlannerState`, appends tool history, and records latency.
- If retrieval reports `paper_not_retrievable`, the graph stores the original
  retrieval as `PlannerState.retry_decision`, routes to
  `ensure_papers_retrievable`, then retries the original retrieval without
  asking the LLM.
- After retrieval, the supervisor evaluates knowledge coverage. If evidence is
  missing, stale, or incomplete, it can replace the current plan with
  `discover_then_answer` so the graph searches papers, saves them, ensures they
  are indexed, retrieves again, and only then tries to finish.

`finish`

- Requires `pending_decision` to be a `FinishAction`.
- Calls `validate_finish()` to prevent unsupported early finishes, including
  finishes with partial or insufficient knowledge coverage.
- Calls `GroundedAnswerService.generate()` to produce the final answer.
- Sets `status = "success"` when final generation succeeds.

`max_steps`

- Fails the run with `Maximum planner steps reached.`

## Edges

```text
decide
  ├─ CallToolAction -> execute_tool -> decide
  │                    ├─ insufficient/stale coverage
  │                    │  -> discover/save/ensure/retrieve
  │                    │  -> decide
  │                    └─ paper_not_retrievable
  │                       -> ensure_papers_retrievable
  │                       -> retry original retrieve_evidence
  │                       -> decide
  ├─ FinishAction   -> finish -> END
  ├─ max steps      -> max_steps -> END
  └─ failure        -> END
```

The repeated `execute_tool -> decide` edge is the dynamic planning loop.
LangGraph owns the loop; the supervisor decides the branch; planner and tools
stay modular.

## Evaluation Gate

Before freezing or changing planner behavior, run:

```bash
python -m scripts.evaluate_dynamic_planner
pytest
```

The deterministic eval covers current frozen planner contracts, including:

- existing KB answer with a policy KB probe,
- missing KB answer followed by discovery and retrieval,
- partial or stale KB evidence followed by discovery, save, ingestion, and
  retrieval,
- unindexed paper retrieval followed by automatic ensure-and-retry recovery,
- new-paper discovery/ingestion/retrieval,
- multi-paper compare retrieval.
