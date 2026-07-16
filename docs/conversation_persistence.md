# Conversation Persistence

This layer adds durable conversation history around the existing LangGraph
planner without changing planner or production tool behavior.

## Runtime Flow

`ConversationAgentService.run_turn(...)` owns one user turn:

1. Open or create a `conversation_threads` row.
2. Append the user request to `conversation_messages`.
3. Start an `agent_runs` row for this turn.
4. Build compact context from previous messages only.
5. Invoke `LangGraphAgentRunner.run(...)`.
6. Persist executed tool records into `agent_steps`.
7. Append the assistant answer only when the planner run succeeds.
8. Mark the run completed or failed.
9. Refresh the rolling thread summary when the message threshold is reached.

## SQLite Schema

The default database path is `data/metadata/conversations.sqlite3`.

- `conversation_threads`: one row per chat thread, including title, status, and
  rolling summary.
- `conversation_messages`: user and assistant messages with sequence numbers and
  safe structured metadata.
- `agent_runs`: one row per agent execution tied to the triggering user message.
- `agent_steps`: tool-level trace records tied to a run.

Conversation messages are intentionally separate from traces. User-facing chat
history can be rendered without loading planner diagnostics, while debugging can
inspect every run and tool observation through `agent_runs` and `agent_steps`.

## Planner Context

The graph state now carries conversation fields in `PlannerState`:

- `thread_id`
- `run_id`
- `current_user_message_id`
- `final_assistant_message_id`
- `conversation_summary`
- `recent_messages`
- `active_paper_ids`

`ConversationContextBuilder` sends only a compact window to the planner. It does
not perform semantic retrieval over chat history. Active papers are extracted
from structured message metadata keys such as `paper_ids`, `active_paper_ids`,
and `cited_paper_ids`.

## Storage Safety

Repository writes sanitize JSON metadata before persistence. Sensitive key names
such as `api_key`, `authorization`, `password`, `secret`, and `token` are
removed, long strings are truncated, and large lists are capped.

## Local Smoke

Run the deterministic conversation smoke without a live LLM:

```bash
python -m scripts.conversation_smoke_run
```

Use a temporary database during testing:

```bash
python -m scripts.conversation_smoke_run --db-path /tmp/conversations.sqlite3
```
