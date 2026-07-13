# Tool Architecture

The project now separates paper-research behavior into three layers.

## Three Layers

| Layer | Role | Examples |
| --- | --- | --- |
| Core functions | Small technical operations that are independently testable. | arXiv API search, PDF extraction, chunking, BM25 scoring, Chroma upsert |
| Deterministic workflows | Mandatory ordered sequences that preserve prerequisites and invariants. | paper discovery, paper retrieval preparation, report generation |
| Agent-facing tools | Safe high-level capabilities that a future LLM planner may choose. | `discover_papers`, `retrieve_evidence`, `generate_paper_report` |

The LLM planner should choose the needed capability. Deterministic code decides the exact technical order.

## Production Tool Catalog

Only these tools should be sent to the future production planner.

| Agent capability | Internal workflow | Persistent side effect |
| --- | --- | --- |
| `discover_papers` | plan -> search -> seen filter -> dedup -> rank -> relevance filter | No persistent side effect; mutates runtime state |
| `list_papers` | SQLite metadata query | No |
| `get_paper_metadata` | resolve metadata -> inspect local artifacts -> inspect vector index | No |
| `save_papers_to_kb` | validate explicit ids -> idempotent SQLite save/skip | Yes, non-destructive |
| `ensure_papers_retrievable` | fetch -> extract -> chunk -> embed -> index, skipping existing stages | Yes, non-destructive |
| `retrieve_evidence` | validate indexed papers -> hybrid retrieval -> compact evidence output | No |
| `summarize_papers` | resolve explicit ids -> abstract summary workflow | Runtime state mutation |
| `generate_paper_report` | resolve explicit ids -> ensure summaries -> generate markdown report | Runtime state mutation |

## Development Tools

Development tools remain available for tests, scripts, and diagnostics, but should not be planner-facing:

- fake paper search/ranking/report tools
- retrieval evaluation tools
- embedding and vector-store smoke scripts
- scoring debug scripts

## Admin Tools

Administrative cleanup is intentionally separated from production planning:

- remove fetched paper files
- remove papers from the SQLite knowledge base

Global deletion or reset operations must require trusted application confirmation. An LLM-provided argument such as `{"confirmed": true}` is not trusted confirmation.

## Grounded RAG Generation

Grounded answer generation remains reusable, but it should become a terminal graph node in the future dynamic planner:

```text
planner -> retrieve_evidence -> evidence sufficiency check -> grounded generation -> citation validation
```

This avoids exposing `retrieve_evidence` and `answer_with_rag` as competing planner choices.
