# Agentic AI Research Assistant

A Python research-assistant prototype that searches arXiv, ranks candidate papers,
summarizes selected abstracts with an LLM, generates a markdown report, and stores
seen papers in a local SQLite knowledge base.

## Current Capabilities

- Live arXiv retrieval with stricter title/abstract queries and AI/ML/NLP category filters
- Optional LLM query planning for turning long user prompts into structured arXiv search terms
- Deduplication by `paper_id` or normalized title
- Hybrid relevance scoring:
  - BM25 lexical score
  - semantic similarity score
  - title/key phrase match score
  - recency score
  - hard/soft gates for topic-specific core signals such as RAG, RLHF, and RLVR
- Relevance filtering using final score and component scores
- OpenAI-backed abstract summaries with abstract fallback when the LLM call fails
- Markdown report generation from selected papers
- SQLite metadata storage for seen/selected papers
- Local Chroma vector storage for full-paper RAG chunks
- Metadata-aware dense retrieval with hard filters and soft metadata hint reranking
- Knowledge-base tools for filtering seen papers, saving papers, and removing stored papers
- Test coverage for state models, tools, runner workflows, arXiv parsing, scoring, LLM clients, and storage

## Project Layout

```text
app/
  agent/
    runner.py                 # Workflow runner
    state.py                  # AgentState, Paper, PaperSummary, SearchPlan
  llm/
    client.py                 # OpenAI/Gemini client wrappers
    fake_llm.py               # Test fake LLM
  storage/
    paper_store.py            # SQLite paper metadata store
  vectorstores/
    chroma_store.py           # Persistent local Chroma adapter
    metadata.py               # Retrieval metadata normalization/filter translation
  retrieval/
    retriever.py              # Metadata-aware semantic retriever
  services/
    chunk_indexing.py         # Chunk-to-vector-store ingestion service
  tools/
    arxiv_tools.py            # arXiv search and query building
    filter_relevant_papers.py # Relevance filtering
    knowledge_base_tools.py   # Save/filter/remove papers in SQLite
    llm_query_planner_tools.py
    llm_summary_tools.py
    report_tools.py
    scoring_tools.py
    registry.py
scripts/
  debug_scoring_run.py
  remove_papers_test.py
  test_openai_query_planner.py
  test_openai_summary.py
tests/
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For OpenAI-backed query planning and summaries, create a local `.env` file:

```bash
cp .env.example .env
```

Then paste your key into `.env`:

```text
OPENAI_API_KEY="your_key_here"
```

The LLM clients load `.env` automatically. Do not commit API keys: `.env` is
ignored by git. Local data is written under `data/`, which is also ignored.

## Run

Run the main research workflow:

```bash
python -m app.main
```

The current main flow:

1. `search_arxiv_papers`
2. `filter_seen_papers`
3. `deduplicate_papers`
4. `rank_papers_by_similarity`
5. `filter_relevant_papers`
6. `summarize_papers_with_llm`
7. `generate_report_from_abstracts`
8. `save_selected_papers_to_kb`

The output includes:

- final markdown report
- knowledge-base save report

## Local Vector Store RAG

The project keeps two storage responsibilities separate:

- SQLite is the canonical paper metadata/history store. It remembers papers,
  topics, deduplication state, and selected/seen history.
- Chroma stores one vector record per chunk: the raw chunk document, the
  precomputed embedding from the existing BGE embedding layer, and a flat
  retrieval metadata projection.

The default persistent Chroma collection is `research_paper_chunks_v1` under
`data/vector_store/chroma`. Collection metadata validates the embedding model,
embedding dimension, distance metric, and metadata schema version so incompatible
vector spaces are not mixed silently.

Required chunk metadata includes `paper_id`, `knowledge_base_ids`, `source`,
`language`, `title`, `published_year`, `published_yyyymmdd`, `section`,
`section_group`, `chunk_type`, `chunk_index`, `word_count`, `text_source`,
`metadata_schema_version`, `embedding_model_id`, and `embedding_dimension`.
Semantic tag fields such as `topics`, `methods`, `datasets`, `tasks`, `models`,
and `evaluation_metrics` are normalized to lowercase snake case.

Retrieval uses:

- hard filters such as paper IDs, knowledge base IDs, source, section group, and
  date ranges inside the vector search scope;
- dense semantic search using query embeddings from the existing embedding layer;
- optional soft metadata hints that rerank candidates without excluding them.

Index selected paper chunks after chunking:

```bash
python -m scripts.vector_store_smoke_run
```

Search an existing embedding JSONL file directly:

```bash
python -m scripts.search_embeddings_run \
  --embeddings-path data/papers/arxiv_2601_17212v1/embeddings.jsonl \
  --query "query-aware diversity for retrieval augmented generation"
```

For code-level use, call `index_chunks(...)` from `app.services.chunk_indexing`
or `MetadataAwareRetriever.retrieve(...)` from `app.retrieval.retriever`. To
delete/reindex one paper, call `ChromaVectorStore.delete_by_paper(paper_id)` and
then index its chunks again. For a test reset, delete only the temporary Chroma
directory or use a new temporary path; do not call client-wide reset operations.

## Utility Scripts

Debug retrieval and scoring:

```bash
python -m scripts.debug_scoring_run
```

Test OpenAI query planning only:

```bash
python -m scripts.test_openai_query_planner
```

Test OpenAI summary only:

```bash
python -m scripts.test_openai_summary
```

Remove the latest saved RAG demo papers:

```bash
python -m scripts.remove_papers_test
```

Remove every paper from the local SQLite knowledge base:

```bash
python -m scripts.remove_papers_test --all
```

## Tests

Run all tests:

```bash
pytest
```

Current coverage includes:

- state and Pydantic model behavior
- arXiv XML parsing and query construction
- fake and live-tool-compatible workflows
- hybrid scoring and hard/soft relevance gates
- OpenAI/Gemini client wrappers
- LLM query planning and summary fallback behavior
- SQLite paper store save/filter/remove behavior
- registry and runner integration

## Notes

This is still a prototype, but it now has the core loop of a usable research
assistant: retrieve candidates, score them, filter them, summarize selected
papers, report results, and remember what has already been seen.
