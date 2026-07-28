from __future__ import annotations

import argparse
import json
import os
import sqlite3
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, text

from app.config import get_settings
from app.conversations.postgres_repository import PostgresConversationRepository
from app.storage.postgres_paper_store import PostgresPaperStore


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate local SQLite metadata databases to Postgres."
    )
    parser.add_argument("--conversation-db", default=None)
    parser.add_argument("--paper-db", default=None)
    parser.add_argument("--database-url", default=None)
    parser.add_argument("--skip-conversations", action="store_true")
    parser.add_argument("--skip-papers", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    database_url = args.database_url or settings.database_url or os.getenv("DATABASE_URL")
    if not database_url:
        raise SystemExit("DATABASE_URL or --database-url is required.")

    engine = create_engine(database_url, pool_pre_ping=True, future=True)
    PostgresConversationRepository(database_url=database_url)
    PostgresPaperStore(database_url=database_url)

    if not args.skip_conversations:
        conversation_db = Path(args.conversation_db or settings.conversation_db_path)
        counts = migrate_conversations(conversation_db, engine)
        print(f"migrated_conversations={counts}")

    if not args.skip_papers:
        paper_db = Path(args.paper_db or settings.paper_db_path)
        counts = migrate_papers(paper_db, engine)
        print(f"migrated_papers={counts}")


def migrate_conversations(sqlite_path: Path, engine) -> dict[str, int]:
    if not sqlite_path.exists():
        return {"threads": 0, "messages": 0, "runs": 0, "steps": 0}

    with sqlite3.connect(sqlite_path) as source, engine.begin() as target:
        source.row_factory = sqlite3.Row
        threads = source.execute("SELECT * FROM conversation_threads").fetchall()
        messages = source.execute(
            "SELECT * FROM conversation_messages ORDER BY thread_id, sequence_number"
        ).fetchall()
        runs = source.execute("SELECT * FROM agent_runs").fetchall()
        steps = source.execute(
            "SELECT * FROM agent_steps ORDER BY run_id, step_number"
        ).fetchall()

        for row in threads:
            target.execute(
                text(
                    """
                    INSERT INTO conversation_threads (
                        thread_id, user_id, title, created_at, updated_at, status,
                        conversation_summary, summary_updated_at
                    )
                    VALUES (
                        :thread_id, :user_id, :title, :created_at, :updated_at,
                        :status, :conversation_summary, :summary_updated_at
                    )
                    ON CONFLICT (thread_id) DO UPDATE SET
                        user_id = EXCLUDED.user_id,
                        title = EXCLUDED.title,
                        updated_at = EXCLUDED.updated_at,
                        status = EXCLUDED.status,
                        conversation_summary = EXCLUDED.conversation_summary,
                        summary_updated_at = EXCLUDED.summary_updated_at
                    """
                ),
                dict(row),
            )

        for row in messages:
            payload = dict(row)
            payload["metadata_json"] = _json_text(payload.get("metadata_json"), "{}")
            target.execute(
                text(
                    """
                    INSERT INTO conversation_messages (
                        message_id, thread_id, role, content, created_at,
                        sequence_number, metadata_json
                    )
                    VALUES (
                        :message_id, :thread_id, :role, :content, :created_at,
                        :sequence_number, CAST(:metadata_json AS jsonb)
                    )
                    ON CONFLICT (message_id) DO UPDATE SET
                        role = EXCLUDED.role,
                        content = EXCLUDED.content,
                        metadata_json = EXCLUDED.metadata_json
                    """
                ),
                payload,
            )

        for row in runs:
            payload = dict(row)
            payload["token_usage_json"] = _json_text(
                payload.get("token_usage_json"),
                "null",
            )
            target.execute(
                text(
                    """
                    INSERT INTO agent_runs (
                        run_id, thread_id, user_request_message_id, status,
                        started_at, completed_at, latency_ms, token_usage_json,
                        estimated_cost, error_type, error_message, graph_thread_id
                    )
                    VALUES (
                        :run_id, :thread_id, :user_request_message_id, :status,
                        :started_at, :completed_at, :latency_ms,
                        CAST(:token_usage_json AS jsonb), :estimated_cost,
                        :error_type, :error_message, :graph_thread_id
                    )
                    ON CONFLICT (run_id) DO UPDATE SET
                        status = EXCLUDED.status,
                        completed_at = EXCLUDED.completed_at,
                        latency_ms = EXCLUDED.latency_ms,
                        token_usage_json = EXCLUDED.token_usage_json,
                        estimated_cost = EXCLUDED.estimated_cost,
                        error_type = EXCLUDED.error_type,
                        error_message = EXCLUDED.error_message
                    """
                ),
                payload,
            )

        for row in steps:
            payload = dict(row)
            payload["arguments_json"] = _json_text(
                payload.get("arguments_json"),
                "null",
            )
            payload["observation_json"] = _json_text(
                payload.get("observation_json"),
                "null",
            )
            target.execute(
                text(
                    """
                    INSERT INTO agent_steps (
                        step_id, run_id, step_number, node_name, decision_type,
                        tool_name, arguments_json, observation_status,
                        observation_json, latency_ms, created_at
                    )
                    VALUES (
                        :step_id, :run_id, :step_number, :node_name, :decision_type,
                        :tool_name, CAST(:arguments_json AS jsonb),
                        :observation_status, CAST(:observation_json AS jsonb),
                        :latency_ms, :created_at
                    )
                    ON CONFLICT (step_id) DO UPDATE SET
                        observation_status = EXCLUDED.observation_status,
                        observation_json = EXCLUDED.observation_json,
                        latency_ms = EXCLUDED.latency_ms
                    """
                ),
                payload,
            )

    return {
        "threads": len(threads),
        "messages": len(messages),
        "runs": len(runs),
        "steps": len(steps),
    }


def migrate_papers(sqlite_path: Path, engine) -> dict[str, int]:
    if not sqlite_path.exists():
        return {"papers": 0, "paper_topics": 0}

    with sqlite3.connect(sqlite_path) as source, engine.begin() as target:
        source.row_factory = sqlite3.Row
        papers = source.execute("SELECT * FROM papers").fetchall()
        topics = source.execute("SELECT * FROM paper_topics ORDER BY id").fetchall()

        for row in papers:
            payload = dict(row)
            payload["authors_json"] = _json_text(payload.get("authors_json"), "[]")
            payload["external_ids_json"] = _json_text(
                payload.get("external_ids_json"),
                "{}",
            )
            payload["provenance_json"] = _json_text(
                payload.get("provenance_json"),
                "[]",
            )
            target.execute(
                text(
                    """
                    INSERT INTO papers (
                        paper_id, title, authors_json, source, url, abstract,
                        published_date, doi, arxiv_id, semantic_scholar_id,
                        external_ids_json, provenance_json, venue, citation_count,
                        open_access_pdf_url, first_seen_at, last_seen_at
                    )
                    VALUES (
                        :paper_id, :title, CAST(:authors_json AS jsonb), :source,
                        :url, :abstract, :published_date, :doi, :arxiv_id,
                        :semantic_scholar_id, CAST(:external_ids_json AS jsonb),
                        CAST(:provenance_json AS jsonb), :venue, :citation_count,
                        :open_access_pdf_url, :first_seen_at, :last_seen_at
                    )
                    ON CONFLICT (paper_id) DO UPDATE SET
                        title = EXCLUDED.title,
                        authors_json = EXCLUDED.authors_json,
                        source = EXCLUDED.source,
                        url = EXCLUDED.url,
                        abstract = EXCLUDED.abstract,
                        published_date = EXCLUDED.published_date,
                        doi = EXCLUDED.doi,
                        arxiv_id = EXCLUDED.arxiv_id,
                        semantic_scholar_id = EXCLUDED.semantic_scholar_id,
                        external_ids_json = EXCLUDED.external_ids_json,
                        provenance_json = EXCLUDED.provenance_json,
                        venue = EXCLUDED.venue,
                        citation_count = EXCLUDED.citation_count,
                        open_access_pdf_url = EXCLUDED.open_access_pdf_url,
                        last_seen_at = EXCLUDED.last_seen_at
                    """
                ),
                payload,
            )

        for row in topics:
            payload = dict(row)
            target.execute(
                text(
                    """
                    INSERT INTO paper_topics (id, paper_id, topic, score, selected, seen_at)
                    VALUES (:id, :paper_id, :topic, :score, :selected, :seen_at)
                    ON CONFLICT (id) DO UPDATE SET
                        paper_id = EXCLUDED.paper_id,
                        topic = EXCLUDED.topic,
                        score = EXCLUDED.score,
                        selected = EXCLUDED.selected,
                        seen_at = EXCLUDED.seen_at
                    """
                ),
                payload,
            )
        if topics:
            target.execute(
                text(
                    """
                    SELECT setval(
                        pg_get_serial_sequence('paper_topics', 'id'),
                        (SELECT MAX(id) FROM paper_topics)
                    )
                    """
                )
            )

    return {"papers": len(papers), "paper_topics": len(topics)}


def _json_text(value: Any, fallback: str) -> str:
    if value is None:
        return fallback
    if isinstance(value, str):
        try:
            json.loads(value)
        except json.JSONDecodeError:
            return fallback
        return value
    return json.dumps(value)


if __name__ == "__main__":
    main()
