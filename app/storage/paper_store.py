from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable

from app.agent.state import Paper


DEFAULT_DB_PATH = Path("data/metadata/papers.sqlite3")


class PaperStore:
    """
    Persistent metadata store for papers.

    This store is responsible for remembering papers across multiple runs.
    It uses paper_id as the stable unique key.
    """

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS papers (
                    paper_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    authors_json TEXT NOT NULL,
                    source TEXT NOT NULL,
                    url TEXT,
                    abstract TEXT,
                    published_date TEXT,
                    first_seen_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    last_seen_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS paper_topics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    paper_id TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    score REAL,
                    selected INTEGER DEFAULT 0,
                    seen_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (paper_id) REFERENCES papers (paper_id)
                )
                """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_paper_topics_paper_id
                ON paper_topics (paper_id)
                """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_paper_topics_topic
                ON paper_topics (topic)
                """
            )

    def paper_exists(self, paper_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM papers WHERE paper_id = ? LIMIT 1",
                (paper_id,),
            ).fetchone()

        return row is not None

    def get_seen_paper_ids(self) -> set[str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT paper_id FROM papers").fetchall()

        return {row[0] for row in rows}

    def get_all_paper_ids(self) -> list[str]:
        """
        Return all stored paper ids in insertion order.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT paper_id FROM papers ORDER BY first_seen_at, paper_id"
            ).fetchall()

        return [row[0] for row in rows]

    def save_paper(
        self,
        paper: Paper,
        topic: str,
        selected: bool = False,
    ) -> None:
        authors_json = json.dumps(paper.authors)

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO papers (
                    paper_id,
                    title,
                    authors_json,
                    source,
                    url,
                    abstract,
                    published_date
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(paper_id) DO UPDATE SET
                    title = excluded.title,
                    authors_json = excluded.authors_json,
                    source = excluded.source,
                    url = excluded.url,
                    abstract = excluded.abstract,
                    published_date = excluded.published_date,
                    last_seen_at = CURRENT_TIMESTAMP
                """,
                (
                    paper.paper_id,
                    paper.title,
                    authors_json,
                    paper.source,
                    paper.url,
                    paper.abstract,
                    paper.published_date,
                ),
            )

            conn.execute(
                """
                INSERT INTO paper_topics (
                    paper_id,
                    topic,
                    score,
                    selected
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    paper.paper_id,
                    topic,
                    paper.score,
                    int(selected),
                ),
            )

    def save_papers(
        self,
        papers: Iterable[Paper],
        topic: str,
        selected: bool = False,
    ) -> int:
        count = 0

        for paper in papers:
            self.save_paper(
                paper=paper,
                topic=topic,
                selected=selected,
            )
            count += 1

        return count

    def remove_paper(self, paper_id: str) -> bool:
        """
        Remove one paper and its topic history from the store.
        """
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM paper_topics WHERE paper_id = ?",
                (paper_id,),
            )
            cursor = conn.execute(
                "DELETE FROM papers WHERE paper_id = ?",
                (paper_id,),
            )

        return cursor.rowcount > 0

    def remove_papers(self, paper_ids: Iterable[str]) -> int:
        """
        Remove multiple papers from the store.
        """
        removed_count = 0

        for paper_id in paper_ids:
            if self.remove_paper(paper_id):
                removed_count += 1

        return removed_count
