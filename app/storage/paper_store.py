from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Iterable

from app.config import get_settings
from app.agent.state import Paper


DEFAULT_DB_PATH = Path("data/metadata/papers.sqlite3")
DEFAULT_PAPERS_DIR = Path("data/papers")


class PaperStore:
    """
    Persistent metadata store for papers.

    This store is responsible for remembering papers across multiple runs.
    It uses paper_id as the stable unique key.
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        papers_dir: str | Path | None = None,
    ):
        settings = get_settings()
        self.db_path = Path(db_path or settings.paper_db_path)
        self.papers_dir = Path(papers_dir or settings.papers_dir)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.papers_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        """Open a SQLite connection to the paper metadata database."""

        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        """Create the metadata tables and indexes used by the store."""

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
                    doi TEXT,
                    arxiv_id TEXT,
                    semantic_scholar_id TEXT,
                    external_ids_json TEXT NOT NULL DEFAULT '{}',
                    provenance_json TEXT NOT NULL DEFAULT '[]',
                    venue TEXT,
                    citation_count INTEGER,
                    open_access_pdf_url TEXT,
                    first_seen_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    last_seen_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            self._ensure_optional_paper_columns(conn)

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

    def _ensure_optional_paper_columns(self, conn: sqlite3.Connection) -> None:
        """Add optional metadata columns to existing SQLite paper stores."""

        existing_columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(papers)").fetchall()
        }
        optional_columns = {
            "doi": "TEXT",
            "arxiv_id": "TEXT",
            "semantic_scholar_id": "TEXT",
            "external_ids_json": "TEXT NOT NULL DEFAULT '{}'",
            "provenance_json": "TEXT NOT NULL DEFAULT '[]'",
            "venue": "TEXT",
            "citation_count": "INTEGER",
            "open_access_pdf_url": "TEXT",
        }
        for column_name, column_type in optional_columns.items():
            if column_name not in existing_columns:
                conn.execute(
                    f"ALTER TABLE papers ADD COLUMN {column_name} {column_type}"
                )

    def paper_exists(self, paper_id: str) -> bool:
        """Return whether a paper id has already been stored."""

        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM papers WHERE paper_id = ? LIMIT 1",
                (paper_id,),
            ).fetchone()

        return row is not None

    def get_seen_paper_ids(self) -> set[str]:
        """Return all paper ids known to the metadata database."""

        with self._connect() as conn:
            rows = conn.execute("SELECT paper_id FROM papers").fetchall()

        return {row[0] for row in rows}

    def get_saved_paper_ids(self) -> set[str]:
        """Return paper ids that the user explicitly saved for later use."""

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT paper_id
                FROM paper_topics
                WHERE selected = 1
                """
            ).fetchall()

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

    def get_paper(self, paper_id: str) -> Paper | None:
        """Return one stored paper as a Paper model, if present."""

        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT paper_id, title, authors_json, source, url, abstract,
                       published_date, doi, arxiv_id, semantic_scholar_id,
                       external_ids_json, provenance_json, venue, citation_count,
                       open_access_pdf_url
                FROM papers
                WHERE paper_id = ?
                """,
                (paper_id,),
            ).fetchone()

        if row is None:
            return None

        return Paper(
            paper_id=row[0],
            title=row[1],
            authors=json.loads(row[2]),
            source=row[3],
            url=row[4] or "",
            abstract=row[5],
            published_date=row[6],
            doi=row[7],
            arxiv_id=row[8],
            semantic_scholar_id=row[9],
            external_ids=_loads_json_dict(row[10]),
            provenance=_loads_json_list(row[11]),
            venue=row[12],
            citation_count=row[13],
            open_access_pdf_url=row[14],
        )

    def get_paper_record(self, paper_id: str) -> dict | None:
        """Return compact stored metadata for one paper, including KB dates."""

        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT paper_id, title, authors_json, source, url, abstract,
                       published_date, first_seen_at, last_seen_at, doi, arxiv_id,
                       semantic_scholar_id, external_ids_json, provenance_json,
                       venue, citation_count, open_access_pdf_url
                FROM papers
                WHERE paper_id = ?
                """,
                (paper_id,),
            ).fetchone()

        if row is None:
            return None

        return {
            "paper_id": row[0],
            "title": row[1],
            "authors": json.loads(row[2]),
            "source": row[3],
            "url": row[4],
            "abstract": row[5],
            "published_date": row[6],
            "added_date": row[7],
            "last_seen_at": row[8],
            "doi": row[9],
            "arxiv_id": row[10],
            "semantic_scholar_id": row[11],
            "external_ids": _loads_json_dict(row[12]),
            "provenance": _loads_json_list(row[13]),
            "venue": row[14],
            "citation_count": row[15],
            "open_access_pdf_url": row[16],
        }

    def list_paper_records(
        self,
        *,
        paper_ids: list[str] | None = None,
        published_after: str | None = None,
        published_before: str | None = None,
        added_after: str | None = None,
        limit: int = 10,
        sort_by: str = "published_date",
        descending: bool = True,
    ) -> list[dict]:
        """List stored paper metadata without mutating runtime state."""

        clauses: list[str] = []
        params: list[str | int] = []

        if paper_ids:
            placeholders = ", ".join("?" for _ in paper_ids)
            clauses.append(f"paper_id IN ({placeholders})")
            params.extend(paper_ids)
        if published_after:
            clauses.append("published_date >= ?")
            params.append(published_after)
        if published_before:
            clauses.append("published_date <= ?")
            params.append(published_before)
        if added_after:
            clauses.append("first_seen_at >= ?")
            params.append(added_after)

        sort_column = {
            "published_date": "published_date",
            "added_date": "first_seen_at",
            "relevance": "last_seen_at",
        }.get(sort_by, "published_date")
        order = "DESC" if descending else "ASC"
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT paper_id, title, authors_json, source, url, abstract,
                       published_date, first_seen_at, last_seen_at, doi, arxiv_id,
                       semantic_scholar_id, external_ids_json, provenance_json,
                       venue, citation_count, open_access_pdf_url
                FROM papers
                {where}
                ORDER BY {sort_column} {order}, paper_id {order}
                LIMIT ?
                """,
                params,
            ).fetchall()

        return [
            {
                "paper_id": row[0],
                "title": row[1],
                "authors": json.loads(row[2]),
                "source": row[3],
                "url": row[4],
                "abstract": row[5],
                "published_date": row[6],
                "added_date": row[7],
                "last_seen_at": row[8],
                "doi": row[9],
                "arxiv_id": row[10],
                "semantic_scholar_id": row[11],
                "external_ids": _loads_json_dict(row[12]),
                "provenance": _loads_json_list(row[13]),
                "venue": row[14],
                "citation_count": row[15],
                "open_access_pdf_url": row[16],
            }
            for row in rows
        ]

    def paper_dir(self, paper_id: str) -> Path:
        """Return and create the filesystem directory for one paper."""

        path = self.papers_dir / _safe_paper_id(paper_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def pdf_path(self, paper_id: str) -> Path:
        """Return the expected local PDF path for one paper."""

        return self.paper_dir(paper_id) / "paper.pdf"

    def raw_text_path(self, paper_id: str) -> Path:
        """Return the raw extracted text path for one paper."""

        return self.paper_dir(paper_id) / "raw_text.txt"

    def clean_text_path(self, paper_id: str) -> Path:
        """Return the cleaned text path for one paper."""

        return self.paper_dir(paper_id) / "clean_text.txt"

    def chunks_path(self, paper_id: str) -> Path:
        """Return the JSONL chunk path for one paper."""

        return self.paper_dir(paper_id) / "chunks.jsonl"

    def embeddings_path(self, paper_id: str) -> Path:
        """Return the JSONL embeddings path for one paper."""

        return self.paper_dir(paper_id) / "embeddings.jsonl"

    def save_raw_text(self, paper_id: str, text: str) -> Path:
        """Save raw extracted PDF text and return the written path."""

        path = self.raw_text_path(paper_id)
        path.write_text(text, encoding="utf-8")
        return path

    def save_clean_text(self, paper_id: str, text: str) -> Path:
        """Save cleaned paper text and return the written path."""

        path = self.clean_text_path(paper_id)
        path.write_text(text, encoding="utf-8")
        return path

    def save_paper(
        self,
        paper: Paper,
        topic: str,
        selected: bool = False,
    ) -> None:
        """Upsert paper metadata and append topic-level run history."""

        authors_json = json.dumps(paper.authors)
        external_ids_json = json.dumps(paper.external_ids)
        provenance_json = json.dumps(paper.provenance)

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
                    published_date,
                    doi,
                    arxiv_id,
                    semantic_scholar_id,
                    external_ids_json,
                    provenance_json,
                    venue,
                    citation_count,
                    open_access_pdf_url
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(paper_id) DO UPDATE SET
                    title = excluded.title,
                    authors_json = excluded.authors_json,
                    source = excluded.source,
                    url = excluded.url,
                    abstract = excluded.abstract,
                    published_date = excluded.published_date,
                    doi = excluded.doi,
                    arxiv_id = excluded.arxiv_id,
                    semantic_scholar_id = excluded.semantic_scholar_id,
                    external_ids_json = excluded.external_ids_json,
                    provenance_json = excluded.provenance_json,
                    venue = excluded.venue,
                    citation_count = excluded.citation_count,
                    open_access_pdf_url = excluded.open_access_pdf_url,
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
                    paper.doi,
                    paper.arxiv_id,
                    paper.semantic_scholar_id,
                    external_ids_json,
                    provenance_json,
                    paper.venue,
                    paper.citation_count,
                    paper.open_access_pdf_url,
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
        """Save a batch of papers and return how many were processed."""

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


def _safe_paper_id(paper_id: str) -> str:
    """Convert a paper id into a safe directory name."""

    safe_id = paper_id.strip().lower()
    safe_id = re.sub(r"[^a-z0-9]+", "_", safe_id)
    safe_id = re.sub(r"_+", "_", safe_id).strip("_")
    return safe_id or "unknown_paper"


def _loads_json_dict(value: str | None) -> dict:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _loads_json_list(value: str | None) -> list:
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []
