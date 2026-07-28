from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from sqlalchemy import bindparam, create_engine, text
from sqlalchemy.engine import Engine

from app.agent.state import Paper
from app.config import get_settings


class PostgresPaperStore:
    """Postgres-backed paper metadata store with local artifact path helpers."""

    def __init__(
        self,
        *,
        database_url: str | None = None,
        engine: Engine | None = None,
        papers_dir: str | Path | None = None,
        initialize: bool = True,
    ) -> None:
        settings = get_settings()
        self.database_url = database_url or settings.database_url
        if engine is None and not self.database_url:
            raise ValueError("PostgresPaperStore requires DATABASE_URL.")
        self.papers_dir = Path(papers_dir or settings.papers_dir)
        self.papers_dir.mkdir(parents=True, exist_ok=True)
        self.engine = engine or create_engine(
            self.database_url,
            pool_pre_ping=True,
            future=True,
        )
        if initialize:
            self._init_db()

    def _init_db(self) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS papers (
                        paper_id TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        authors_json JSONB NOT NULL,
                        source TEXT NOT NULL,
                        url TEXT,
                        abstract TEXT,
                        published_date TEXT,
                        doi TEXT,
                        arxiv_id TEXT,
                        semantic_scholar_id TEXT,
                        external_ids_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                        provenance_json JSONB NOT NULL DEFAULT '[]'::jsonb,
                        venue TEXT,
                        citation_count INTEGER,
                        open_access_pdf_url TEXT,
                        first_seen_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                        last_seen_at TIMESTAMPTZ NOT NULL DEFAULT now()
                    )
                    """
                )
            )
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS paper_topics (
                        id BIGSERIAL PRIMARY KEY,
                        paper_id TEXT NOT NULL REFERENCES papers(paper_id)
                            ON DELETE CASCADE,
                        topic TEXT NOT NULL,
                        score REAL,
                        selected INTEGER DEFAULT 0,
                        seen_at TIMESTAMPTZ NOT NULL DEFAULT now()
                    )
                    """
                )
            )
            conn.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS idx_paper_topics_paper_id
                    ON paper_topics (paper_id)
                    """
                )
            )
            conn.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS idx_paper_topics_topic
                    ON paper_topics (topic)
                    """
                )
            )
            conn.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS idx_papers_published_date
                    ON papers (published_date)
                    """
                )
            )

    def paper_exists(self, paper_id: str) -> bool:
        with self.engine.connect() as conn:
            row = conn.execute(
                text("SELECT 1 FROM papers WHERE paper_id = :paper_id LIMIT 1"),
                {"paper_id": paper_id},
            ).one_or_none()
        return row is not None

    def get_seen_paper_ids(self) -> set[str]:
        with self.engine.connect() as conn:
            rows = conn.execute(text("SELECT paper_id FROM papers")).all()
        return {row[0] for row in rows}

    def get_saved_paper_ids(self) -> set[str]:
        with self.engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT DISTINCT paper_id
                    FROM paper_topics
                    WHERE selected = 1
                    """
                )
            ).all()
        return {row[0] for row in rows}

    def get_all_paper_ids(self) -> list[str]:
        with self.engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT paper_id
                    FROM papers
                    ORDER BY first_seen_at, paper_id
                    """
                )
            ).all()
        return [row[0] for row in rows]

    def get_paper(self, paper_id: str) -> Paper | None:
        with self.engine.connect() as conn:
            row = conn.execute(
                text(
                    """
                    SELECT paper_id, title, authors_json, source, url, abstract,
                           published_date, doi, arxiv_id, semantic_scholar_id,
                           external_ids_json, provenance_json, venue, citation_count,
                           open_access_pdf_url
                    FROM papers
                    WHERE paper_id = :paper_id
                    """
                ),
                {"paper_id": paper_id},
            ).mappings().one_or_none()
        if row is None:
            return None
        return Paper(
            paper_id=row["paper_id"],
            title=row["title"],
            authors=_json_list(row["authors_json"]),
            source=row["source"],
            url=row["url"] or "",
            abstract=row["abstract"],
            published_date=row["published_date"],
            doi=row["doi"],
            arxiv_id=row["arxiv_id"],
            semantic_scholar_id=row["semantic_scholar_id"],
            external_ids=_json_dict(row["external_ids_json"]),
            provenance=_json_list(row["provenance_json"]),
            venue=row["venue"],
            citation_count=row["citation_count"],
            open_access_pdf_url=row["open_access_pdf_url"],
        )

    def get_paper_record(self, paper_id: str) -> dict | None:
        records = self.list_paper_records(paper_ids=[paper_id], limit=1)
        return records[0] if records else None

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
        clauses: list[str] = []
        params: dict[str, Any] = {"limit": limit}
        if paper_ids:
            clauses.append("paper_id IN :paper_ids")
            params["paper_ids"] = list(paper_ids)
        if published_after:
            clauses.append("published_date >= :published_after")
            params["published_after"] = published_after
        if published_before:
            clauses.append("published_date <= :published_before")
            params["published_before"] = published_before
        if added_after:
            clauses.append("first_seen_at >= :added_after")
            params["added_after"] = added_after

        sort_column = {
            "published_date": "published_date",
            "added_date": "first_seen_at",
            "relevance": "last_seen_at",
        }.get(sort_by, "published_date")
        order = "DESC" if descending else "ASC"
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        statement = text(
            f"""
            SELECT paper_id, title, authors_json, source, url, abstract,
                   published_date, first_seen_at, last_seen_at, doi, arxiv_id,
                   semantic_scholar_id, external_ids_json, provenance_json,
                   venue, citation_count, open_access_pdf_url
            FROM papers
            {where}
            ORDER BY {sort_column} {order}, paper_id {order}
            LIMIT :limit
            """
        )
        if paper_ids:
            statement = statement.bindparams(bindparam("paper_ids", expanding=True))
        with self.engine.connect() as conn:
            rows = conn.execute(statement, params).mappings().all()
        return [_record_from_mapping(row) for row in rows]

    def paper_dir(self, paper_id: str) -> Path:
        path = self.papers_dir / _safe_paper_id(paper_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def pdf_path(self, paper_id: str) -> Path:
        return self.paper_dir(paper_id) / "paper.pdf"

    def raw_text_path(self, paper_id: str) -> Path:
        return self.paper_dir(paper_id) / "raw_text.txt"

    def clean_text_path(self, paper_id: str) -> Path:
        return self.paper_dir(paper_id) / "clean_text.txt"

    def chunks_path(self, paper_id: str) -> Path:
        return self.paper_dir(paper_id) / "chunks.jsonl"

    def embeddings_path(self, paper_id: str) -> Path:
        return self.paper_dir(paper_id) / "embeddings.jsonl"

    def save_raw_text(self, paper_id: str, text_value: str) -> Path:
        path = self.raw_text_path(paper_id)
        path.write_text(text_value, encoding="utf-8")
        return path

    def save_clean_text(self, paper_id: str, text_value: str) -> Path:
        path = self.clean_text_path(paper_id)
        path.write_text(text_value, encoding="utf-8")
        return path

    def save_paper(
        self,
        paper: Paper,
        topic: str,
        selected: bool = False,
    ) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO papers (
                        paper_id, title, authors_json, source, url, abstract,
                        published_date, doi, arxiv_id, semantic_scholar_id,
                        external_ids_json, provenance_json, venue, citation_count,
                        open_access_pdf_url
                    )
                    VALUES (
                        :paper_id, :title, CAST(:authors_json AS jsonb), :source,
                        :url, :abstract, :published_date, :doi, :arxiv_id,
                        :semantic_scholar_id, CAST(:external_ids_json AS jsonb),
                        CAST(:provenance_json AS jsonb), :venue, :citation_count,
                        :open_access_pdf_url
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
                        last_seen_at = now()
                    """
                ),
                _paper_params(paper),
            )
            conn.execute(
                text(
                    """
                    INSERT INTO paper_topics (paper_id, topic, score, selected)
                    VALUES (:paper_id, :topic, :score, :selected)
                    """
                ),
                {
                    "paper_id": paper.paper_id,
                    "topic": topic,
                    "score": paper.score,
                    "selected": int(selected),
                },
            )

    def save_papers(
        self,
        papers: Iterable[Paper],
        topic: str,
        selected: bool = False,
    ) -> int:
        count = 0
        for paper in papers:
            self.save_paper(paper=paper, topic=topic, selected=selected)
            count += 1
        return count

    def remove_paper(self, paper_id: str) -> bool:
        with self.engine.begin() as conn:
            result = conn.execute(
                text("DELETE FROM papers WHERE paper_id = :paper_id"),
                {"paper_id": paper_id},
            )
        return int(result.rowcount or 0) > 0

    def remove_papers(self, paper_ids: Iterable[str]) -> int:
        removed_count = 0
        for paper_id in paper_ids:
            if self.remove_paper(paper_id):
                removed_count += 1
        return removed_count


def _paper_params(paper: Paper) -> dict[str, Any]:
    return {
        "paper_id": paper.paper_id,
        "title": paper.title,
        "authors_json": json.dumps(paper.authors),
        "source": paper.source,
        "url": paper.url,
        "abstract": paper.abstract,
        "published_date": paper.published_date,
        "doi": paper.doi,
        "arxiv_id": paper.arxiv_id,
        "semantic_scholar_id": paper.semantic_scholar_id,
        "external_ids_json": json.dumps(paper.external_ids),
        "provenance_json": json.dumps(paper.provenance),
        "venue": paper.venue,
        "citation_count": paper.citation_count,
        "open_access_pdf_url": paper.open_access_pdf_url,
    }


def _record_from_mapping(row) -> dict[str, Any]:
    return {
        "paper_id": row["paper_id"],
        "title": row["title"],
        "authors": _json_list(row["authors_json"]),
        "source": row["source"],
        "url": row["url"],
        "abstract": row["abstract"],
        "published_date": row["published_date"],
        "added_date": row["first_seen_at"].isoformat()
        if hasattr(row["first_seen_at"], "isoformat")
        else row["first_seen_at"],
        "last_seen_at": row["last_seen_at"].isoformat()
        if hasattr(row["last_seen_at"], "isoformat")
        else row["last_seen_at"],
        "doi": row["doi"],
        "arxiv_id": row["arxiv_id"],
        "semantic_scholar_id": row["semantic_scholar_id"],
        "external_ids": _json_dict(row["external_ids_json"]),
        "provenance": _json_list(row["provenance_json"]),
        "venue": row["venue"],
        "citation_count": row["citation_count"],
        "open_access_pdf_url": row["open_access_pdf_url"],
    }


def _safe_paper_id(paper_id: str) -> str:
    safe_id = paper_id.strip().lower()
    safe_id = re.sub(r"[^a-z0-9]+", "_", safe_id)
    safe_id = re.sub(r"_+", "_", safe_id).strip("_")
    return safe_id or "unknown_paper"


def _json_dict(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _json_list(value: Any) -> list:
    if isinstance(value, list):
        return value
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []
