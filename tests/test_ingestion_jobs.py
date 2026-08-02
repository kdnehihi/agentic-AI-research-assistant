import time
from datetime import datetime, timezone

from app.agent.state import Paper
from app.services.ingestion_jobs import IngestionJob, IngestionJobQueue
from app.services.ingestion_job_store import SQLiteIngestionJobStore


def test_ingestion_job_queue_runs_prepare_func_in_background():
    calls = []

    def prepare(state, *, paper_ids):
        calls.append((state.candidate_papers[0].paper_id, paper_ids))
        return {
            "status": "success",
            "ready_paper_ids": paper_ids,
            "summary": "Prepared paper.",
        }

    queue = IngestionJobQueue(prepare_func=prepare)
    job = queue.submit(
        papers=[
            Paper(
                paper_id="paper:1",
                title="Paper One",
                source="manual",
                url="https://example.com/1",
            )
        ],
        paper_ids=["paper:1"],
        knowledge_base_id="default",
    )

    completed = _wait_for_terminal_job(queue, job.job_id)

    assert completed is not None
    assert completed.status == "success"
    assert completed.result["ready_paper_ids"] == ["paper:1"]
    assert calls == [("paper:1", ["paper:1"])]


def test_ingestion_job_queue_records_prepare_failures():
    def prepare(state, *, paper_ids):
        del state, paper_ids
        raise RuntimeError("prepare exploded")

    queue = IngestionJobQueue(prepare_func=prepare)
    job = queue.submit(
        papers=[
            Paper(
                paper_id="paper:bad",
                title="Bad Paper",
                source="manual",
                url="https://example.com/bad",
            )
        ],
        paper_ids=["paper:bad"],
        knowledge_base_id="default",
    )

    completed = _wait_for_terminal_job(queue, job.job_id)

    assert completed is not None
    assert completed.status == "failed"
    assert completed.error == "prepare exploded"


def test_ingestion_job_queue_returns_before_slow_prepare_and_persists_result(tmp_path):
    def prepare(state, *, paper_ids):
        assert state.candidate_papers[0].paper_id == "paper:slow"
        time.sleep(0.15)
        return {
            "status": "success",
            "ready_paper_ids": paper_ids,
            "summary": "Prepared slow paper.",
        }

    store = SQLiteIngestionJobStore(tmp_path / "ingestion_jobs.sqlite3")
    queue = IngestionJobQueue(prepare_func=prepare, store=store)

    started_at = time.monotonic()
    job = queue.submit(
        papers=[
            Paper(
                paper_id="paper:slow",
                title="Slow Paper",
                source="manual",
                url="https://example.com/slow",
            )
        ],
        paper_ids=["paper:slow"],
        knowledge_base_id="default",
    )
    submit_latency = time.monotonic() - started_at

    assert submit_latency < 0.10
    assert store.get(job.job_id) is not None

    completed = _wait_for_terminal_job(queue, job.job_id)
    assert completed is not None
    assert completed.status == "success"

    reopened_store = SQLiteIngestionJobStore(tmp_path / "ingestion_jobs.sqlite3")
    persisted = reopened_store.get(job.job_id)
    assert persisted is not None
    assert persisted.status == "success"
    assert persisted.result["ready_paper_ids"] == ["paper:slow"]


def test_ingestion_job_queue_resumes_persisted_queued_jobs(tmp_path):
    calls = []
    store = SQLiteIngestionJobStore(tmp_path / "ingestion_jobs.sqlite3")
    now = datetime.now(timezone.utc)
    paper = Paper(
        paper_id="paper:resume",
        title="Resume Paper",
        source="manual",
        url="https://example.com/resume",
    )
    job = IngestionJob(
        job_id="job-resume",
        status="queued",
        paper_ids=["paper:resume"],
        knowledge_base_id="default",
        created_at=now,
        updated_at=now,
    )
    store.create(
        job,
        ([paper.model_dump(mode="json")], ["paper:resume"], "default"),
    )

    def prepare(state, *, paper_ids):
        calls.append((state.candidate_papers[0].paper_id, paper_ids))
        return {
            "status": "success",
            "ready_paper_ids": paper_ids,
            "summary": "Resumed paper.",
        }

    queue = IngestionJobQueue(prepare_func=prepare, store=store)
    queue.start()

    completed = _wait_for_terminal_job(queue, job.job_id)
    assert completed is not None
    assert completed.status == "success"
    assert calls == [("paper:resume", ["paper:resume"])]


def _wait_for_terminal_job(queue, job_id, timeout_seconds=2.0):
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        job = queue.get(job_id)
        if job is not None and job.status in {"success", "partial_success", "failed"}:
            return job
        time.sleep(0.01)
    return queue.get(job_id)
