import time

from app.agent.state import Paper
from app.services.ingestion_jobs import IngestionJobQueue


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


def _wait_for_terminal_job(queue, job_id, timeout_seconds=2.0):
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        job = queue.get(job_id)
        if job is not None and job.status in {"success", "partial_success", "failed"}:
            return job
        time.sleep(0.01)
    return queue.get(job_id)
