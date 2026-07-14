from app.agent.observation_factory import ObservationFactory


def test_raw_success_normalizes_to_planner_success():
    observation = ObservationFactory().from_tool_result(
        tool_name="discover_papers",
        raw_result={
            "status": "success",
            "selected_paper_ids": ["p1"],
            "summary": "ok",
        },
    )

    assert observation.status == "success"
    assert observation.state_changes["known_paper_ids_added"] == ["p1"]


def test_partial_success_is_preserved():
    observation = ObservationFactory().from_tool_result(
        tool_name="save_papers_to_kb",
        raw_result={
            "status": "partial_success",
            "inserted_paper_ids": ["p1"],
            "failed": [{"paper_id": "p2", "error_type": "x"}],
            "summary": "partial",
        },
    )

    assert observation.status == "partial_success"
    assert observation.state_changes["saved_paper_ids_added"] == ["p1"]


def test_missing_prerequisite_becomes_prerequisite_missing():
    observation = ObservationFactory().from_tool_result(
        tool_name="retrieve_evidence",
        raw_result={
            "status": "failed",
            "error_type": "paper_not_retrievable",
            "missing_paper_ids": ["p1"],
            "summary": "missing",
        },
    )

    assert observation.status == "prerequisite_missing"
    assert observation.retryable is True
    assert observation.error_type == "paper_not_retrievable"


def test_failed_execution_becomes_tool_error():
    observation = ObservationFactory().from_tool_result(
        tool_name="retrieve_evidence",
        raw_result={
            "status": "failed",
            "error_type": "retrieval_failure",
            "summary": "boom",
        },
    )

    assert observation.status == "tool_error"
    assert observation.error_type == "retrieval_failure"


def test_skipped_already_complete_becomes_success():
    observation = ObservationFactory().from_tool_result(
        tool_name="ensure_papers_retrievable",
        raw_result={
            "status": "skipped",
            "already_ready_paper_ids": ["p1"],
            "summary": "already indexed",
        },
    )

    assert observation.status == "success"
    assert observation.state_changes["retrievable_paper_ids_added"] == ["p1"]


def test_absent_status_is_safe_tool_error_and_compacts_sensitive_fields():
    observation = ObservationFactory().from_tool_result(
        tool_name="retrieve_evidence",
        raw_result={
            "summary": "x",
            "evidence": [
                {
                    "chunk_id": "c1",
                    "text": "word " * 500,
                    "raw": {"stack_trace": "secret"},
                    "api_key": "secret",
                }
            ],
        },
    )

    assert observation.status == "tool_error"
    evidence = observation.result["evidence"][0]
    assert "raw" not in evidence
    assert "api_key" not in evidence
    assert len(evidence["text"]) < 1300

