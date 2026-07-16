import sqlite3

from app.conversations.sqlite_repository import SQLiteConversationRepository


def test_conversation_repository_thread_messages_summary_and_delete(tmp_path):
    repo = SQLiteConversationRepository(tmp_path / "conversations.sqlite3")
    thread = repo.create_thread(title="Agentic RAG")

    first = repo.append_message(
        thread_id=thread.thread_id,
        role="user",
        content="Find papers.",
        metadata_json={"paper_ids": ["p1"], "api_key": "secret"},
    )
    second = repo.append_message(
        thread_id=thread.thread_id,
        role="assistant",
        content="Found p1.",
        metadata_json={"paper_ids": ["p1"]},
    )
    messages = repo.list_messages(thread.thread_id)

    assert [message.sequence_number for message in messages] == [1, 2]
    assert messages[0].message_id == first.message_id
    assert messages[1].message_id == second.message_id
    assert "api_key" not in messages[0].metadata_json

    repo.update_summary(thread.thread_id, "Discussed p1.", summary_updated_at=thread.created_at)
    assert repo.get_thread(thread.thread_id).conversation_summary == "Discussed p1."

    assert repo.delete_thread(thread.thread_id) is True
    assert repo.get_thread(thread.thread_id) is None
    assert repo.list_messages(thread.thread_id) == []


def test_conversation_repository_thread_isolation_and_foreign_keys(tmp_path):
    repo = SQLiteConversationRepository(tmp_path / "conversations.sqlite3")
    t1 = repo.create_thread(title="T1")
    t2 = repo.create_thread(title="T2")
    repo.append_message(thread_id=t1.thread_id, role="user", content="A")
    repo.append_message(thread_id=t2.thread_id, role="user", content="B")

    assert [message.content for message in repo.list_messages(t1.thread_id)] == ["A"]
    assert [message.content for message in repo.list_messages(t2.thread_id)] == ["B"]

    try:
        repo.append_message(thread_id="missing", role="user", content="bad")
    except sqlite3.IntegrityError:
        pass
    else:
        raise AssertionError("Expected foreign-key failure for missing thread")


def test_agent_run_and_step_persistence(tmp_path):
    repo = SQLiteConversationRepository(tmp_path / "conversations.sqlite3")
    thread = repo.create_thread(title="Trace")
    message = repo.append_message(thread_id=thread.thread_id, role="user", content="Q")
    run = repo.start_run(
        thread_id=thread.thread_id,
        user_request_message_id=message.message_id,
    )
    repo.append_step(
        run_id=run.run_id,
        step_number=1,
        node_name="execute_tool",
        decision_type="call_tool",
        tool_name="retrieve_evidence",
        arguments_json={"query": "Q", "authorization": "secret"},
        observation_status="success",
        observation_json={"retrieved": 1},
        latency_ms=12.5,
    )
    repo.complete_run(run.run_id, latency_ms=20.0)

    stored_run = repo.get_run(run.run_id)
    steps = repo.list_steps(run.run_id)

    assert stored_run.status == "completed"
    assert stored_run.latency_ms == 20.0
    assert steps[0].tool_name == "retrieve_evidence"
    assert "authorization" not in steps[0].arguments_json


def test_failed_run_persistence(tmp_path):
    repo = SQLiteConversationRepository(tmp_path / "conversations.sqlite3")
    thread = repo.create_thread(title="Failure")
    message = repo.append_message(thread_id=thread.thread_id, role="user", content="Q")
    run = repo.start_run(
        thread_id=thread.thread_id,
        user_request_message_id=message.message_id,
    )

    repo.fail_run(
        run.run_id,
        error_type="PlannerError",
        error_message="boom",
        latency_ms=5.0,
    )

    stored_run = repo.get_run(run.run_id)
    assert stored_run.status == "failed"
    assert stored_run.error_type == "PlannerError"
    assert stored_run.error_message == "boom"
