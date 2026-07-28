from datetime import datetime, timezone

from app.conversations.context_builder import (
    ConversationContextBuilder,
    active_paper_ids_from_messages,
)
from app.conversations.models import ConversationMessage
from app.conversations.sqlite_repository import SQLiteConversationRepository


def test_context_builder_uses_summary_recent_window_and_active_papers(tmp_path):
    repo = SQLiteConversationRepository(tmp_path / "conversations.sqlite3")
    thread = repo.create_thread(title="Context")
    for index in range(6):
        repo.append_message(
            thread_id=thread.thread_id,
            role="user" if index % 2 == 0 else "assistant",
            content=f"message {index}",
            metadata_json={"paper_ids": [f"p{index}"]},
        )
    repo.update_summary(
        thread.thread_id,
        "Earlier conversation summary.",
        summary_updated_at=thread.created_at,
    )

    context = ConversationContextBuilder(repo, recent_message_limit=3).build(
        thread_id=thread.thread_id
    )

    assert context.conversation_summary == "Earlier conversation summary."
    assert [message.content for message in context.recent_messages] == [
        "message 3",
        "message 4",
        "message 5",
    ]
    assert context.active_paper_ids == ["p5", "p4", "p3"]


def test_context_builder_excludes_current_message_with_before_sequence(tmp_path):
    repo = SQLiteConversationRepository(tmp_path / "conversations.sqlite3")
    thread = repo.create_thread(title="Before")
    first = repo.append_message(thread_id=thread.thread_id, role="user", content="first")
    second = repo.append_message(thread_id=thread.thread_id, role="user", content="second")

    context = ConversationContextBuilder(repo, recent_message_limit=10).build(
        thread_id=thread.thread_id,
        before_sequence=second.sequence_number,
    )

    assert [message.sequence_number for message in context.recent_messages] == [
        first.sequence_number
    ]


def test_active_paper_ids_prefers_recent_high_priority_context_update():
    older = ConversationMessage(
        message_id="m1",
        thread_id="t1",
        role="assistant",
        content="Older answer.",
        created_at=datetime.now(timezone.utc),
        sequence_number=1,
        metadata_json={"active_paper_ids": ["old-1", "old-2"]},
    )
    latest_ready = ConversationMessage(
        message_id="m2",
        thread_id="t1",
        role="system",
        content="Prepared papers for RAG: new-1",
        created_at=datetime.now(timezone.utc),
        sequence_number=2,
        metadata_json={
            "message_type": "paper_context_update",
            "context_priority": 100,
            "active_paper_ids": ["new-1"],
        },
    )

    assert active_paper_ids_from_messages([older, latest_ready]) == ["new-1"]
