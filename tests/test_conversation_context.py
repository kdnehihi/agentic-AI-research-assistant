from app.conversations.context_builder import ConversationContextBuilder
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
    assert context.active_paper_ids == ["p3", "p4", "p5"]


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
