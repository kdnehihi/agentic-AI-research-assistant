import json

from app.agent.state import AgentState, Paper
from app.storage.paper_store import PaperStore
from app.tools.chunking_tools import (
    Section,
    chunk_section,
    chunk_selected_papers_by_section,
    chunk_text_by_sections,
    detect_sections,
    save_chunks_jsonl,
)


def test_detect_sections_finds_inline_research_headings():
    text = (
        "Title and author block. Abstract This is the abstract. "
        "Introduction This is the introduction. "
        "2 Related Work This is related work. "
        "3 Methodology This is methodology. "
        "Conclusion This is the conclusion."
    )

    sections = detect_sections(text)

    assert [section.title for section in sections] == [
        "Front Matter",
        "Abstract",
        "Introduction",
        "Related Work",
        "Methodology",
        "Conclusion",
    ]
    assert sections[1].text == "This is the abstract."


def test_detect_sections_splits_abstract_after_title_metadata():
    text = (
        "DF-RAG: Query-Aware Diversity for Retrieval-Augmented Generation "
        "Alice Researcher University alice@example.com Abstract "
        "Retrieval-augmented generation needs diverse evidence. "
        "Introduction This paper studies query-aware retrieval."
    )

    sections = detect_sections(text)

    assert [section.title for section in sections] == [
        "Front Matter",
        "Abstract",
        "Introduction",
    ]
    assert sections[0].text.startswith("DF-RAG")
    assert sections[1].text == "Retrieval-augmented generation needs diverse evidence."


def test_chunk_section_uses_target_size_and_overlap():
    section = Section(
        title="Method",
        text=" ".join(f"word{i}" for i in range(1800)),
    )

    chunks = chunk_section(
        section=section,
        paper_id="arxiv:test",
        min_chunk_words=700,
        target_chunk_words=850,
        max_chunk_words=900,
        overlap_words=200,
    )

    assert len(chunks) == 3
    assert [chunk.word_count for chunk in chunks] == [850, 850, 500]
    assert chunks[0].start_word == 0
    assert chunks[1].start_word == 650
    assert chunks[2].start_word == 1300
    assert chunks[0].section == "Method"
    assert chunks[0].section_index == 0
    assert chunks[1].section_chunk_index == 1
    assert chunks[0].section_word_count == 1800


def test_chunk_text_by_sections_keeps_chunk_metadata():
    text = (
        "Abstract " + " ".join(f"a{i}" for i in range(50)) + ". "
        "Introduction " + " ".join(f"I{i}" for i in range(950))
    )

    chunks = chunk_text_by_sections(
        text=text,
        paper_id="arxiv:chunk",
        min_chunk_words=100,
        target_chunk_words=300,
        max_chunk_words=350,
        overlap_words=50,
    )

    assert chunks[0].chunk_id == "arxiv:chunk::chunk:0"
    assert chunks[0].section == "Abstract"
    assert chunks[1].section == "Introduction"
    assert chunks[0].section_index == 0
    assert chunks[1].section_index == 1
    assert chunks[1].section_chunk_index == 0
    assert all(chunk.word_count <= 350 for chunk in chunks)


def test_save_chunks_jsonl_writes_one_json_object_per_line(tmp_path):
    section = Section(title="Abstract", text=" ".join(f"word{i}" for i in range(20)))
    chunks = chunk_section(
        section=section,
        paper_id="arxiv:jsonl",
        min_chunk_words=10,
        target_chunk_words=10,
        max_chunk_words=12,
        overlap_words=2,
    )

    output_path = save_chunks_jsonl(chunks, tmp_path / "chunks.jsonl")
    rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]

    assert len(rows) == len(chunks)
    assert rows[0]["paper_id"] == "arxiv:jsonl"
    assert rows[0]["section"] == "Abstract"
    assert rows[0]["section_index"] == 0
    assert rows[0]["section_chunk_index"] == 0
    assert rows[0]["start_word"] == 0
    assert rows[0]["section_word_count"] == 20


def test_chunk_selected_papers_by_section_saves_chunks_and_updates_state(tmp_path):
    store = PaperStore(
        db_path=tmp_path / "metadata" / "papers.sqlite3",
        papers_dir=tmp_path / "papers",
    )
    paper = Paper(
        paper_id="arxiv:agent",
        title="Agent Paper",
        source="arxiv",
        url="https://arxiv.org/abs/agent",
    )
    state = AgentState(topic="chunk agent", max_papers=1)
    state.set_selected_papers([paper])

    clean_text_path = store.save_clean_text(
        paper_id=paper.paper_id,
        text=(
            "Abstract " + " ".join(f"a{i}" for i in range(80)) + ". "
            "Introduction " + " ".join(f"I{i}" for i in range(1000))
        ),
    )
    state.set_paper_text_paths({paper.paper_id: str(clean_text_path)})

    observation = chunk_selected_papers_by_section(
        state=state,
        file_store=store,
        min_chunk_words=100,
        target_chunk_words=300,
        max_chunk_words=350,
        overlap_words=50,
    )

    chunks_path = store.paper_dir(paper.paper_id) / "chunks.jsonl"
    rows = [
        json.loads(line)
        for line in chunks_path.read_text(encoding="utf-8").splitlines()
    ]

    assert observation["status"] == "success"
    assert observation["processed"] == 1
    assert observation["chunks"] == len(rows)
    assert state.paper_chunk_paths == {paper.paper_id: str(chunks_path)}
    assert rows[0]["section"] == "Abstract"
    assert rows[0]["chunk_id"] == "arxiv:agent::chunk:0"
    assert rows[0]["section_index"] == 0
    assert rows[0]["section_chunk_index"] == 0
