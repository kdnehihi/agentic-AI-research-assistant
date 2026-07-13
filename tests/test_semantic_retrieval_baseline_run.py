import json

from scripts.semantic_retrieval_baseline_run import (
    build_eval_cases_for_chunks,
    build_eval_cases_from_existing_chunks,
    bm25_scores_for_query,
    final_local_score,
    metadata_intent_score,
)
from app.retrieval.evaluation import RetrievalEvalCase


def test_build_eval_cases_for_chunks_supports_basic_and_expanded_queries():
    chunks = [
        _chunk("paper_1::chunk:0", "Introduction"),
        _chunk("paper_1::chunk:1", "Method"),
        _chunk("paper_1::chunk:2", "Method"),
    ]

    basic_cases = build_eval_cases_for_chunks(
        chunks=chunks,
        paper_id="paper_1",
        query_style="basic",
        max_cases=3,
    )
    expanded_cases = build_eval_cases_for_chunks(
        chunks=chunks,
        paper_id="paper_1",
        query_style="expanded",
        max_cases=3,
    )

    assert basic_cases[0].query == "What problem motivates this paper?"
    assert expanded_cases[0].query == (
        "What research problem, limitation, or gap motivates the proposed work?"
    )
    assert basic_cases[1].relevant_chunk_ids == (
        "paper_1::chunk:1",
        "paper_1::chunk:2",
    )
    assert basic_cases[1].section_groups == ("method",)


def test_build_eval_cases_from_existing_chunks_selects_requested_paper(tmp_path):
    papers_dir = tmp_path / "papers"
    first_paper_dir = papers_dir / "paper_1"
    second_paper_dir = papers_dir / "paper_2"
    first_paper_dir.mkdir(parents=True)
    second_paper_dir.mkdir(parents=True)
    _write_chunks(first_paper_dir / "chunks.jsonl", [_chunk("paper_1::chunk:0", "Abstract")])
    _write_chunks(second_paper_dir / "chunks.jsonl", [_chunk("paper_2::chunk:0", "Abstract")])

    cases_by_style = build_eval_cases_from_existing_chunks(
        papers_dir=papers_dir,
        paper_ids=("paper_2",),
        query_style="both",
    )

    assert set(cases_by_style) == {"basic", "expanded"}
    assert cases_by_style["basic"][0].paper_id == "paper_2"
    assert cases_by_style["expanded"][0].case_id == "paper_2::expanded::abstract"


def test_hybrid_helpers_score_bm25_and_metadata():
    rows = [
        {
            "chunk_id": "method",
            "embedding_text": "Section: Method\nContent:\nretrieval pipeline method",
            "section_group": "method",
        },
        {
            "chunk_id": "intro",
            "embedding_text": "Section: Introduction\nContent:\nresearch problem motivation",
            "section_group": "introduction",
        },
    ]
    case = RetrievalEvalCase(
        query="What method or approach is proposed?",
        relevant_chunk_ids=("method",),
        section_groups=("method",),
    )

    bm25_scores = bm25_scores_for_query("method retrieval", rows)

    assert bm25_scores["method"] > bm25_scores["intro"]
    assert metadata_intent_score(case.query, rows[0]) == 1.0
    assert metadata_intent_score(case.query, rows[1]) == 0.0
    assert final_local_score(
        ranking_mode="hybrid",
        semantic_score=0.5,
        bm25_score=1.0,
        metadata_score=1.0,
        semantic_weight=0.65,
        bm25_weight=0.25,
        metadata_weight=0.10,
    ) > 0.5


def _chunk(chunk_id: str, section: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "paper_id": chunk_id.split("::", 1)[0],
        "section": section,
        "section_index": 0,
        "chunk_index": 0,
        "section_chunk_index": 0,
        "start_word": 0,
        "end_word": 5,
        "word_count": 5,
        "text": f"{section} text for semantic retrieval.",
    }


def _write_chunks(path, chunks):
    path.write_text(
        "\n".join(json.dumps(chunk) for chunk in chunks),
        encoding="utf-8",
    )
