from scripts.dynamic_planner_smoke_run import _compact_final_answer


def test_compact_final_answer_trims_full_evidence_text():
    compact = _compact_final_answer(
        {
            "answer": "Summary [E1].",
            "evidence_chunks": [
                {
                    "evidence_id": "E1",
                    "chunk_id": "paper::chunk:1",
                    "paper_id": "paper",
                    "title": "A Paper",
                    "section": "Limitations",
                    "section_group": "discussion",
                    "rank": 1,
                    "final_score": 0.91,
                    "text": "long full text should not be printed",
                }
            ],
        }
    )

    assert compact["answer"] == "Summary [E1]."
    assert compact["evidence_chunks"] == [
        {
            "evidence_id": "E1",
            "chunk_id": "paper::chunk:1",
            "paper_id": "paper",
            "title": "A Paper",
            "section": "Limitations",
            "section_group": "discussion",
            "rank": 1,
            "final_score": 0.91,
        }
    ]
