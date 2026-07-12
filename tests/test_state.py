import pytest

from app.agent.state import AgentState, Paper, PaperSummary, ToolLog


def test_create_agent_state():
    state = AgentState(
        topic="RLHF RLVR reasoning models",
        max_papers=3,
    )

    assert state.run_id is not None
    assert state.topic == "RLHF RLVR reasoning models"
    assert state.max_papers == 3
    assert state.status == "initialized"
    assert state.current_step is None
    assert state.searched_sources == []
    assert state.candidate_papers == []
    assert state.selected_papers == []
    assert state.paper_summaries == []
    assert state.paper_text_paths == {}
    assert state.paper_chunk_paths == {}
    assert state.paper_embedding_paths == {}
    assert state.report is None
    assert state.eval_results is None
    assert state.tool_call_count == 0
    assert state.tool_logs == []
    assert state.error is None


def test_create_paper():
    paper = Paper(
        paper_id="arxiv:2501.00001",
        title="RLHF for Reasoning Models",
        authors=["Alice Nguyen", "Bob Chen"],
        source="arxiv",
        url="https://arxiv.org/abs/2501.00001",
        abstract="This paper studies RLHF for improving reasoning models.",
        published_date="2025-01-01",
        score=3.5,
        score_components={"semantic": 0.7},
        relevant_reasons=["Mentions RLHF", "Focuses on reasoning"],
    )

    assert paper.paper_id == "arxiv:2501.00001"
    assert paper.title == "RLHF for Reasoning Models"
    assert paper.authors == ["Alice Nguyen", "Bob Chen"]
    assert paper.source == "arxiv"
    assert paper.url == "https://arxiv.org/abs/2501.00001"
    assert paper.score == 3.5
    assert paper.score_components == {"semantic": 0.7}
    assert len(paper.relevant_reasons) == 2


def test_create_paper_summary():
    summary = PaperSummary(
        paper_id="arxiv:2501.00001",
        title="RLHF for Reasoning Models",
        one_sentence_summary="This paper studies RLHF for reasoning models.",
        detailed_summary="The paper explores how feedback-based optimization improves reasoning behavior.",
        method="RLHF",
        main_contribution="Applies RLHF to reasoning-centric language models.",
        why_it_matters="It helps improve reasoning quality.",
        limitations="Only evaluated on limited reasoning benchmarks.",
        based_on="abstract_only",
    )

    assert summary.paper_id == "arxiv:2501.00001"
    assert summary.title == "RLHF for Reasoning Models"
    assert summary.method == "RLHF"
    assert summary.based_on == "abstract_only"


def test_agent_state_transitions():
    state = AgentState(
        topic="RLHF reasoning models",
        max_papers=2,
    )

    paper = Paper(
        paper_id="arxiv:2501.00001",
        title="RLHF for Reasoning Models",
        authors=["Alice Nguyen"],
        source="arxiv",
        url="https://arxiv.org/abs/2501.00001",
        abstract="This paper studies RLHF for improving reasoning models.",
    )

    summary = PaperSummary(
        paper_id="arxiv:2501.00001",
        title="RLHF for Reasoning Models",
        one_sentence_summary="This paper studies RLHF for reasoning models.",
    )

    state.mark_running()
    assert state.status == "running"

    state.set_current_step("SEARCH_ARXIV")
    assert state.current_step == "SEARCH_ARXIV"

    state.add_searched_source("arxiv")
    state.add_searched_source("arxiv")
    assert state.searched_sources == ["arxiv"]

    state.add_candidate_paper(paper)
    assert len(state.candidate_papers) == 1
    assert state.candidate_papers[0].title == "RLHF for Reasoning Models"

    state.add_selected_paper(paper)
    assert len(state.selected_papers) == 1

    state.add_paper_summary(summary)
    assert len(state.paper_summaries) == 1

    state.set_paper_text_paths({"arxiv:2501.00001": "data/papers/text.txt"})
    assert state.paper_text_paths == {"arxiv:2501.00001": "data/papers/text.txt"}

    state.set_paper_chunk_paths({"arxiv:2501.00001": "data/papers/chunks.jsonl"})
    assert state.paper_chunk_paths == {"arxiv:2501.00001": "data/papers/chunks.jsonl"}

    state.set_paper_embedding_paths({"arxiv:2501.00001": "data/papers/embeddings.jsonl"})
    assert state.paper_embedding_paths == {
        "arxiv:2501.00001": "data/papers/embeddings.jsonl"
    }

    state.set_report("# Research Report\n\nThis is a test report.")
    assert state.report is not None
    assert "Research Report" in state.report

    eval_results = {
        "task_success": True,
        "num_selected_papers": 1,
        "duplicate_count": 0,
        "has_sources": True,
        "format_pass": True,
    }

    state.set_eval_results(eval_results)
    assert state.eval_results == eval_results

    state.mark_success()
    assert state.status == "success"
    assert state.current_step is None


def test_set_candidate_and_selected_papers():
    state = AgentState(topic="RLHF", max_papers=2)

    papers = [
        Paper(
            paper_id="arxiv:1",
            title="Paper 1",
            authors=["A"],
            source="arxiv",
            url="https://arxiv.org/abs/1",
        ),
        Paper(
            paper_id="arxiv:2",
            title="Paper 2",
            authors=["B"],
            source="arxiv",
            url="https://arxiv.org/abs/2",
        ),
    ]

    state.set_candidate_papers(papers)
    assert len(state.candidate_papers) == 2

    state.set_selected_papers(papers[:1])
    assert len(state.selected_papers) == 1
    assert state.selected_papers[0].title == "Paper 1"


def test_add_tool_log():
    state = AgentState(topic="RLHF")

    tool_log = ToolLog(
        tool_name="SEARCH_ARXIV",
        input_args={
            "query": "RLHF reasoning models",
            "max_results": 10,
        },
        status="success",
        output_summary="Found 10 papers.",
        latency_ms=120.5,
    )

    state.add_tool_log(tool_log)

    assert state.tool_call_count == 1
    assert len(state.tool_logs) == 1
    assert state.tool_logs[0].tool_name == "SEARCH_ARXIV"
    assert state.tool_logs[0].status == "success"
    assert state.tool_logs[0].output_summary == "Found 10 papers."


def test_can_call_tools():
    state = AgentState(topic="RLHF")

    assert state.can_call_tools(max_tool_calls=3) is False

    state.mark_running()
    assert state.can_call_tools(max_tool_calls=3) is True

    state.add_tool_log(
        ToolLog(
            tool_name="TOOL_1",
            status="success",
            output_summary="ok",
        )
    )
    state.add_tool_log(
        ToolLog(
            tool_name="TOOL_2",
            status="success",
            output_summary="ok",
        )
    )
    state.add_tool_log(
        ToolLog(
            tool_name="TOOL_3",
            status="success",
            output_summary="ok",
        )
    )

    assert state.tool_call_count == 3
    assert state.can_call_tools(max_tool_calls=3) is False


def test_mark_failed():
    state = AgentState(topic="RLHF")

    state.mark_running()
    state.set_current_step("SEARCH_ARXIV")

    state.mark_failed("arXiv API failed.")

    assert state.status == "failed"
    assert state.current_step is None
    assert state.error == "arXiv API failed."


def test_add_error_marks_failed():
    state = AgentState(topic="RLHF")

    state.mark_running()
    state.add_error("Something went wrong.")

    assert state.status == "failed"
    assert state.error == "Something went wrong."
    assert state.current_step is None


def test_debug_summary_returns_string():
    state = AgentState(topic="RLHF")
    state.mark_running()
    state.set_current_step("TEST_STEP")

    summary = state.debug_summary()

    assert isinstance(summary, str)
    assert "Run ID:" in summary
    assert "Topic: RLHF" in summary
    assert "Status: running" in summary
    assert "Current Step: TEST_STEP" in summary


def test_extra_field_is_forbidden():
    with pytest.raises(Exception):
        Paper(
            title="Invalid Paper",
            source="arxiv",
            url="https://arxiv.org/abs/test",
            random_field="this should not be allowed",
        )


def test_invalid_status_assignment_is_blocked():
    state = AgentState(topic="RLHF")

    with pytest.raises(Exception):
        state.status = "done"
