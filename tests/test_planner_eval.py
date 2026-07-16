from app.agent.planner_eval import evaluate_planner_cases, summarize_eval_results


def test_default_planner_eval_suite_passes():
    results = evaluate_planner_cases()
    summary = summarize_eval_results(results)

    assert summary["failed"] == 0
    assert summary["passed"] == summary["total"]


def test_default_planner_eval_covers_missing_kb_then_discovery_flow():
    results = {result.name: result for result in evaluate_planner_cases()}

    result = results["missing_kb_discovers_prepares_and_retrieves"]

    assert result.passed is True
    assert result.tool_sequence == [
        "retrieve_evidence",
        "discover_papers",
        "ensure_papers_retrievable",
        "retrieve_evidence",
    ]
