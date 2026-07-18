from app import main as app_main


class FakeOpenAIClient:
    def generate(self, prompt: str, **kwargs) -> str:
        return "OpenAI generated summary."


class FailingOpenAIClient:
    def generate(self, prompt: str, **kwargs) -> str:
        raise RuntimeError("quota exceeded")


def test_main_runs_openai_summary_workflow_without_network(monkeypatch, capsys):
    def fake_run_workflow(self, workflow):
        assert workflow == app_main.SEARCH_AND_FILTER_WORKFLOW
        assert "fetch_selected_papers" in workflow
        self.registry.execute("search_fake_papers", self.state)
        self.registry.execute("deduplicate_papers", self.state)
        self.registry.execute("rank_papers_by_similarity", self.state)
        self.registry.execute("filter_relevant_papers", self.state)

    monkeypatch.setattr(app_main.AgentRunner, "run_workflow", fake_run_workflow)
    monkeypatch.setattr(
        app_main,
        "create_default_llm_client",
        lambda: FakeOpenAIClient(),
    )
    monkeypatch.setattr(app_main, "TOPIC", "RLHF RLVR reasoning models")
    monkeypatch.setattr(
        app_main,
        "save_selected_papers_to_kb",
        lambda state: {"status": "success"},
    )

    app_main.main()

    captured = capsys.readouterr()
    assert "===== FINAL REPORT =====" in captured.out
    assert "===== KNOWLEDGE BASE SAVE REPORT =====" in captured.out
    assert "OpenAI generated summary." in captured.out


def test_main_falls_back_to_abstract_summary_when_llm_fails(monkeypatch, capsys):
    def fake_run_workflow(self, workflow):
        assert workflow == app_main.SEARCH_AND_FILTER_WORKFLOW
        assert "fetch_selected_papers" in workflow
        self.registry.execute("search_fake_papers", self.state)
        self.registry.execute("deduplicate_papers", self.state)
        self.registry.execute("rank_papers_by_similarity", self.state)
        self.registry.execute("filter_relevant_papers", self.state)

    monkeypatch.setattr(app_main.AgentRunner, "run_workflow", fake_run_workflow)
    monkeypatch.setattr(
        app_main,
        "create_default_llm_client",
        lambda: FailingOpenAIClient(),
    )
    monkeypatch.setattr(app_main, "TOPIC", "RLHF RLVR reasoning models")
    monkeypatch.setattr(
        app_main,
        "save_selected_papers_to_kb",
        lambda state: {"status": "success"},
    )
    app_main.main()

    captured = capsys.readouterr()
    assert "===== FINAL REPORT =====" in captured.out
    assert "===== KNOWLEDGE BASE SAVE REPORT =====" in captured.out
    assert "# Paper Research Report" in captured.out
    assert "- Summary:" in captured.out
