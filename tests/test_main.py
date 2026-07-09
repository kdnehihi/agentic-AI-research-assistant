from app import main as app_main


class FakeGeminiClient:
    def generate(self, prompt: str, **kwargs) -> str:
        return "Gemini generated summary."


class FailingGeminiClient:
    def generate(self, prompt: str, **kwargs) -> str:
        raise RuntimeError("quota exceeded")


def test_main_runs_gemini_summary_workflow_without_network(monkeypatch, capsys):
    def fake_run_workflow(self, workflow):
        assert workflow == app_main.SEARCH_AND_FILTER_WORKFLOW
        self.registry.execute("search_fake_papers", self.state)
        self.registry.execute("deduplicate_papers", self.state)
        self.registry.execute("rank_papers", self.state)
        self.registry.execute("filter_relevant_papers", self.state)

    monkeypatch.setattr(app_main.AgentRunner, "run_workflow", fake_run_workflow)
    monkeypatch.setattr(app_main, "GeminiLLMClient", FakeGeminiClient)

    app_main.main()

    captured = capsys.readouterr()
    assert "===== FINAL REPORT =====" in captured.out
    assert "Gemini generated summary." in captured.out


def test_main_falls_back_to_abstract_summary_when_llm_fails(monkeypatch, capsys):
    def fake_run_workflow(self, workflow):
        assert workflow == app_main.SEARCH_AND_FILTER_WORKFLOW
        self.registry.execute("search_fake_papers", self.state)
        self.registry.execute("deduplicate_papers", self.state)
        self.registry.execute("rank_papers", self.state)
        self.registry.execute("filter_relevant_papers", self.state)

    monkeypatch.setattr(app_main.AgentRunner, "run_workflow", fake_run_workflow)
    monkeypatch.setattr(app_main, "GeminiLLMClient", FailingGeminiClient)

    app_main.main()

    captured = capsys.readouterr()
    assert "===== FINAL REPORT =====" in captured.out
    assert "# Paper Research Report" in captured.out
    assert "reinforcement learning from human feedback" in captured.out.lower()
