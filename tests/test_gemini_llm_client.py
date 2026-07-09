import pytest

from app.llm.client import GeminiLLMClient, _extract_gemini_response_text


def test_gemini_llm_client_requires_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    client = GeminiLLMClient()

    with pytest.raises(RuntimeError, match="GEMINI_API_KEY is not set"):
        client.generate("Summarize this paper.")


def test_gemini_llm_client_explains_missing_sdk(monkeypatch):
    client = GeminiLLMClient(api_key="test-key")

    def fake_import(name, *args, **kwargs):
        if name == "google":
            raise ImportError("missing google-genai")
        return original_import(name, *args, **kwargs)

    original_import = __import__
    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(RuntimeError, match="google-genai is not installed"):
        client.generate("Summarize this paper.")


def test_extract_gemini_response_text_uses_text_property():
    class Response:
        text = "  Gemini summary  "

    assert _extract_gemini_response_text(Response()) == "Gemini summary"


def test_extract_gemini_response_text_falls_back_to_candidate_parts():
    class Part:
        text = "summary from candidate"

    class Content:
        parts = [Part()]

    class Candidate:
        content = Content()

    class Response:
        candidates = [Candidate()]

    assert _extract_gemini_response_text(Response()) == "summary from candidate"
