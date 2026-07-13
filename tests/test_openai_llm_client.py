import pytest

from app.llm.client import (
    MissingOpenAIAPIKeyError,
    OpenAILLMClient,
    _extract_response_text,
)


def test_openai_llm_client_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = OpenAILLMClient(load_env=False)

    with pytest.raises(MissingOpenAIAPIKeyError):
        client.generate("Summarize this paper.")


def test_extract_response_text_uses_output_text():
    class Response:
        output_text = "  concise summary  "

    assert _extract_response_text(Response()) == "concise summary"


def test_extract_response_text_falls_back_to_output_content():
    class Content:
        text = "summary from content"

    class Item:
        content = [Content()]

    class Response:
        output = [Item()]

    assert _extract_response_text(Response()) == "summary from content"
