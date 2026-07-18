import sys
import types

import pytest

from app.llm.client import (
    LangChainOpenAILLMClient,
    MissingOpenAIAPIKeyError,
    OpenAILLMClient,
    _extract_langchain_response_text,
    _extract_response_text,
    create_default_llm_client,
)


def test_langchain_openai_llm_client_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = LangChainOpenAILLMClient(load_env=False)

    with pytest.raises(MissingOpenAIAPIKeyError):
        client.generate("Summarize this paper.")


def test_langchain_openai_llm_client_invokes_chat_model(monkeypatch):
    calls = {}

    class FakeResponse:
        content = "  LangChain summary  "

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            calls["init"] = kwargs

        def invoke(self, prompt, **kwargs):
            calls["prompt"] = prompt
            calls["kwargs"] = kwargs
            return FakeResponse()

    fake_module = types.SimpleNamespace(ChatOpenAI=FakeChatOpenAI)
    monkeypatch.setitem(sys.modules, "langchain_openai", fake_module)

    client = LangChainOpenAILLMClient(
        api_key="test-key",
        model="test-model",
        load_env=False,
    )

    assert client.generate("hello", temperature=0) == "LangChain summary"
    assert calls["init"]["api_key"] == "test-key"
    assert calls["init"]["model"] == "test-model"
    assert calls["prompt"] == "hello"
    assert calls["kwargs"] == {"temperature": 0}


def test_create_default_llm_client_uses_langchain_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "langchain_openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    client = create_default_llm_client()

    assert isinstance(client, LangChainOpenAILLMClient)


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


def test_extract_langchain_response_text_handles_string_and_parts():
    class TextPart:
        text = "part text"

    class Response:
        content = [{"text": "dict text"}, TextPart()]

    assert _extract_langchain_response_text("  direct text  ") == "direct text"
    assert _extract_langchain_response_text(Response()) == "dict text\npart text"
