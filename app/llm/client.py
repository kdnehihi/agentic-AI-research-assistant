from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Protocol, Any

from dotenv import load_dotenv


class LLMClient(Protocol):
    """
    A protocol that defines the interface for an LLM agent.
    This is used to ensure that any agent implementation adheres to the expected methods and properties.
    """

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """
        Generate a response based on the given prompt.

        Args:
            prompt (str): The input prompt for the LLM.
            **kwargs (Any): Additional keyword arguments that may be required by specific implementations.

        Returns:
            str: The generated response from the LLM.
        """
        ...

    def stream_generate(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        """Generate text incrementally when the backend supports streaming."""

        ...


class MissingOpenAIAPIKeyError(RuntimeError):
    """Raised when a real OpenAI client is used without an API key."""


class LangChainOpenAILLMClient:
    """
    LangChain-backed OpenAI chat client.

    This is the default generation backend. The project keeps its small
    `LLMClient.generate(prompt)` protocol so planner, answer generation, and
    tools do not depend directly on LangChain concepts.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        load_env: bool = True,
        **model_kwargs: Any,
    ) -> None:
        if load_env:
            load_dotenv()
        self.api_key = _clean_env_secret(api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model or os.getenv("OPENAI_MODEL") or "gpt-4.1-mini"
        self.model_kwargs = dict(model_kwargs)
        self._client: Any | None = None

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text through LangChain's ChatOpenAI wrapper."""

        if not self.api_key:
            raise MissingOpenAIAPIKeyError(
                "OPENAI_API_KEY is not set. Add it to your environment before "
                "using LangChainOpenAILLMClient."
            )

        model_override = kwargs.pop("model", None)
        client = self._get_client(model=model_override)
        response = client.invoke(prompt, **kwargs)
        return _extract_langchain_response_text(response)

    def stream_generate(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        """Stream text through LangChain's ChatOpenAI wrapper."""

        if not self.api_key:
            raise MissingOpenAIAPIKeyError(
                "OPENAI_API_KEY is not set. Add it to your environment before "
                "using LangChainOpenAILLMClient."
            )

        model_override = kwargs.pop("model", None)
        client = self._get_client(model=model_override)
        try:
            for chunk in client.stream(prompt, **kwargs):
                text = _extract_langchain_response_text(
                    chunk,
                    strip_whitespace=False,
                )
                if text:
                    yield text
        except Exception:
            yield self.generate(prompt, **kwargs)

    def _get_client(self, *, model: str | None = None) -> Any:
        if model is not None:
            return self._build_client(model=model)
        if self._client is None:
            self._client = self._build_client(model=self.model)
        return self._client

    def _build_client(self, *, model: str) -> Any:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "langchain-openai is not installed. Run "
                "`python -m pip install -r requirements.txt` before using "
                "LangChainOpenAILLMClient."
            ) from exc

        return ChatOpenAI(
            api_key=self.api_key,
            model=model,
            **self.model_kwargs,
        )


def create_default_llm_client() -> LLMClient:
    """Create the configured default text generation client."""

    if os.getenv("LLM_PROVIDER") is None:
        load_dotenv()
    provider = (os.getenv("LLM_PROVIDER") or "langchain_openai").strip().lower()
    if provider in {"langchain_openai", "langchain-openai", "langchain"}:
        return LangChainOpenAILLMClient(load_env=False)
    raise ValueError(
        "Unsupported LLM_PROVIDER. Expected one of: "
        "langchain_openai."
    )


def _clean_env_secret(value: str | None) -> str | None:
    if value is None:
        return None
    return value.strip().strip('"').strip("'")


def _extract_langchain_response_text(
    response: Any,
    *,
    strip_whitespace: bool = True,
) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip() if strip_whitespace else content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))
            else:
                text = getattr(item, "text", None)
                if text:
                    parts.append(str(text))
        text = "\n".join(parts)
        return text.strip() if strip_whitespace else text
    text = str(content)
    return text.strip() if strip_whitespace else text
