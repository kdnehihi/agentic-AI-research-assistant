from __future__ import annotations

import os
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


class OpenAILLMClient:
    """
    OpenAI-backed LLM client.

    The client reads OPENAI_API_KEY from the environment by default. Keep this
    client opt-in so tests and local fake workflows never call the API by
    accident.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        load_env: bool = True,
    ) -> None:
        if load_env:
            load_dotenv()
        self.api_key = _clean_env_secret(api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model or os.getenv("OPENAI_MODEL") or "gpt-4.1-mini"
        self._client: Any | None = None

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """
        Generate text from an OpenAI model.

        Optional kwargs are passed through to responses.create, so callers can
        set model, max_output_tokens, temperature, or other supported options.
        """
        if not self.api_key:
            raise MissingOpenAIAPIKeyError(
                "OPENAI_API_KEY is not set. Add it to your environment before "
                "using OpenAILLMClient."
            )

        client = self._get_client()
        request_kwargs = {
            "model": kwargs.pop("model", self.model),
            "input": prompt,
        }
        request_kwargs.update(kwargs)

        response = client.responses.create(**request_kwargs)
        return _extract_response_text(response)

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key)
        return self._client


class GeminiLLMClient:
    """
    Gemini-backed LLM client.

    The client reads GEMINI_API_KEY from the environment by default. Use
    GEMINI_MODEL to override the default free-tier friendly model.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        load_env: bool = True,
    ) -> None:
        if load_env:
            load_dotenv()
        self.api_key = _clean_env_secret(api_key or os.getenv("GEMINI_API_KEY"))
        self.model = model or os.getenv("GEMINI_MODEL") or "gemini-2.5-flash"
        self._client: Any | None = None

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """
        Generate text from a Gemini model.

        Optional kwargs are passed through to models.generate_content.
        """
        if not self.api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Add it to your environment before "
                "using GeminiLLMClient."
            )

        client = self._get_client()
        request_kwargs = {
            "model": kwargs.pop("model", self.model),
            "contents": prompt,
        }
        request_kwargs.update(kwargs)

        response = client.models.generate_content(**request_kwargs)
        return _extract_gemini_response_text(response)

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from google import genai
            except ImportError as exc:
                raise RuntimeError(
                    "google-genai is not installed. Run "
                    "`python -m pip install -r requirements.txt` or "
                    "`python -m pip install google-genai` before using "
                    "GeminiLLMClient."
                ) from exc

            self._client = genai.Client(api_key=self.api_key)
        return self._client


def create_default_llm_client() -> LLMClient:
    """Create the configured default text generation client."""

    if os.getenv("LLM_PROVIDER") is None:
        load_dotenv()
    provider = (os.getenv("LLM_PROVIDER") or "langchain_openai").strip().lower()
    if provider in {"langchain_openai", "langchain-openai", "langchain"}:
        return LangChainOpenAILLMClient(load_env=False)
    if provider in {"openai", "openai_direct", "direct_openai"}:
        return OpenAILLMClient(load_env=False)
    if provider == "gemini":
        return GeminiLLMClient(load_env=False)
    raise ValueError(
        "Unsupported LLM_PROVIDER. Expected one of: "
        "langchain_openai, openai, gemini."
    )


def _clean_env_secret(value: str | None) -> str | None:
    if value is None:
        return None
    return value.strip().strip('"').strip("'")


def _extract_langchain_response_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
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
        return "\n".join(parts).strip()
    return str(content).strip()


def _extract_response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text.strip()

    text_parts: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                text_parts.append(text)

    return "\n".join(text_parts).strip()


def _extract_gemini_response_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if text:
        return text.strip()

    text_parts: list[str] = []
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", []) or []:
            part_text = getattr(part, "text", None)
            if part_text:
                text_parts.append(part_text)

    return "\n".join(text_parts).strip()
