from __future__ import annotations
from typing import Any


class FakeLLMClient:
    """
    A fake LLM client for testing purposes.
    This class simulates the behavior of a real LLM client, allowing for testing without relying on an actual LLM service.
    """

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """
        Generate a response based on the given prompt.

        Args:
            prompt (str): The input prompt for the LLM.
            **kwargs (Any): Additional keyword arguments that may be required by specific implementations.

        Returns:
            str: A simulated response from the fake LLM.
        """
        prompt_lower = prompt.lower()
        if "summarize" in prompt_lower:
            return "This is a fake summary of the papers."
        elif "rank" in prompt_lower:
            return "This is a fake ranking of the papers."
        elif "report" in prompt_lower:
            return "This is a fake report generated from the papers."
        else:
            return "This is a generic fake response from the LLM."
