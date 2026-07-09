from __future__ import annotations
from typing import Protocol, Any


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
