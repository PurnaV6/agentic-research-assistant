"""Thin wrapper around the Groq chat-completions API.

Kept as a small interface (not the raw SDK client) so tests can swap in a
fake implementation without touching the orchestrator logic or needing a
real API key.
"""
from __future__ import annotations

from typing import Any, Protocol

from groq import Groq

from agent.config import GROQ_API_KEY, MODEL_NAME


class LLMClient(Protocol):
    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> Any:
        """Return the assistant message object (with .content and .tool_calls)."""


class GroqLLMClient:
    def __init__(self, api_key: str = GROQ_API_KEY, model: str = MODEL_NAME):
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Get a free key at https://console.groq.com "
                "and set it in a .env file or environment variable."
            )
        self._client = Groq(api_key=api_key)
        self._model = model

    def chat(self, messages: list[dict], tools: list[dict] | None = None):
        kwargs: dict = {"model": self._model, "messages": messages, "temperature": 0.2}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        response = self._client.chat.completions.create(**kwargs)
        return response.choices[0].message
