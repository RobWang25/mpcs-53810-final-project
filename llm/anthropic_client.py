"""
Anthropic Claude API client implementation.

This is the real backend used to query Claude. It's only imported when
actually running tournaments with live LLMs — the rest of the codebase
doesn't depend on the anthropic SDK being installed.

Usage:
    from llm.anthropic_client import AnthropicClient
    client = AnthropicClient(model="claude-sonnet-4-6", api_key=...)
"""

from __future__ import annotations

import anthropic
from typing import Any

from llm.client import (
    LLMClient,
    LLMResponse,
    get_env_value,
    parse_action_from_response,
)


class AnthropicClient(LLMClient):
    """Real Anthropic API client using the official SDK."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: str | None = None,
        max_tokens: int = 50,
        temperature: float = 1.0,
    ) -> None:
        resolved_api_key = api_key or get_env_value("ANTHROPIC_API_KEY")
        if not resolved_api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set in ipd_project/.env and was not passed explicitly."
            )

        self._anthropic = anthropic
        self._client = anthropic.Anthropic(api_key=resolved_api_key)
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature

    @property
    def model_id(self) -> str:
        return self._model

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        message = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        # Extract text content
        text = ""
        for block in message.content:
            if hasattr(block, "text"):
                text += block.text

        return LLMResponse(
            text=text,
            raw_action=parse_action_from_response(text),
            model_id=self._model,
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
        )
