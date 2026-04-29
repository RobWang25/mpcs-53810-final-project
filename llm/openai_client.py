"""
OpenAI API client implementation.

Real backend for querying GPT models. Only imported when running with
live OpenAI calls.

Usage:
    from llm.openai_client import OpenAIClient
    client = OpenAIClient(model="gpt-4o", api_key=...)
"""

from __future__ import annotations

import openai

from llm.client import (
    LLMClient,
    LLMResponse,
    get_env_value,
    parse_action_from_response,
)


class OpenAIClient(LLMClient):
    """Real OpenAI API client using the official SDK."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        max_tokens: int = 50,
        temperature: float = 1.0,
    ) -> None:
        resolved_api_key = api_key or get_env_value("OPENAI_API_KEY")
        if not resolved_api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set in ipd_project/.env and was not passed explicitly."
            )

        self._openai = openai
        self._client = openai.OpenAI(api_key=resolved_api_key)
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature

    @property
    def model_id(self) -> str:
        return self._model

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        text = response.choices[0].message.content or ""

        return LLMResponse(
            text=text,
            raw_action=parse_action_from_response(text),
            model_id=self._model,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )
