"""
OpenAI API client implementation with rate limiting.

Real backend for querying GPT models. Includes a configurable minimum
interval between calls to stay within the account's TPM (tokens per
minute) limit, and handles parameter differences between standard models
(gpt-4o) and reasoning models (gpt-5*, o1, o3, o4).

Tier 0 (default for new accounts) with gpt-4o:
    TPM limit: 30,000
    Tokens per call: ~1,500
    -> ~20 calls/min safe -> min_seconds_between_calls = 3.0

Tier 1+ (after spending $5+):
    TPM limit: 500,000
    -> rate limiting not needed; set min_seconds_between_calls = 0

Request a tier upgrade at: platform.openai.com -> Settings -> Limits

Usage:
    from llm.openai_client import OpenAIClient
    client = OpenAIClient(model="gpt-4o", min_seconds_between_calls=3.0)
"""

from __future__ import annotations

import time

import openai

from llm.client import (
    LLMClient,
    LLMResponse,
    get_env_value,
    parse_action_from_response,
)


class OpenAIClient(LLMClient):
    """Real OpenAI API client using the official SDK, with rate limiting."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        max_tokens: int = 150,
        temperature: float = 1.0,
        min_seconds_between_calls: float = 3.0,
    ) -> None:
        """
        Args:
            model: which model to query (e.g. "gpt-4o", "gpt-5-mini")
            api_key: API key (defaults to OPENAI_API_KEY env var)
            max_tokens: max output tokens per call. 150 gives the model
                room for a brief rationale without truncation.
            temperature: sampling temperature (ignored for reasoning models)
            min_seconds_between_calls: minimum interval between API calls
                in seconds. Set to 3.0 for Tier 0 (30K TPM) accounts to
                stay under the rate limit. Set to 0 if you're on Tier 1+
                with much higher throughput.
        """
        resolved_api_key = api_key or get_env_value("OPENAI_API_KEY")
        if not resolved_api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set in ipd_project/.env and was "
                "not passed explicitly."
            )

        self._openai = openai
        self._client = openai.OpenAI(api_key=resolved_api_key)
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._min_interval = float(min_seconds_between_calls)
        self._last_call_time: float = 0.0

    @property
    def model_id(self) -> str:
        return self._model

    def _throttle(self) -> None:
        """Sleep just long enough to maintain the minimum call interval."""
        if self._min_interval <= 0:
            return
        elapsed = time.time() - self._last_call_time
        wait = self._min_interval - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_call_time = time.time()

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        self._throttle()

        # Reasoning models (gpt-5*, o1, o3, o4) use max_completion_tokens
        # instead of max_tokens, and don't accept custom temperature.
        is_reasoning_model = (
            self._model.startswith("gpt-5")
            or self._model.startswith("o1")
            or self._model.startswith("o3")
            or self._model.startswith("o4")
        )
        token_param = (
            "max_completion_tokens" if is_reasoning_model else "max_tokens"
        )

        kwargs = {
            "model": self._model,
            token_param: self._max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if not is_reasoning_model:
            kwargs["temperature"] = self._temperature

        response = self._client.chat.completions.create(**kwargs)
        text = response.choices[0].message.content or ""

        return LLMResponse(
            text=text,
            raw_action=parse_action_from_response(text),
            model_id=self._model,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )