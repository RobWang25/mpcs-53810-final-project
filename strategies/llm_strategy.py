"""
LLM-backed strategy.

`LLMStrategy` wraps an `LLMClient` and a `FramingCondition` together as a
plug-in replacement for any classical `Strategy`. It can be dropped into
the same match engine and tournament code with no changes.

A fallback action is used when the LLM response cannot be parsed (e.g. the
model refuses, returns garbage, or the call fails). Fallback defaults to
COOPERATE — chosen because it's the more conservative choice that avoids
poisoning a cooperative match due to a single parse failure.
"""

from __future__ import annotations

import logging

from llm.client import LLMClient, parse_action_from_response
from llm.prompts import FramingCondition, build_user_prompt, get_template
from strategies.base import Strategy
from utils.game import Action, GameHistory

logger = logging.getLogger(__name__)


class LLMStrategy(Strategy):
    """
    A strategy backed by an LLM client + a specific prompt framing.

    Each call to `decide()` is one API call. To bound API costs, it's the
    caller's responsibility to limit match length via MatchConfig.max_rounds.
    """

    def __init__(
        self,
        client: LLMClient,
        framing: FramingCondition,
        fallback_action: Action = Action.COOPERATE,
        custom_name: str | None = None,
    ) -> None:
        self.client = client
        self.framing = framing
        self.template = get_template(framing)
        self.fallback_action = fallback_action

        # Tracking
        self.parse_failures = 0
        self.total_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

        # Reasoning logs — useful for qualitative analysis later
        self.reasoning_logs: list[str] = []

        if custom_name is not None:
            self.name = custom_name
        else:
            self.name = f"{client.model_id}:{framing.value}"

    def decide(self, history: GameHistory) -> Action:
        user_prompt = build_user_prompt(history)
        self.total_calls += 1

        try:
            response = self.client.complete(self.template.system_prompt, user_prompt)
        except Exception as e:
            logger.warning(f"LLM call failed: {e}. Using fallback action.")
            self.parse_failures += 1
            return self.fallback_action

        self.total_input_tokens += response.input_tokens
        self.total_output_tokens += response.output_tokens
        self.reasoning_logs.append(response.text)

        parsed = parse_action_from_response(response.text)
        if parsed == "COOPERATE":
            return Action.COOPERATE
        if parsed == "DEFECT":
            return Action.DEFECT

        logger.warning(
            f"Could not parse action from response: {response.text!r}. Using fallback."
        )
        self.parse_failures += 1
        return self.fallback_action

    def reset(self) -> None:
        """Clear per-game logs but keep cumulative call statistics."""
        self.reasoning_logs = []

    def stats(self) -> dict:
        """Summary statistics for this strategy across all calls so far."""
        return {
            "total_calls": self.total_calls,
            "parse_failures": self.parse_failures,
            "parse_failure_rate": (
                self.parse_failures / self.total_calls if self.total_calls else 0.0
            ),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
        }
