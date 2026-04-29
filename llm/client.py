"""
LLM client abstraction.

We define a simple `LLMClient` protocol that all model backends implement.
This lets the rest of the code stay agnostic about which provider is being
used. We also provide a `MockLLMClient` for cost-free testing — it simulates
LLM responses with configurable cooperation probabilities.

Real backends (Anthropic, OpenAI) live in separate modules and are only
imported when needed.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LLMResponse:
    """A single response from an LLM."""

    text: str  # Full response text
    raw_action: str  # Just the parsed action token (e.g. "COOPERATE", "DEFECT", or "?")
    model_id: str
    input_tokens: int = 0
    output_tokens: int = 0


class LLMClient(ABC):
    """Abstract LLM backend."""

    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Submit a single completion request and return the response."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Unique identifier for this model + backend combination."""


class MockLLMClient(LLMClient):
    """
    Mock LLM that returns COOPERATE or DEFECT according to a configurable
    cooperation probability. Useful for testing the pipeline end-to-end
    without spending API credits.

    Optionally, the mock can imitate a known classical strategy (e.g. TFT)
    so we can verify the LLM strategy wrapper integrates correctly.
    """

    def __init__(
        self,
        coop_probability: float = 0.7,
        model_id: str = "mock-llm",
        seed: int | None = None,
    ) -> None:
        self._coop_prob = coop_probability
        self._model_id = model_id
        self._rng = random.Random(seed)

    @property
    def model_id(self) -> str:
        return self._model_id

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        action = "COOPERATE" if self._rng.random() < self._coop_prob else "DEFECT"
        return LLMResponse(
            text=action,
            raw_action=action,
            model_id=self._model_id,
            input_tokens=len(system_prompt.split()) + len(user_prompt.split()),
            output_tokens=1,
        )


def get_env_value(name: str) -> str | None:
    """Read a single value directly from the project's .env file."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.is_file():
        return None

    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        if key.strip() != name:
            continue

        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        return value

    return None


def parse_action_from_response(text: str) -> str:
    """
    Robustly parse a COOPERATE/DEFECT decision from an LLM response.

    Returns "COOPERATE", "DEFECT", or "?" if neither can be reliably extracted.
    Strategy:
      1. Look for an exact match of either token (case-insensitive)
      2. If both appear, take the LAST one mentioned (the model's final answer)
      3. If neither appears, return "?"
    """
    upper = text.upper()
    # Find last occurrences
    coop_idx = upper.rfind("COOPERATE")
    def_idx = upper.rfind("DEFECT")

    if coop_idx == -1 and def_idx == -1:
        return "?"
    if coop_idx == -1:
        return "DEFECT"
    if def_idx == -1:
        return "COOPERATE"
    # Both appear — take the later one (likely the model's conclusion)
    return "COOPERATE" if coop_idx > def_idx else "DEFECT"
