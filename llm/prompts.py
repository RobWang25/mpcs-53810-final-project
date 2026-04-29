"""
Prompt templates for the three framing conditions.

The system prompt defines the persona/context. The user prompt presents the
game state at a single decision point. We separate these so the system prompt
can be cached across many calls within a single match.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from utils.game import Action, GameHistory


class FramingCondition(Enum):
    """The three prompt framing conditions for our experiment."""

    NEUTRAL = "neutral"
    RATIONAL = "rational"
    COOPERATIVE = "cooperative"


@dataclass(frozen=True)
class PromptTemplate:
    """A prompt template for a single framing condition."""

    framing: FramingCondition
    system_prompt: str


# Common game description shared across all framings
_GAME_DESCRIPTION = """\
You are playing the iterated Prisoner's Dilemma. In each round, you and your \
opponent simultaneously choose to either COOPERATE or DEFECT. Payoffs per \
round are:

  - Both cooperate: you get 3, opponent gets 3
  - You cooperate, opponent defects: you get 0, opponent gets 5
  - You defect, opponent cooperates: you get 5, opponent gets 0
  - Both defect: you get 1, opponent gets 1

The game continues for an uncertain number of rounds — after each round there \
is a fixed probability the game ends. Your goal is to maximize your total \
accumulated payoff across all rounds played.

You will be shown the history of all previous rounds before each decision. \
Respond with exactly one word: either COOPERATE or DEFECT."""


_NEUTRAL_SYSTEM = _GAME_DESCRIPTION

_RATIONAL_SYSTEM = (
    "You are a rational economic agent. You make decisions to maximize your "
    "expected payoff using game-theoretic reasoning.\n\n"
    + _GAME_DESCRIPTION
)

_COOPERATIVE_SYSTEM = (
    "You are a cooperative partner. You value building mutual trust and "
    "achieving good outcomes for both parties.\n\n"
    + _GAME_DESCRIPTION
)


PROMPT_TEMPLATES: dict[FramingCondition, PromptTemplate] = {
    FramingCondition.NEUTRAL: PromptTemplate(
        framing=FramingCondition.NEUTRAL,
        system_prompt=_NEUTRAL_SYSTEM,
    ),
    FramingCondition.RATIONAL: PromptTemplate(
        framing=FramingCondition.RATIONAL,
        system_prompt=_RATIONAL_SYSTEM,
    ),
    FramingCondition.COOPERATIVE: PromptTemplate(
        framing=FramingCondition.COOPERATIVE,
        system_prompt=_COOPERATIVE_SYSTEM,
    ),
}


def format_history(history: GameHistory) -> str:
    """Format the game history as a human-readable string for the prompt."""
    if not history.rounds:
        return "No rounds have been played yet. This is round 1."

    lines = [f"History of {len(history)} rounds played so far:"]
    for i, r in enumerate(history.rounds, start=1):
        my = "COOPERATE" if r.my_action == Action.COOPERATE else "DEFECT"
        opp = "COOPERATE" if r.opp_action == Action.COOPERATE else "DEFECT"
        lines.append(
            f"  Round {i}: You played {my}, opponent played {opp}. "
            f"You got {r.my_payoff}, opponent got {r.opp_payoff}."
        )

    my_total = history.my_total_payoff
    opp_total = history.opp_total_payoff
    lines.append(f"\nYour total so far: {my_total}. Opponent's total: {opp_total}.")
    lines.append(f"\nYou are about to play round {len(history) + 1}.")
    return "\n".join(lines)


def build_user_prompt(history: GameHistory) -> str:
    """Build the user-side prompt for a single decision."""
    history_str = format_history(history)
    return (
        f"{history_str}\n\n"
        "What is your action? Respond with exactly one word: "
        "either COOPERATE or DEFECT."
    )


def get_template(framing: FramingCondition) -> PromptTemplate:
    """Look up the template for a given framing."""
    return PROMPT_TEMPLATES[framing]
