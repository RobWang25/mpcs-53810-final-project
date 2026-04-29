"""
Classical Prisoner's Dilemma strategies.

These are the canonical baseline strategies from the evolutionary game theory
literature. They are used both as opponents for LLM strategies and as
benchmark contestants in the evolutionary tournament.
"""

from __future__ import annotations

import random

from strategies.base import Strategy
from utils.game import Action, GameHistory


class TitForTat(Strategy):
    """Cooperate first, then mirror opponent's last move."""

    name = "TitForTat"

    def decide(self, history: GameHistory) -> Action:
        if not history.rounds:
            return Action.COOPERATE
        return history.opp_actions[-1]


class AlwaysDefect(Strategy):
    """Always defect, regardless of history."""

    name = "AlwaysDefect"

    def decide(self, history: GameHistory) -> Action:
        return Action.DEFECT


class AlwaysCooperate(Strategy):
    """Always cooperate, regardless of history."""

    name = "AlwaysCooperate"

    def decide(self, history: GameHistory) -> Action:
        return Action.COOPERATE


class Pavlov(Strategy):
    """
    Win-Stay-Lose-Shift.

    Cooperate first. After that, repeat the previous action if it earned
    R or T (a "win"), and switch otherwise.
    """

    name = "Pavlov"

    def decide(self, history: GameHistory) -> Action:
        if not history.rounds:
            return Action.COOPERATE
        last = history.rounds[-1]
        # Won the round if payoff was R (3) or T (5)
        won = last.my_payoff >= 3
        if won:
            return last.my_action
        # Lost — switch
        return Action.DEFECT if last.my_action == Action.COOPERATE else Action.COOPERATE


class GrimTrigger(Strategy):
    """
    Cooperate until the opponent defects once; then defect forever.
    """

    name = "GrimTrigger"

    def __init__(self) -> None:
        self._triggered = False

    def reset(self) -> None:
        self._triggered = False

    def decide(self, history: GameHistory) -> Action:
        if self._triggered:
            return Action.DEFECT
        if history.opp_actions and Action.DEFECT in history.opp_actions:
            self._triggered = True
            return Action.DEFECT
        return Action.COOPERATE


class GenerousTitForTat(Strategy):
    """
    Tit-for-Tat with forgiveness. Mirrors opponent's last move, but
    cooperates with probability `forgiveness` even after opponent defection.

    A standard value of forgiveness = 0.1 is used by default. This makes
    the strategy more robust to noise and to chains of mutual retaliation.
    """

    name = "GenerousTitForTat"

    def __init__(self, forgiveness: float = 0.1, rng: random.Random | None = None) -> None:
        self.forgiveness = forgiveness
        self._rng = rng if rng is not None else random.Random()

    def decide(self, history: GameHistory) -> Action:
        if not history.rounds:
            return Action.COOPERATE
        last_opp = history.opp_actions[-1]
        if last_opp == Action.COOPERATE:
            return Action.COOPERATE
        # Opponent defected — forgive with small probability
        if self._rng.random() < self.forgiveness:
            return Action.COOPERATE
        return Action.DEFECT


# Convenient registry of classical strategies
CLASSICAL_STRATEGIES: dict[str, type[Strategy]] = {
    "TitForTat": TitForTat,
    "AlwaysDefect": AlwaysDefect,
    "AlwaysCooperate": AlwaysCooperate,
    "Pavlov": Pavlov,
    "GrimTrigger": GrimTrigger,
    "GenerousTitForTat": GenerousTitForTat,
}


def make_classical_strategy(name: str) -> Strategy:
    """Factory for instantiating a classical strategy by name."""
    if name not in CLASSICAL_STRATEGIES:
        raise ValueError(
            f"Unknown classical strategy: {name}. "
            f"Available: {sorted(CLASSICAL_STRATEGIES)}"
        )
    return CLASSICAL_STRATEGIES[name]()
