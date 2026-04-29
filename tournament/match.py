"""
Match engine for iterated Prisoner's Dilemma.

A `Match` runs a single iterated game between two strategies, with either
fixed-length or shadow-of-the-future (probabilistic) termination. It returns
the full history of the game from each player's perspective.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from strategies.base import Strategy
from utils.game import Action, GameHistory, RoundResult


@dataclass
class MatchConfig:
    """Configuration for a single iterated match."""

    # Probability the game terminates after each round (0 = never, 1 = single round)
    # Set to 0 to disable shadow-of-the-future and use fixed length only.
    termination_prob: float = 0.10

    # Hard cap on rounds. Always enforced even with probabilistic termination,
    # to bound API costs and ensure simulations terminate.
    max_rounds: int = 100

    # Minimum rounds to play even if probabilistic termination triggers early.
    # 0 means termination can happen at any point.
    min_rounds: int = 1

    # Random seed for reproducibility of termination decisions.
    seed: int | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.termination_prob <= 1:
            raise ValueError(f"termination_prob must be in [0, 1], got {self.termination_prob}")
        if self.max_rounds < 1:
            raise ValueError(f"max_rounds must be >= 1, got {self.max_rounds}")
        if self.min_rounds < 0 or self.min_rounds > self.max_rounds:
            raise ValueError(f"min_rounds must be in [0, max_rounds={self.max_rounds}]")


@dataclass
class MatchResult:
    """Outcome of a single match."""

    strategy_a_name: str
    strategy_b_name: str
    history_a: GameHistory  # From strategy A's perspective
    rounds_played: int

    @property
    def history_b(self) -> GameHistory:
        return self.history_a.flip()

    @property
    def payoff_a(self) -> int:
        return self.history_a.my_total_payoff

    @property
    def payoff_b(self) -> int:
        return self.history_a.opp_total_payoff

    @property
    def avg_payoff_a(self) -> float:
        return self.payoff_a / self.rounds_played if self.rounds_played else 0.0

    @property
    def avg_payoff_b(self) -> float:
        return self.payoff_b / self.rounds_played if self.rounds_played else 0.0

    @property
    def coop_rate_a(self) -> float:
        if self.rounds_played == 0:
            return 0.0
        return sum(1 for a in self.history_a.my_actions if a == Action.COOPERATE) / self.rounds_played

    @property
    def coop_rate_b(self) -> float:
        if self.rounds_played == 0:
            return 0.0
        return sum(1 for a in self.history_a.opp_actions if a == Action.COOPERATE) / self.rounds_played


def play_match(
    strategy_a: Strategy,
    strategy_b: Strategy,
    config: MatchConfig | None = None,
) -> MatchResult:
    """
    Run a single iterated PD match between two strategies.

    Each strategy gets a fresh `reset()` call so stateful strategies don't
    carry state across matches. The game runs until either:
      - The hard `max_rounds` cap is reached, or
      - Probabilistic termination triggers (after `min_rounds`).
    """
    if config is None:
        config = MatchConfig()

    rng = random.Random(config.seed)

    strategy_a.reset()
    strategy_b.reset()

    history_a = GameHistory()  # From A's perspective

    rounds_played = 0
    while rounds_played < config.max_rounds:
        history_b = history_a.flip()  # From B's perspective

        action_a = strategy_a.decide(history_a)
        action_b = strategy_b.decide(history_b)

        history_a.append(RoundResult.from_actions(action_a, action_b))
        rounds_played += 1

        # Check probabilistic termination
        if rounds_played >= config.min_rounds and config.termination_prob > 0:
            if rng.random() < config.termination_prob:
                break

    return MatchResult(
        strategy_a_name=strategy_a.name,
        strategy_b_name=strategy_b.name,
        history_a=history_a,
        rounds_played=rounds_played,
    )
