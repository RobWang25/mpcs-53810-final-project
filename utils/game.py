"""
Core game definitions for Iterated Prisoner's Dilemma.

Standard payoff convention (T > R > P > S, 2R > T + S):
    T = 5  Temptation (you defect, opponent cooperates)
    R = 3  Reward     (mutual cooperation)
    P = 1  Punishment (mutual defection)
    S = 0  Sucker     (you cooperate, opponent defects)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class Action(Enum):
    """The two available actions in Prisoner's Dilemma."""

    COOPERATE = "C"
    DEFECT = "D"

    def __str__(self) -> str:
        return self.value


# Payoff constants
TEMPTATION = 5
REWARD = 3
PUNISHMENT = 1
SUCKER = 0

# Payoff matrix: PAYOFFS[my_action][opp_action] = (my_payoff, opp_payoff)
PAYOFFS: dict[Tuple[Action, Action], Tuple[int, int]] = {
    (Action.COOPERATE, Action.COOPERATE): (REWARD, REWARD),
    (Action.COOPERATE, Action.DEFECT): (SUCKER, TEMPTATION),
    (Action.DEFECT, Action.COOPERATE): (TEMPTATION, SUCKER),
    (Action.DEFECT, Action.DEFECT): (PUNISHMENT, PUNISHMENT),
}


def get_payoff(my_action: Action, opp_action: Action) -> Tuple[int, int]:
    """Return (my_payoff, opp_payoff) for a single round."""
    return PAYOFFS[(my_action, opp_action)]


@dataclass(frozen=True)
class RoundResult:
    """Outcome of a single PD round."""

    my_action: Action
    opp_action: Action
    my_payoff: int
    opp_payoff: int

    @classmethod
    def from_actions(cls, my_action: Action, opp_action: Action) -> RoundResult:
        my_p, opp_p = get_payoff(my_action, opp_action)
        return cls(my_action, opp_action, my_p, opp_p)


@dataclass
class GameHistory:
    """
    Sequential history of a game from one player's perspective.

    `rounds[i]` is the i-th round result. `my_actions` and `opp_actions`
    are convenience accessors.
    """

    rounds: list[RoundResult]

    def __init__(self, rounds: list[RoundResult] | None = None) -> None:
        self.rounds = rounds if rounds is not None else []

    def __len__(self) -> int:
        return len(self.rounds)

    def append(self, result: RoundResult) -> None:
        self.rounds.append(result)

    @property
    def my_actions(self) -> list[Action]:
        return [r.my_action for r in self.rounds]

    @property
    def opp_actions(self) -> list[Action]:
        return [r.opp_action for r in self.rounds]

    @property
    def my_total_payoff(self) -> int:
        return sum(r.my_payoff for r in self.rounds)

    @property
    def opp_total_payoff(self) -> int:
        return sum(r.opp_payoff for r in self.rounds)

    def flip(self) -> GameHistory:
        """Return the same history from the opponent's perspective."""
        flipped = [
            RoundResult(r.opp_action, r.my_action, r.opp_payoff, r.my_payoff)
            for r in self.rounds
        ]
        return GameHistory(flipped)

    def last_round(self) -> RoundResult | None:
        return self.rounds[-1] if self.rounds else None
