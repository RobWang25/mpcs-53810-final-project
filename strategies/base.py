"""
Base strategy interface.

Every strategy — classical or LLM-based — implements the same simple contract:
given the history of the game so far (from this strategy's perspective),
return the next action.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from utils.game import Action, GameHistory


class Strategy(ABC):
    """Abstract base class for all PD strategies."""

    name: str

    @abstractmethod
    def decide(self, history: GameHistory) -> Action:
        """
        Return the next action given the game history so far.

        `history` is from this strategy's perspective: history.my_actions
        are this strategy's past moves, history.opp_actions are the
        opponent's past moves.
        """

    def reset(self) -> None:
        """
        Reset any internal state between games.

        Default no-op. Stateful strategies (e.g. Grim Trigger) override this.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
