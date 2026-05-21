"""
Adapters between our internal Strategy interface and axelrod-python's Player.

This lets us delegate the classical strategy implementations to axelrod
(canonical, peer-reviewed) while keeping our LLM strategies in our own
codebase. Our strategies are converted into axelrod.Player subclasses
on the fly, so they can be used in any axelrod tournament.

Updated to include 3 additional defection-capable strategies that produce
more strategic diversity than the original 6:
  - Joss (FirstByJoss): TFT but defects ~10% randomly
  - SuspiciousTFT: like TFT but defects on the first move
  - Tester (SecondByTester): probes opponent for exploitability

These were added because in the original smoke test only the AlwaysDefect
column of the payoff matrix exhibited any variation — every other strategy
fully cooperated with every non-AD opponent. Adding defection-capable
strategies creates meaningful evolutionary pressure across more columns.
"""

from __future__ import annotations

from typing import Type

import axelrod as axl

from strategies.base import Strategy
from utils.game import Action, GameHistory, RoundResult


# Map between our Action and axelrod's Action
_OUR_TO_AXL = {Action.COOPERATE: axl.Action.C, Action.DEFECT: axl.Action.D}
_AXL_TO_OUR = {axl.Action.C: Action.COOPERATE, axl.Action.D: Action.DEFECT}


def to_axelrod_action(action: Action) -> axl.Action:
    return _OUR_TO_AXL[action]


def from_axelrod_action(action: axl.Action) -> Action:
    return _AXL_TO_OUR[action]


def build_game_history_from_axl(
    self_history: list[axl.Action],
    opponent_history: list[axl.Action],
) -> GameHistory:
    """
    Build our internal GameHistory from axelrod-style histories.

    Axelrod records moves but not per-round payoffs; we recompute them.
    """
    from utils.game import get_payoff

    rounds = []
    for my_axl, opp_axl in zip(self_history, opponent_history):
        my = from_axelrod_action(my_axl)
        opp = from_axelrod_action(opp_axl)
        my_p, opp_p = get_payoff(my, opp)
        rounds.append(RoundResult(my, opp, my_p, opp_p))
    return GameHistory(rounds)


def wrap_as_axelrod_player(strategy: Strategy) -> axl.Player:
    """Wrap one of our `Strategy` instances as an axelrod `Player`."""
    return _make_axl_player(strategy)


def _make_axl_player(strategy: Strategy) -> axl.Player:
    """Create an axelrod.Player that delegates to our Strategy."""
    captured_strategy = strategy
    captured_name = strategy.name

    class WrappedPlayer(axl.Player):
        classifier = {
            "memory_depth": float("inf"),
            "stochastic": True,  # Conservative — we don't know about LLMs
            "long_run_time": True,  # LLM calls are slow
            "inspects_source": False,
            "manipulates_source": False,
            "manipulates_state": False,
            "makes_use_of": set(),
        }

        # Note: axelrod introspects __init__ via `inspect.signature` and
        # requires the first parameter to literally be named `self`.
        def __init__(self) -> None:
            super().__init__()
            self._strategy = captured_strategy
            self.name = captured_name

        def strategy(self, opponent: axl.Player) -> axl.Action:
            history = build_game_history_from_axl(
                list(self.history), list(opponent.history)
            )
            our_action = self._strategy.decide(history)
            return to_axelrod_action(our_action)

        def reset(self) -> None:
            super().reset()
            self._strategy.reset()

    WrappedPlayer.name = captured_name
    return WrappedPlayer()


def make_axelrod_player_factory(strategy_factory):
    """Wrap a strategy factory as a callable returning fresh axelrod Players."""

    def factory():
        strategy = strategy_factory()
        return _make_axl_player(strategy)

    return factory


# Expanded set of classical strategies for the tournament.
# Six original strategies + three additional defection-capable variants
# that create meaningful strategic differentiation across the tournament.
AXELROD_CLASSICAL_FACTORIES: dict[str, callable] = {
    # --- Original 6 ---
    "TitForTat": axl.TitForTat,
    "AlwaysDefect": axl.Defector,
    "AlwaysCooperate": axl.Cooperator,
    "Pavlov": axl.WinStayLoseShift,
    "GrimTrigger": axl.Grudger,  # axelrod calls Grim Trigger 'Grudger'
    "GenerousTFT": lambda: axl.GTFT(),  # axelrod calls Generous TFT 'GTFT'
    # --- New 3: defection-capable variants for strategic diversity ---
    "Joss": axl.FirstByJoss,              # TFT but defects ~10% randomly
    "SuspiciousTFT": axl.SuspiciousTitForTat,  # TFT but defects first move
    "Tester": axl.SecondByTester,         # Probes opponent for exploitability
}