"""
Behavioral fingerprint metrics.

This module operationalizes the qualitative "strategic fingerprint" notion
from Payne & Alloui-Cros (2025) into concrete, measurable quantities.

A fingerprint is computed from a collection of game histories. It can be
computed for a single match, aggregated across matches against a given
opponent, or aggregated across an entire tournament.

Metrics:
  - cooperation_rate: overall fraction of moves that were COOPERATE
  - first_move_coop: fraction of games where the first move was COOPERATE
  - retaliation_rate: P(D | opponent played D last round)
  - forgiveness_rate: P(C | I played D last round AND opponent played D last round)
  - opportunism_rate: P(D | opponent played C last round)
  - good_partner_rate: P(C | opponent played C last round) — distinct from
    retaliation_rate; measures pure niceness against cooperative opponents
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

from utils.game import Action, GameHistory


@dataclass(frozen=True)
class Fingerprint:
    """
    Behavioral fingerprint — a quantitative strategic profile.

    All metrics are floats in [0, 1] except where indicated. NaN-style
    sentinels are used (-1.0) when the relevant denominator was zero
    (e.g. forgiveness_rate is undefined if the strategy never retaliates).
    """

    # Sample size info
    games_played: int
    total_rounds: int

    # Core metrics
    cooperation_rate: float
    first_move_coop: float
    retaliation_rate: float  # -1.0 if never had opportunity
    forgiveness_rate: float  # -1.0 if never had opportunity
    opportunism_rate: float  # -1.0 if never had opportunity
    good_partner_rate: float  # -1.0 if never had opportunity

    # Bookkeeping for downstream analysis
    n_first_moves: int
    n_retaliation_opportunities: int
    n_forgiveness_opportunities: int
    n_opportunism_opportunities: int
    n_good_partner_opportunities: int

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        def fmt(v: float) -> str:
            return "n/a" if v < 0 else f"{v:.3f}"

        return (
            f"Fingerprint(games={self.games_played}, rounds={self.total_rounds}, "
            f"coop={self.cooperation_rate:.3f}, "
            f"first_move_coop={fmt(self.first_move_coop)}, "
            f"retaliation={fmt(self.retaliation_rate)}, "
            f"forgiveness={fmt(self.forgiveness_rate)}, "
            f"opportunism={fmt(self.opportunism_rate)}, "
            f"good_partner={fmt(self.good_partner_rate)})"
        )


def _safe_rate(numerator: int, denominator: int) -> float:
    """Return numerator/denominator, or -1.0 if denominator is zero."""
    return numerator / denominator if denominator > 0 else -1.0


def compute_fingerprint(histories: Iterable[GameHistory]) -> Fingerprint:
    """
    Compute a behavioral fingerprint from a collection of game histories.

    Each `GameHistory` should be from the perspective of the strategy being
    profiled (i.e. `my_actions` are this strategy's moves).
    """
    histories = list(histories)

    total_rounds = 0
    coop_moves = 0

    first_moves_coop = 0
    n_first_moves = 0

    retaliation_count = 0
    n_retaliation_opps = 0

    forgiveness_count = 0
    n_forgiveness_opps = 0

    opportunism_count = 0
    n_opportunism_opps = 0

    good_partner_count = 0
    n_good_partner_opps = 0

    for hist in histories:
        if not hist.rounds:
            continue

        total_rounds += len(hist)
        coop_moves += sum(1 for a in hist.my_actions if a == Action.COOPERATE)

        # First move
        n_first_moves += 1
        if hist.my_actions[0] == Action.COOPERATE:
            first_moves_coop += 1

        # Look at consecutive (prev_round, this_round) pairs
        for i in range(1, len(hist)):
            prev = hist.rounds[i - 1]
            curr = hist.rounds[i]

            opp_d_last = prev.opp_action == Action.DEFECT
            opp_c_last = prev.opp_action == Action.COOPERATE
            my_d_last = prev.my_action == Action.DEFECT

            # Retaliation: I defect now given opponent defected last round
            if opp_d_last:
                n_retaliation_opps += 1
                if curr.my_action == Action.DEFECT:
                    retaliation_count += 1

            # Forgiveness: I cooperate now given mutual defection last round
            #   (i.e., I previously retaliated and now move on)
            if opp_d_last and my_d_last:
                n_forgiveness_opps += 1
                if curr.my_action == Action.COOPERATE:
                    forgiveness_count += 1

            # Opportunism: I defect now given opponent cooperated last round
            if opp_c_last:
                n_opportunism_opps += 1
                if curr.my_action == Action.DEFECT:
                    opportunism_count += 1

            # Good partner: I cooperate now given opponent cooperated last round
            if opp_c_last:
                n_good_partner_opps += 1
                if curr.my_action == Action.COOPERATE:
                    good_partner_count += 1

    return Fingerprint(
        games_played=len([h for h in histories if h.rounds]),
        total_rounds=total_rounds,
        cooperation_rate=_safe_rate(coop_moves, total_rounds),
        first_move_coop=_safe_rate(first_moves_coop, n_first_moves),
        retaliation_rate=_safe_rate(retaliation_count, n_retaliation_opps),
        forgiveness_rate=_safe_rate(forgiveness_count, n_forgiveness_opps),
        opportunism_rate=_safe_rate(opportunism_count, n_opportunism_opps),
        good_partner_rate=_safe_rate(good_partner_count, n_good_partner_opps),
        n_first_moves=n_first_moves,
        n_retaliation_opportunities=n_retaliation_opps,
        n_forgiveness_opportunities=n_forgiveness_opps,
        n_opportunism_opportunities=n_opportunism_opps,
        n_good_partner_opportunities=n_good_partner_opps,
    )
