"""
Sanity tests for classical strategies and the match engine.

Run with: python -m tests.test_classical
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make project root importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strategies.classical import (
    AlwaysCooperate,
    AlwaysDefect,
    GenerousTitForTat,
    GrimTrigger,
    Pavlov,
    TitForTat,
)
from tournament.match import MatchConfig, play_match
from utils.game import Action, GameHistory, RoundResult


def test_always_defect_vs_always_cooperate() -> None:
    """AD vs AC: defector gets T=5 every round, cooperator gets S=0."""
    config = MatchConfig(termination_prob=0.0, max_rounds=10)
    result = play_match(AlwaysDefect(), AlwaysCooperate(), config)

    assert result.rounds_played == 10
    assert result.payoff_a == 50  # 10 rounds * T=5
    assert result.payoff_b == 0   # 10 rounds * S=0
    assert result.coop_rate_a == 0.0
    assert result.coop_rate_b == 1.0
    print(f"AD vs AC: A={result.payoff_a}, B={result.payoff_b}")


def test_tft_vs_tft_full_cooperation() -> None:
    """TFT vs TFT should fully cooperate — both get R=3 every round."""
    config = MatchConfig(termination_prob=0.0, max_rounds=20)
    result = play_match(TitForTat(), TitForTat(), config)

    assert result.payoff_a == 60  # 20 * R=3
    assert result.payoff_b == 60
    assert result.coop_rate_a == 1.0
    assert result.coop_rate_b == 1.0
    print(f"TFT vs TFT: full cooperation, both got {result.payoff_a}")


def test_tft_vs_always_defect() -> None:
    """TFT vs AD: TFT cooperates round 1, then defects forever."""
    config = MatchConfig(termination_prob=0.0, max_rounds=10)
    result = play_match(TitForTat(), AlwaysDefect(), config)

    # Round 1: TFT plays C, AD plays D -> TFT gets S=0, AD gets T=5
    # Rounds 2-10: TFT plays D, AD plays D -> both get P=1
    expected_tft = 0 + 9 * 1
    expected_ad = 5 + 9 * 1
    assert result.payoff_a == expected_tft, f"TFT got {result.payoff_a}, expected {expected_tft}"
    assert result.payoff_b == expected_ad, f"AD got {result.payoff_b}, expected {expected_ad}"
    print(f"TFT vs AD: TFT={result.payoff_a}, AD={result.payoff_b}")


def test_pavlov_recovers_from_mutual_defection() -> None:
    """Pavlov should switch to C after mutual defection (a 'lose')."""
    config = MatchConfig(termination_prob=0.0, max_rounds=4)
    result = play_match(Pavlov(), AlwaysDefect(), config)

    actions = result.history_a.my_actions
    # Round 1: Pavlov C (default first move). AD D. Pavlov got S=0 -> lose -> switch to D.
    # Round 2: Pavlov D, AD D. Pavlov got P=1 -> lose -> switch to C.
    # Round 3: Pavlov C, AD D. Pavlov got S=0 -> lose -> switch to D.
    # Round 4: Pavlov D, AD D. Pavlov got P=1 -> lose -> switch to C.
    expected = [Action.COOPERATE, Action.DEFECT, Action.COOPERATE, Action.DEFECT]
    assert actions == expected, f"Pavlov played {actions}, expected {expected}"
    print(f"Pavlov vs AD: oscillates as expected: {[str(a) for a in actions]}")


def test_grim_trigger_never_forgives() -> None:
    """Once the opponent defects once, GT defects forever."""
    config = MatchConfig(termination_prob=0.0, max_rounds=10)

    # Build a custom opponent that defects exactly on round 3, otherwise C
    from strategies.base import Strategy

    class DefectsOnRound3(Strategy):
        name = "DefectsOnRound3"

        def decide(self, history: GameHistory) -> Action:
            if len(history) == 2:  # About to play round 3
                return Action.DEFECT
            return Action.COOPERATE

    result = play_match(GrimTrigger(), DefectsOnRound3(), config)
    actions = result.history_a.my_actions

    # Rounds 1-3: GT plays C (no opponent defection in history yet at decision time)
    # After round 3, opponent has defected -> GT defects rounds 4-10
    expected = [Action.COOPERATE] * 3 + [Action.DEFECT] * 7
    assert actions == expected, f"GT played {actions}, expected {expected}"
    print(f"GrimTrigger never forgives after a defection")


def test_generous_tft_forgiveness() -> None:
    """With forgiveness=1.0, GTFT becomes Always Cooperate after opp defection."""
    import random as random_module

    config = MatchConfig(termination_prob=0.0, max_rounds=10)
    gtft = GenerousTitForTat(forgiveness=1.0, rng=random_module.Random(42))
    result = play_match(gtft, AlwaysDefect(), config)

    # GTFT with forgiveness=1.0 should cooperate every round despite opp defecting
    assert all(a == Action.COOPERATE for a in result.history_a.my_actions)
    print(f"GenerousTFT(forgiveness=1.0) cooperates always")

    # With forgiveness=0.0, it should be plain TFT
    gtft_strict = GenerousTitForTat(forgiveness=0.0, rng=random_module.Random(42))
    result_strict = play_match(gtft_strict, AlwaysDefect(), config)
    expected = [Action.COOPERATE] + [Action.DEFECT] * 9
    assert result_strict.history_a.my_actions == expected
    print(f"GenerousTFT(forgiveness=0.0) reduces to plain TFT")


def test_probabilistic_termination() -> None:
    """With high termination_prob, games should be short on average."""
    config = MatchConfig(termination_prob=0.5, max_rounds=100, min_rounds=1, seed=42)
    lengths = []
    for trial_seed in range(100):
        config_trial = MatchConfig(
            termination_prob=0.5, max_rounds=100, min_rounds=1, seed=trial_seed
        )
        result = play_match(TitForTat(), TitForTat(), config_trial)
        lengths.append(result.rounds_played)

    avg_length = sum(lengths) / len(lengths)
    # Expected length with p=0.5 is 1/p = 2
    assert 1.5 < avg_length < 3.5, f"Avg length {avg_length} far from expected ~2"
    print(f"Probabilistic termination: avg length {avg_length:.2f} (expected ~2)")


def test_history_flip_consistency() -> None:
    """history.flip() should give consistent perspective from opponent."""
    config = MatchConfig(termination_prob=0.0, max_rounds=10)
    result = play_match(TitForTat(), AlwaysDefect(), config)

    h_a = result.history_a
    h_b = h_a.flip()

    assert h_a.my_actions == h_b.opp_actions
    assert h_a.opp_actions == h_b.my_actions
    assert h_a.my_total_payoff == h_b.opp_total_payoff
    print(f"History flip is self-consistent")


if __name__ == "__main__":
    test_always_defect_vs_always_cooperate()
    test_tft_vs_tft_full_cooperation()
    test_tft_vs_always_defect()
    test_pavlov_recovers_from_mutual_defection()
    test_grim_trigger_never_forgives()
    test_generous_tft_forgiveness()
    test_probabilistic_termination()
    test_history_flip_consistency()
    print("\nAll classical strategy and match engine tests passed.")
