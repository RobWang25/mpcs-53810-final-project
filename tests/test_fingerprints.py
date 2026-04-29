"""
Tests for behavioral fingerprint metrics.

We validate the metrics by computing fingerprints for classical strategies
playing against known opponents and checking that the resulting profile
matches the strategy's known characteristics.

Run with: python -m tests.test_fingerprints
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fingerprints.metrics import compute_fingerprint
from strategies.classical import (
    AlwaysCooperate,
    AlwaysDefect,
    GenerousTitForTat,
    GrimTrigger,
    Pavlov,
    TitForTat,
)
from tournament.match import MatchConfig, play_match


def collect_histories(strategy_factory, opponents, n_games_each=5, max_rounds=50):
    """Run `strategy` against each opponent multiple times; return histories."""
    histories = []
    for opp_factory in opponents:
        for game_idx in range(n_games_each):
            config = MatchConfig(
                termination_prob=0.0,
                max_rounds=max_rounds,
                seed=game_idx,
            )
            result = play_match(strategy_factory(), opp_factory(), config)
            histories.append(result.history_a)
    return histories


def test_always_cooperate_fingerprint() -> None:
    """AC: cooperation_rate=1.0, never retaliates, never opportunistic."""
    opponents = [AlwaysDefect, AlwaysCooperate, TitForTat]
    histories = collect_histories(AlwaysCooperate, opponents)
    fp = compute_fingerprint(histories)

    assert fp.cooperation_rate == 1.0
    assert fp.first_move_coop == 1.0
    assert fp.retaliation_rate == 0.0  # Never retaliates
    assert fp.opportunism_rate == 0.0  # Never opportunistic
    assert fp.good_partner_rate == 1.0  # Always reciprocates cooperation
    print(f"AlwaysCooperate fingerprint: {fp}")


def test_always_defect_fingerprint() -> None:
    """AD: cooperation_rate=0, always retaliates and is always opportunistic."""
    opponents = [AlwaysDefect, AlwaysCooperate, TitForTat]
    histories = collect_histories(AlwaysDefect, opponents)
    fp = compute_fingerprint(histories)

    assert fp.cooperation_rate == 0.0
    assert fp.first_move_coop == 0.0
    assert fp.retaliation_rate == 1.0  # Always defects after opp defection
    assert fp.opportunism_rate == 1.0  # Always defects after opp cooperation too
    assert fp.good_partner_rate == 0.0  # Never reciprocates cooperation
    print(f"AlwaysDefect fingerprint: {fp}")


def test_tft_fingerprint() -> None:
    """
    TFT: first_move=C (1.0), retaliation=1.0, opportunism=0.0, good_partner=1.0,
    forgiveness=0.0 (TFT is unforgiving — it sticks with retaliation until
    opponent cooperates).
    """
    opponents = [AlwaysDefect, AlwaysCooperate, TitForTat]
    histories = collect_histories(TitForTat, opponents)
    fp = compute_fingerprint(histories)

    assert fp.first_move_coop == 1.0
    assert fp.retaliation_rate == 1.0
    assert fp.opportunism_rate == 0.0
    assert fp.good_partner_rate == 1.0
    assert fp.forgiveness_rate == 0.0
    print(f"TitForTat fingerprint: {fp}")


def test_generous_tft_forgiveness() -> None:
    """
    Generous TFT with forgiveness=0.3: forgiveness rate after mutual D should
    be ~0.3 (it cooperates with prob 0.3 even after opp defection).
    """
    import random as random_module

    opponents = [AlwaysDefect]  # Force lots of mutual D situations
    histories = []
    for game_idx in range(20):
        config = MatchConfig(termination_prob=0.0, max_rounds=50, seed=game_idx)
        gtft = GenerousTitForTat(forgiveness=0.3, rng=random_module.Random(game_idx))
        result = play_match(gtft, AlwaysDefect(), config)
        histories.append(result.history_a)

    fp = compute_fingerprint(histories)
    # Retaliation rate should be ~0.7 (the complement of forgiveness)
    assert 0.55 < fp.retaliation_rate < 0.85, (
        f"Retaliation rate {fp.retaliation_rate} far from expected ~0.7"
    )
    print(
        f"GenerousTFT(0.3) retaliation rate: {fp.retaliation_rate:.3f} "
        f"(expected ~0.7)"
    )


def test_grim_trigger_fingerprint() -> None:
    """
    Grim Trigger: forgiveness rate must be 0.0 (never forgives). Retaliation
    rate is 1.0 (always defects after any defection).
    """
    opponents = [AlwaysDefect, AlwaysCooperate, TitForTat]
    histories = collect_histories(GrimTrigger, opponents)
    fp = compute_fingerprint(histories)

    assert fp.first_move_coop == 1.0
    assert fp.retaliation_rate == 1.0
    assert fp.forgiveness_rate == 0.0
    print(f"GrimTrigger fingerprint: {fp}")


def test_pavlov_fingerprint() -> None:
    """
    Pavlov is Win-Stay-Lose-Shift. Its fingerprint should be more nuanced:
    it forgives more than TFT (it switches from D to C after mutual D, since
    mutual D is a 'lose' for both), and it's opportunistic against AC
    (it stays with C after winning, but exploits naively against weak ones).
    Mainly we check that forgiveness rate > 0.
    """
    opponents = [AlwaysDefect, AlwaysCooperate, TitForTat]
    histories = collect_histories(Pavlov, opponents)
    fp = compute_fingerprint(histories)

    # Pavlov should actually have HIGH forgiveness vs AD because mutual D is a
    # "lose" and Pavlov shifts to C after losing.
    assert fp.forgiveness_rate > 0.5, (
        f"Pavlov forgiveness rate too low: {fp.forgiveness_rate}"
    )
    print(f"Pavlov fingerprint: {fp}")


if __name__ == "__main__":
    test_always_cooperate_fingerprint()
    test_always_defect_fingerprint()
    test_tft_fingerprint()
    test_generous_tft_forgiveness()
    test_grim_trigger_fingerprint()
    test_pavlov_fingerprint()
    print("\nAll fingerprint tests passed.")
