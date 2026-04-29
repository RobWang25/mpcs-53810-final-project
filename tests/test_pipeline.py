"""
End-to-end pipeline test using only classical strategies.

This validates the entire pipeline (tournament -> payoff matrix -> replicator
dynamics -> ESS testing) and reproduces classical evolutionary game theory
results — for instance, that TFT is more robust than AC in mixed populations.

Run with: python -m tests.test_pipeline
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from evolution.replicator import (
    ReplicatorConfig,
    simulate_replicator,
    test_ess_against_all,
)
from strategies.classical import (
    AlwaysCooperate,
    AlwaysDefect,
    GenerousTitForTat,
    GrimTrigger,
    Pavlov,
    TitForTat,
)
from tournament.match import MatchConfig
from tournament.tournament import (
    TournamentConfig,
    format_payoff_matrix,
    run_tournament,
)


def test_full_pipeline() -> None:
    """Run the full pipeline on classical strategies."""
    factories = {
        "TitForTat": TitForTat,
        "AlwaysDefect": AlwaysDefect,
        "AlwaysCooperate": AlwaysCooperate,
        "Pavlov": Pavlov,
        "GrimTrigger": GrimTrigger,
        "GenerousTFT": lambda: GenerousTitForTat(forgiveness=0.1),
    }

    print("Running tournament...")
    tournament_config = TournamentConfig(
        match_config=MatchConfig(termination_prob=0.0, max_rounds=100),
        n_replicates_per_pair=3,
        seed=42,
    )
    result = run_tournament(factories, tournament_config, verbose=False)

    print("\nPayoff matrix (avg payoff per round):")
    print(format_payoff_matrix(result))

    print("\nStrategy ranking by avg payoff:")
    for name, score in result.ranked_strategies():
        print(f"  {name:<20} {score:.3f}")

    # Validate some classical results
    avg = result.avg_payoff_per_strategy()
    # AD should beat AC head-to-head (AD vs AC payoff vs AC vs AD payoff)
    payoff_ad_vs_ac = result.payoff_matrix[
        result.strategy_names.index("AlwaysDefect"),
        result.strategy_names.index("AlwaysCooperate"),
    ]
    payoff_ac_vs_ad = result.payoff_matrix[
        result.strategy_names.index("AlwaysCooperate"),
        result.strategy_names.index("AlwaysDefect"),
    ]
    assert payoff_ad_vs_ac == 5.0, f"AD vs AC payoff {payoff_ad_vs_ac} != 5"
    assert payoff_ac_vs_ad == 0.0, f"AC vs AD payoff {payoff_ac_vs_ad} != 0"

    # TFT vs TFT should be 3.0 (mutual cooperation)
    payoff_tft_vs_tft = result.payoff_matrix[
        result.strategy_names.index("TitForTat"),
        result.strategy_names.index("TitForTat"),
    ]
    assert payoff_tft_vs_tft == 3.0, f"TFT vs TFT payoff {payoff_tft_vs_tft} != 3"

    print("\nTournament payoff matrix is consistent with theory")

    # Run replicator dynamics from uniform initial distribution
    print("\nRunning replicator dynamics from uniform initial distribution...")
    n = len(result.strategy_names)
    initial = np.ones(n) / n
    rep_config = ReplicatorConfig(n_generations=500, convergence_threshold=1e-6)
    trajectory = simulate_replicator(
        result.payoff_matrix, initial, result.strategy_names, rep_config
    )

    print(f"Converged: {trajectory.converged} after {trajectory.final_generation} generations")
    print("Final distribution:")
    final_dist = trajectory.final_distribution
    for name, share in sorted(final_dist.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<20} {share:.4f}")

    # Classical result: AlwaysDefect should die out in a population dominated by
    # nice retaliatory strategies. AC can survive once AD is extinct because
    # there's no one left to exploit it — this is a known artifact of
    # uniform-mixed initial conditions.
    ad_share = final_dist["AlwaysDefect"]
    assert ad_share < 0.01, f"AlwaysDefect should die out, but has share {ad_share}"
    print(f"\nAlwaysDefect correctly dies out ({ad_share:.4f}) — classical Axelrod result")
    print("   (Note: AlwaysCooperate survives because once AD is extinct, no one exploits it)")

    # Test ESS: is TFT stable against invaders?
    print("\nESS test: TFT vs all invaders")
    ess_results = test_ess_against_all(
        result.payoff_matrix,
        result.strategy_names,
        incumbent="TitForTat",
        invader_share=0.05,
    )
    for invader, res in ess_results.items():
        status = "STABLE" if res.is_stable else "INVADED"
        print(
            f"  {invader:<20} -> {status:8} "
            f"(invader share: {res.initial_invader_share:.3f} -> "
            f"{res.final_invader_share:.3f})"
        )

    # AD invading TFT: AD shouldn't be able to invade a population of TFT
    # (because TFT vs TFT payoff = 3 > AD vs TFT payoff which is mostly 1 with one D defection)
    tft_vs_ad_invasion = ess_results["AlwaysDefect"]
    assert tft_vs_ad_invasion.is_stable, (
        "TFT should be stable against AD invasion in classical IPD"
    )
    print("\nTFT correctly resists AD invasion (classical Axelrod result)")

    return result, trajectory


if __name__ == "__main__":
    test_full_pipeline()
    print("\nFull pipeline test passed.")
