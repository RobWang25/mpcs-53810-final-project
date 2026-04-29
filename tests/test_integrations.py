"""
Integration tests for axelrod and nashpy.

We verify:
1. Axelrod-backed tournament produces results consistent with our hand-rolled
   tournament (small numerical differences are OK due to RNG / stochastic
   strategies; we check that classical results hold).
2. Nashpy-backed replicator dynamics gives qualitatively similar trajectories
   to our hand-rolled discrete update.
3. Mutation and Moran process give sensible additional analyses.

Run with: python -m tests.test_integrations
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
from fingerprints.metrics import compute_fingerprint
from integrations.axelrod_adapter import AXELROD_CLASSICAL_FACTORIES
from integrations.axelrod_tournament import run_tournament_axelrod
from integrations.nashpy_dynamics import (
    simulate_moran_process,
    simulate_replicator_nashpy,
    simulate_replicator_with_mutation,
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


def test_axelrod_tournament_replicates_classical_results() -> None:
    """Run an axelrod-backed tournament and check classical results hold."""
    factories = AXELROD_CLASSICAL_FACTORIES
    config = TournamentConfig(
        match_config=MatchConfig(termination_prob=0.0, max_rounds=50),
        n_replicates_per_pair=3,
        seed=42,
    )

    print("Running axelrod-backed tournament...")
    result = run_tournament_axelrod(factories, config, verbose=False)
    print(format_payoff_matrix(result))

    # Classical results to verify
    names = result.strategy_names
    M = result.payoff_matrix

    # AD vs AC: defector exploits perfectly
    ad_idx = names.index("AlwaysDefect")
    ac_idx = names.index("AlwaysCooperate")
    assert M[ad_idx, ac_idx] == 5.0, f"AD vs AC = {M[ad_idx, ac_idx]}, expected 5"
    assert M[ac_idx, ad_idx] == 0.0, f"AC vs AD = {M[ac_idx, ad_idx]}, expected 0"

    # TFT vs TFT: mutual cooperation
    tft_idx = names.index("TitForTat")
    assert M[tft_idx, tft_idx] == 3.0, f"TFT vs TFT = {M[tft_idx, tft_idx]}, expected 3"

    # AC vs AC: mutual cooperation
    assert M[ac_idx, ac_idx] == 3.0

    # AD vs AD: mutual defection
    assert M[ad_idx, ad_idx] == 1.0

    print("Axelrod tournament reproduces classical payoffs exactly")


def test_axelrod_tournament_works_with_mixed_strategies() -> None:
    """Mix axelrod's built-in strategies with our custom ones."""
    factories = {
        "TitForTat": AXELROD_CLASSICAL_FACTORIES["TitForTat"],  # Axelrod's
        "AlwaysDefect": AXELROD_CLASSICAL_FACTORIES["AlwaysDefect"],  # Axelrod's
        "OurTFT": TitForTat,  # Ours (will be auto-wrapped)
        "OurAD": AlwaysDefect,  # Ours
    }
    config = TournamentConfig(
        match_config=MatchConfig(termination_prob=0.0, max_rounds=20),
        n_replicates_per_pair=1,
        seed=42,
    )

    print("\nRunning mixed-source tournament (axelrod + custom)...")
    result = run_tournament_axelrod(factories, config, verbose=False)

    # Our TFT and axelrod's TFT should produce identical payoff rows
    M = result.payoff_matrix
    names = result.strategy_names
    tft_row = M[names.index("TitForTat")]
    our_tft_row = M[names.index("OurTFT")]
    np.testing.assert_array_equal(tft_row, our_tft_row)

    ad_row = M[names.index("AlwaysDefect")]
    our_ad_row = M[names.index("OurAD")]
    np.testing.assert_array_equal(ad_row, our_ad_row)

    print("Our custom strategies produce identical results to axelrod's built-ins")


def test_nashpy_replicator_matches_our_replicator() -> None:
    """Both replicator implementations should give qualitatively similar results."""
    # Build the same payoff matrix used in classical pipeline test
    factories = {
        "TitForTat": TitForTat,
        "AlwaysDefect": AlwaysDefect,
        "AlwaysCooperate": AlwaysCooperate,
        "Pavlov": Pavlov,
        "GrimTrigger": GrimTrigger,
        "GenerousTFT": lambda: GenerousTitForTat(forgiveness=0.1),
    }
    config = TournamentConfig(
        match_config=MatchConfig(termination_prob=0.0, max_rounds=50),
        n_replicates_per_pair=3,
        seed=42,
    )
    result = run_tournament(factories, config, verbose=False)

    n = len(result.strategy_names)
    initial = np.ones(n) / n
    rep_config = ReplicatorConfig(n_generations=300, convergence_threshold=1e-7)

    # Our discrete update
    our_traj = simulate_replicator(
        result.payoff_matrix, initial, result.strategy_names, rep_config
    )

    # nashpy continuous update
    nashpy_traj = simulate_replicator_nashpy(
        result.payoff_matrix, initial, result.strategy_names, rep_config
    )

    print("\nReplicator dynamics comparison:")
    print(f"  Our final dist:    {our_traj.final_distribution}")
    print(f"  Nashpy final dist: {nashpy_traj.final_distribution}")

    # Both should have AlwaysDefect dying out
    assert our_traj.final_distribution["AlwaysDefect"] < 0.01
    assert nashpy_traj.final_distribution["AlwaysDefect"] < 0.01

    # Both should rank the cooperative strategies similarly
    our_winners = sorted(
        our_traj.final_distribution.items(), key=lambda kv: -kv[1]
    )[:3]
    nashpy_winners = sorted(
        nashpy_traj.final_distribution.items(), key=lambda kv: -kv[1]
    )[:3]
    our_winner_names = {name for name, _ in our_winners}
    nashpy_winner_names = {name for name, _ in nashpy_winners}

    overlap = our_winner_names & nashpy_winner_names
    assert len(overlap) >= 2, (
        f"Top-3 winners differ too much: ours={our_winner_names}, "
        f"nashpy={nashpy_winner_names}"
    )
    print(f"Both methods agree on top survivors (overlap: {overlap})")


def test_replicator_with_mutation_keeps_strategies_alive() -> None:
    """With mutation, no strategy goes to exactly zero."""
    # Use the same matrix as before
    factories = {
        "TitForTat": TitForTat,
        "AlwaysDefect": AlwaysDefect,
        "AlwaysCooperate": AlwaysCooperate,
    }
    config = TournamentConfig(
        match_config=MatchConfig(termination_prob=0.0, max_rounds=50),
        n_replicates_per_pair=2,
        seed=42,
    )
    result = run_tournament(factories, config, verbose=False)

    n = len(result.strategy_names)
    initial = np.ones(n) / n
    rep_config = ReplicatorConfig(n_generations=300)

    traj = simulate_replicator_with_mutation(
        result.payoff_matrix,
        initial,
        result.strategy_names,
        mutation_rate=0.05,
        config=rep_config,
    )

    # All strategies should have non-trivial share (above mutation noise floor)
    for name, share in traj.final_distribution.items():
        assert share > 0.001, (
            f"With mutation, {name} should not go to zero; got {share}"
        )

    print(f"\nMutation keeps all strategies alive: "
          f"{ {n: f'{s:.4f}' for n, s in traj.final_distribution.items()} }")


def test_moran_process_picks_winner() -> None:
    """In a 2-strategy Moran process, the dominant strategy should fixate often."""
    # Simple 2-strategy game: TFT vs AD
    M = np.array([
        [3.0, 0.99],  # TFT row
        [1.04, 1.0],  # AD row
    ])
    strategy_names = ["TitForTat", "AlwaysDefect"]

    print("\nRunning Moran process (TFT vs AD, 50 individuals)...")
    result = simulate_moran_process(
        M,
        strategy_names,
        initial_population={"TitForTat": 25, "AlwaysDefect": 25},
        n_runs=20,
        seed=42,
    )
    print(f"Fixation probabilities: {result.fixation_probabilities}")

    # TFT should fixate more often than AD given these payoffs
    # (TFT's avg payoff in mixed pop is higher than AD's)
    assert result.fixation_counts["TitForTat"] >= result.fixation_counts["AlwaysDefect"], (
        f"TFT should fixate at least as often as AD: {result.fixation_counts}"
    )
    print("Moran process favors TFT in TFT/AD competition")


if __name__ == "__main__":
    test_axelrod_tournament_replicates_classical_results()
    test_axelrod_tournament_works_with_mixed_strategies()
    test_nashpy_replicator_matches_our_replicator()
    test_replicator_with_mutation_keeps_strategies_alive()
    test_moran_process_picks_winner()
    print("\nAll integration tests passed.")
