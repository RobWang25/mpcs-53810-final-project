"""
Tournament implementation backed by axelrod-python.

This is a drop-in replacement for tournament.tournament.run_tournament.
Internally it uses axelrod's Tournament class, which handles match scheduling,
scoring, and parallelism. We then convert results back to our TournamentResult
format so the rest of our pipeline (fingerprints, replicator dynamics, plotting)
continues to work unchanged.

Key benefits over our hand-rolled tournament:
- Battle-tested implementation (used in published research)
- Handles probabilistic termination natively via prob_end
- Proper random number management
- Easier to extend with axelrod's 200+ built-in strategies
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import axelrod as axl
import numpy as np

from integrations.axelrod_adapter import (
    build_game_history_from_axl,
    make_axelrod_player_factory,
)
from strategies.base import Strategy
from tournament.match import MatchResult
from tournament.tournament import TournamentConfig, TournamentResult
from utils.game import GameHistory


def run_tournament_axelrod(
    factories: dict[str, callable],
    config: TournamentConfig,
    verbose: bool = False,
) -> TournamentResult:
    """
    Run a round-robin tournament using axelrod-python as the backend.

    Args:
        factories: {name: callable_returning_fresh_player_or_strategy}.
                   Players can be axelrod.Player factories OR our Strategy
                   factories — we auto-wrap the latter.
        config: tournament settings (match config, replicates, etc.)

    Returns:
        TournamentResult in our standard format, suitable for downstream
        fingerprint and replicator dynamics analysis.
    """
    # Convert all factories to axelrod player factories
    axl_factories: dict[str, callable] = {}
    for name, factory in factories.items():
        # Test what kind of object the factory produces
        sample = factory()
        if isinstance(sample, axl.Player):
            axl_factories[name] = factory
        elif isinstance(sample, Strategy):
            axl_factories[name] = make_axelrod_player_factory(factory)
        else:
            raise TypeError(
                f"Factory '{name}' returned {type(sample).__name__}; expected "
                f"axelrod.Player or our Strategy"
            )

    names = list(axl_factories.keys())
    n = len(names)
    name_to_idx = {name: i for i, name in enumerate(names)}

    # Pre-build players for the tournament. axelrod Tournament takes a list
    # of player instances and runs all pairings.
    payoff_sum = np.zeros((n, n), dtype=float)
    coop_sum = np.zeros((n, n), dtype=float)
    rounds_sum = np.zeros((n, n), dtype=float)
    counts = np.zeros((n, n), dtype=int)

    histories_by_strategy: dict[str, list[GameHistory]] = defaultdict(list)
    match_results: dict[tuple[str, str], list[MatchResult]] = defaultdict(list)

    if verbose:
        print(f"[axelrod backend] Running {n}x{n} round-robin, "
              f"{config.n_replicates_per_pair} reps/pair...")

    # Run each pairing manually — gives us per-match access to histories,
    # which axelrod's bulk Tournament.play() doesn't easily expose.
    pair_idx = 0
    total_pairs = n * n if config.include_self_play else n * (n - 1)

    for i, name_a in enumerate(names):
        for j, name_b in enumerate(names):
            if not config.include_self_play and i == j:
                continue
            pair_idx += 1
            if verbose:
                print(f"  [{pair_idx}/{total_pairs}] {name_a} vs {name_b}")

            for rep in range(config.n_replicates_per_pair):
                player_a = axl_factories[name_a]()
                player_b = axl_factories[name_b]()

                # Use shadow-of-the-future via prob_end
                match_kwargs = {
                    "turns": config.match_config.max_rounds,
                }
                if config.match_config.termination_prob > 0:
                    match_kwargs["prob_end"] = config.match_config.termination_prob

                # axelrod uses its own RNG — we set seed for reproducibility
                seed = (
                    config.seed * 100000 + i * 1000 + j * 100 + rep
                )
                match = axl.Match(
                    [player_a, player_b], seed=seed, **match_kwargs
                )
                match.play()

                # Extract data
                rounds_played = len(player_a.history)
                if rounds_played == 0:
                    continue

                # Compute per-round avg payoff and cooperation rate
                scores = match.final_score()
                avg_payoff_a = float(scores[0]) / rounds_played
                avg_payoff_b = float(scores[1]) / rounds_played

                coop_a = sum(
                    1 for a in player_a.history if a == axl.Action.C
                ) / rounds_played
                coop_b = sum(
                    1 for a in player_b.history if a == axl.Action.C
                ) / rounds_played

                # Build our internal GameHistory for fingerprint computation
                history_a = build_game_history_from_axl(
                    list(player_a.history), list(player_b.history)
                )

                payoff_sum[i, j] += avg_payoff_a
                coop_sum[i, j] += coop_a
                rounds_sum[i, j] += rounds_played
                counts[i, j] += 1

                histories_by_strategy[name_a].append(history_a)

                # Build a MatchResult-compatible record for callers
                match_result = MatchResult(
                    strategy_a_name=name_a,
                    strategy_b_name=name_b,
                    history_a=history_a,
                    rounds_played=rounds_played,
                )
                match_results[(name_a, name_b)].append(match_result)

    with np.errstate(invalid="ignore", divide="ignore"):
        payoff_matrix = np.where(counts > 0, payoff_sum / np.maximum(counts, 1), 0.0)
        coop_matrix = np.where(counts > 0, coop_sum / np.maximum(counts, 1), 0.0)
        rounds_matrix = np.where(counts > 0, rounds_sum / np.maximum(counts, 1), 0.0)

    return TournamentResult(
        strategy_names=names,
        payoff_matrix=payoff_matrix,
        cooperation_matrix=coop_matrix,
        rounds_matrix=rounds_matrix,
        match_results=dict(match_results),
        histories_by_strategy=dict(histories_by_strategy),
    )
