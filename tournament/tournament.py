"""
Round-robin tournament engine.

Runs a full pairwise tournament between a set of strategies, with K replicate
matches per pair to handle stochasticity. Produces:

  1. A payoff matrix (per-round average payoff for strategy i against j)
  2. Per-strategy game histories — used downstream to compute fingerprints
  3. A summary table of total/average payoffs per strategy

The tournament treats each strategy as a *factory* (callable that returns
a fresh instance), not a singleton instance. This is critical for stateful
strategies like Grim Trigger that need a fresh start for each match.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from strategies.base import Strategy
from tournament.match import MatchConfig, MatchResult, play_match
from utils.game import GameHistory


StrategyFactory = Callable[[], Strategy]


@dataclass
class TournamentConfig:
    """Configuration for the round-robin tournament."""

    match_config: MatchConfig
    n_replicates_per_pair: int = 5  # Match repetitions for statistical reliability
    include_self_play: bool = True  # Include strategy vs. itself
    seed: int = 0


@dataclass
class TournamentResult:
    """All results from a completed tournament."""

    strategy_names: list[str]
    # payoff_matrix[i, j] = average per-round payoff strategy i earns against j
    payoff_matrix: np.ndarray
    # cooperation_matrix[i, j] = average cooperation rate of strategy i against j
    cooperation_matrix: np.ndarray
    # rounds_matrix[i, j] = average rounds played per match between i and j
    rounds_matrix: np.ndarray
    # All match results, keyed by (strategy_i_name, strategy_j_name)
    match_results: dict[tuple[str, str], list[MatchResult]] = field(default_factory=dict)
    # Per-strategy histories: {strategy_name: list[GameHistory]}
    histories_by_strategy: dict[str, list[GameHistory]] = field(default_factory=dict)

    def total_payoff_per_strategy(self) -> dict[str, float]:
        """Sum of avg per-round payoffs against all opponents."""
        return {
            name: float(self.payoff_matrix[i].sum())
            for i, name in enumerate(self.strategy_names)
        }

    def avg_payoff_per_strategy(self) -> dict[str, float]:
        """Average of avg per-round payoffs across all opponents."""
        n = len(self.strategy_names)
        return {
            name: float(self.payoff_matrix[i].mean())
            for i, name in enumerate(self.strategy_names)
        }

    def ranked_strategies(self) -> list[tuple[str, float]]:
        """Strategies ranked by average payoff, descending."""
        avg = self.avg_payoff_per_strategy()
        return sorted(avg.items(), key=lambda kv: -kv[1])

    def to_dict(self) -> dict:
        """Serialize the numerical part of results to JSON-compatible dict."""
        return {
            "strategy_names": self.strategy_names,
            "payoff_matrix": self.payoff_matrix.tolist(),
            "cooperation_matrix": self.cooperation_matrix.tolist(),
            "rounds_matrix": self.rounds_matrix.tolist(),
            "avg_payoff_per_strategy": self.avg_payoff_per_strategy(),
        }


def run_tournament(
    factories: dict[str, StrategyFactory],
    config: TournamentConfig,
    verbose: bool = False,
) -> TournamentResult:
    """
    Run a full round-robin tournament between all strategies.

    Args:
        factories: dict mapping strategy name -> callable producing fresh instances.
        config: tournament settings.
        verbose: print progress.
    """
    names = list(factories.keys())
    n = len(names)
    name_to_idx = {name: i for i, name in enumerate(names)}

    # Sums and counts to compute averages
    payoff_sum = np.zeros((n, n), dtype=float)
    coop_sum = np.zeros((n, n), dtype=float)
    rounds_sum = np.zeros((n, n), dtype=float)
    counts = np.zeros((n, n), dtype=int)

    match_results: dict[tuple[str, str], list[MatchResult]] = defaultdict(list)
    histories_by_strategy: dict[str, list[GameHistory]] = defaultdict(list)

    total_pairs = n * n if config.include_self_play else n * (n - 1)
    pair_idx = 0

    for i, name_a in enumerate(names):
        for j, name_b in enumerate(names):
            if not config.include_self_play and i == j:
                continue
            pair_idx += 1
            if verbose:
                print(
                    f"[{pair_idx}/{total_pairs}] {name_a} vs {name_b} "
                    f"× {config.n_replicates_per_pair}"
                )

            for rep in range(config.n_replicates_per_pair):
                # Different seed per replicate for stochastic termination
                rep_match_config = MatchConfig(
                    termination_prob=config.match_config.termination_prob,
                    max_rounds=config.match_config.max_rounds,
                    min_rounds=config.match_config.min_rounds,
                    seed=config.seed * 10000 + i * 100 + j * 10 + rep,
                )
                strat_a = factories[name_a]()
                strat_b = factories[name_b]()
                result = play_match(strat_a, strat_b, rep_match_config)

                payoff_sum[i, j] += result.avg_payoff_a
                coop_sum[i, j] += result.coop_rate_a
                rounds_sum[i, j] += result.rounds_played
                counts[i, j] += 1

                match_results[(name_a, name_b)].append(result)
                histories_by_strategy[name_a].append(result.history_a)

    # Compute averages, avoiding division by zero
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


def save_payoff_matrix(result: TournamentResult, path: str) -> None:
    """Save the numerical results to a JSON file."""
    with open(path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)


def format_payoff_matrix(result: TournamentResult, decimals: int = 2) -> str:
    """Pretty-print the payoff matrix as a table."""
    names = result.strategy_names
    M = result.payoff_matrix

    # Column widths
    name_w = max(len(n) for n in names) + 2
    cell_w = max(8, decimals + 5)

    lines = []
    header = " " * name_w + "".join(f"{n:>{cell_w}}" for n in names)
    lines.append(header)
    lines.append("-" * len(header))
    for i, name_i in enumerate(names):
        row = f"{name_i:<{name_w}}"
        for j in range(len(names)):
            row += f"{M[i, j]:>{cell_w}.{decimals}f}"
        lines.append(row)
    return "\n".join(lines)
