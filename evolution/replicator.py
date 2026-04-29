"""
Evolutionary dynamics on a payoff matrix.

Implements:
  - Discrete-time replicator dynamics
  - ESS (Evolutionarily Stable Strategy) invasion tests
  - Trajectory simulation and convergence detection

The payoff matrix is treated as fixed input — it comes from a tournament
run and represents stationary strategies (see proposal: stationarity
assumption is explicit).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ReplicatorConfig:
    """Configuration for a replicator dynamics simulation."""

    n_generations: int = 500
    convergence_threshold: float = 1e-6  # Stop early when max change < threshold
    min_share: float = 1e-9  # Floor on population shares to avoid extinction artifacts


@dataclass
class ReplicatorTrajectory:
    """Result of a replicator dynamics simulation."""

    strategy_names: list[str]
    # Shape: (n_generations + 1, n_strategies). Row 0 is initial, last row is final.
    history: np.ndarray
    converged: bool
    final_generation: int

    @property
    def final_distribution(self) -> dict[str, float]:
        return {
            name: float(self.history[self.final_generation, i])
            for i, name in enumerate(self.strategy_names)
        }

    @property
    def initial_distribution(self) -> dict[str, float]:
        return {
            name: float(self.history[0, i]) for i, name in enumerate(self.strategy_names)
        }

    def share_trajectory(self, name: str) -> np.ndarray:
        """Get the time series of population shares for a single strategy."""
        idx = self.strategy_names.index(name)
        return self.history[: self.final_generation + 1, idx]


def replicator_step(x: np.ndarray, M: np.ndarray) -> np.ndarray:
    """
    Single discrete replicator dynamics step.

    Args:
        x: population share vector, shape (n,), sums to 1.
        M: payoff matrix, M[i, j] = payoff strategy i earns vs strategy j.

    Returns:
        Updated population share vector.

    The update rule is:
        x_i_new = x_i * fitness_i / mean_fitness
    where fitness_i = M @ x and mean_fitness = x @ M @ x.

    Payoffs must be non-negative for this update to be well-defined. All our
    PD payoffs are >= 0 by construction (S = 0 is the minimum).
    """
    fitness = M @ x  # Shape (n,)
    mean_fitness = float(x @ fitness)
    if mean_fitness <= 0:
        # Degenerate case — all strategies earning 0. Keep current distribution.
        return x.copy()
    x_new = x * fitness / mean_fitness
    # Renormalize to handle floating point drift
    s = x_new.sum()
    if s > 0:
        x_new = x_new / s
    return x_new


def simulate_replicator(
    M: np.ndarray,
    initial_distribution: np.ndarray,
    strategy_names: list[str],
    config: ReplicatorConfig | None = None,
) -> ReplicatorTrajectory:
    """
    Run discrete replicator dynamics from a given initial distribution.

    The simulation stops early if the change in distribution from one
    generation to the next drops below `convergence_threshold`.
    """
    if config is None:
        config = ReplicatorConfig()

    n = len(strategy_names)
    if M.shape != (n, n):
        raise ValueError(f"M shape {M.shape} doesn't match {n} strategies")
    if initial_distribution.shape != (n,):
        raise ValueError(
            f"initial distribution shape {initial_distribution.shape} != ({n},)"
        )
    if not np.isclose(initial_distribution.sum(), 1.0, atol=1e-6):
        raise ValueError(
            f"initial distribution must sum to 1, got {initial_distribution.sum()}"
        )

    history = np.zeros((config.n_generations + 1, n))
    history[0] = initial_distribution

    converged = False
    final_gen = config.n_generations

    x = initial_distribution.copy()
    for gen in range(config.n_generations):
        x_new = replicator_step(x, M)
        # Apply minimum-share floor to prevent numerical extinction
        x_new = np.maximum(x_new, config.min_share)
        x_new = x_new / x_new.sum()

        history[gen + 1] = x_new

        if np.max(np.abs(x_new - x)) < config.convergence_threshold:
            converged = True
            final_gen = gen + 1
            break

        x = x_new

    return ReplicatorTrajectory(
        strategy_names=strategy_names,
        history=history,
        converged=converged,
        final_generation=final_gen,
    )


@dataclass
class ESSTestResult:
    """Result of testing whether `incumbent` is stable against `invader`."""

    incumbent: str
    invader: str
    initial_invader_share: float
    final_invader_share: float
    invader_grew: bool  # True if invader's share increased
    is_stable: bool  # True if incumbent successfully resisted invasion


def test_ess_invasion(
    M: np.ndarray,
    strategy_names: list[str],
    incumbent: str,
    invader: str,
    invader_share: float = 0.05,
    config: ReplicatorConfig | None = None,
) -> ESSTestResult:
    """
    Test whether `incumbent` resists invasion by `invader`.

    Initialize the population with (1 - invader_share) of incumbent and
    `invader_share` of invader. Run replicator dynamics. If the invader's
    share decreases, the incumbent is stable against this invader.
    """
    n = len(strategy_names)
    name_to_idx = {name: i for i, name in enumerate(strategy_names)}

    if incumbent not in name_to_idx:
        raise ValueError(f"Unknown incumbent: {incumbent}")
    if invader not in name_to_idx:
        raise ValueError(f"Unknown invader: {invader}")

    inc_idx = name_to_idx[incumbent]
    inv_idx = name_to_idx[invader]

    initial = np.zeros(n)
    initial[inc_idx] = 1.0 - invader_share
    initial[inv_idx] = invader_share

    trajectory = simulate_replicator(M, initial, strategy_names, config)

    final_inv_share = trajectory.final_distribution[invader]
    invader_grew = final_inv_share > invader_share + 1e-4
    is_stable = not invader_grew

    return ESSTestResult(
        incumbent=incumbent,
        invader=invader,
        initial_invader_share=invader_share,
        final_invader_share=final_inv_share,
        invader_grew=invader_grew,
        is_stable=is_stable,
    )


def test_ess_against_all(
    M: np.ndarray,
    strategy_names: list[str],
    incumbent: str,
    invader_share: float = 0.05,
    config: ReplicatorConfig | None = None,
) -> dict[str, ESSTestResult]:
    """
    Test whether `incumbent` resists invasion by every other strategy.

    Returns dict mapping invader_name -> ESSTestResult.
    """
    results = {}
    for invader in strategy_names:
        if invader == incumbent:
            continue
        results[invader] = test_ess_invasion(
            M, strategy_names, incumbent, invader, invader_share, config
        )
    return results
