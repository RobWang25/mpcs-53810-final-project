"""
Replicator dynamics and Moran processes via nashpy.

This module wraps nashpy's well-tested implementations and exposes them in
the same `ReplicatorTrajectory` format as our hand-rolled module — so the
plotting and analysis code keeps working unchanged.

Adds three things our custom implementation didn't have:
- Continuous-time replicator dynamics (ODE integration via scipy)
- Replicator-mutation dynamics (replicator with imperfect reproduction)
- Moran process (finite-population stochastic alternative)

These give us robustness checks for our headline replicator-dynamics findings.
"""

from __future__ import annotations

from dataclasses import dataclass

import nashpy as nash
import numpy as np

from evolution.replicator import ReplicatorConfig, ReplicatorTrajectory


def simulate_replicator_nashpy(
    M: np.ndarray,
    initial_distribution: np.ndarray,
    strategy_names: list[str],
    config: ReplicatorConfig | None = None,
) -> ReplicatorTrajectory:
    """
    Continuous-time replicator dynamics via nashpy.

    Internally nashpy uses scipy.integrate.odeint to solve the replicator
    ODE. Returns a `ReplicatorTrajectory` in our standard format so it
    plugs into existing plotting/analysis code unchanged.

    Note: nashpy's continuous-time integration produces smoother trajectories
    than our discrete update. Convergence detection is approximate — we mark
    converged=True if the last 10% of generations show < threshold change.
    """
    if config is None:
        config = ReplicatorConfig()

    n = len(strategy_names)
    if M.shape != (n, n):
        raise ValueError(f"M shape {M.shape} doesn't match {n} strategies")

    game = nash.Game(M)
    timepoints = np.linspace(0, config.n_generations, config.n_generations + 1)
    history = game.replicator_dynamics(y0=initial_distribution, timepoints=timepoints)

    # nashpy returns shape (n_timepoints, n_strategies)
    final_gen = history.shape[0] - 1

    # Convergence check on last 10%
    tail_size = max(1, final_gen // 10)
    tail = history[-tail_size:]
    max_change = np.max(np.abs(np.diff(tail, axis=0))) if tail.shape[0] > 1 else 0.0
    converged = max_change < config.convergence_threshold

    return ReplicatorTrajectory(
        strategy_names=strategy_names,
        history=history,
        converged=converged,
        final_generation=final_gen,
    )


def simulate_replicator_with_mutation(
    M: np.ndarray,
    initial_distribution: np.ndarray,
    strategy_names: list[str],
    mutation_rate: float = 0.01,
    config: ReplicatorConfig | None = None,
) -> ReplicatorTrajectory:
    """
    Replicator dynamics with mutation — robustness check for our results.

    With mutation, a small fraction of each strategy "mutates" into other
    strategies each generation. This breaks the strict assumption of perfect
    reproduction and prevents strategies from going to exactly zero — which
    is unrealistic in a real population.

    Args:
        mutation_rate: probability of mutation per generation. We use a
                       symmetric mutation matrix where each strategy mutates
                       to any other strategy with equal probability.

    The mutation matrix Q has Q[i, j] = probability strategy i produces
    offspring of type j. The diagonal is (1 - mutation_rate); off-diagonal
    entries sum to mutation_rate, divided equally.
    """
    if config is None:
        config = ReplicatorConfig()

    n = len(strategy_names)
    if M.shape != (n, n):
        raise ValueError(f"M shape {M.shape} doesn't match {n} strategies")
    if not 0 <= mutation_rate < 1:
        raise ValueError(f"mutation_rate must be in [0, 1), got {mutation_rate}")

    game = nash.Game(M)

    # Symmetric mutation matrix: equal mutation prob to all other strategies
    if n > 1:
        off_diag = mutation_rate / (n - 1)
    else:
        off_diag = 0.0
    Q = np.full((n, n), off_diag)
    np.fill_diagonal(Q, 1.0 - mutation_rate)

    timepoints = np.linspace(0, config.n_generations, config.n_generations + 1)
    history = game.replicator_dynamics(
        y0=initial_distribution, timepoints=timepoints, mutation_matrix=Q
    )

    final_gen = history.shape[0] - 1
    tail_size = max(1, final_gen // 10)
    tail = history[-tail_size:]
    max_change = np.max(np.abs(np.diff(tail, axis=0))) if tail.shape[0] > 1 else 0.0
    converged = max_change < config.convergence_threshold

    return ReplicatorTrajectory(
        strategy_names=strategy_names,
        history=history,
        converged=converged,
        final_generation=final_gen,
    )


@dataclass
class MoranResult:
    """Result of a Moran process simulation (finite-population stochastic)."""

    strategy_names: list[str]
    fixation_counts: dict[str, int]  # How many runs each strategy fixated
    n_runs: int

    @property
    def fixation_probabilities(self) -> dict[str, float]:
        return {
            name: count / self.n_runs for name, count in self.fixation_counts.items()
        }


def simulate_moran_process(
    M: np.ndarray,
    strategy_names: list[str],
    initial_population: dict[str, int],
    n_runs: int = 50,
    seed: int = 42,
) -> MoranResult:
    """
    Run a Moran process — finite-population evolutionary dynamics.

    Unlike replicator dynamics (deterministic, infinite population), the
    Moran process is stochastic and tracks integer counts of each strategy.
    At each step, one individual is chosen for reproduction (proportional
    to fitness) and one is chosen at random to die. The process eventually
    fixates — only one strategy remains.

    Use this as a robustness check: if your replicator-dynamics conclusion
    is "TFT dominates," running Moran processes from the same initial
    distribution should show TFT fixating most often.

    Args:
        initial_population: {strategy_name: count}. Total determines pop size.
        n_runs: number of independent Moran runs.

    Returns:
        MoranResult with fixation counts per strategy.
    """
    n = len(strategy_names)
    if M.shape != (n, n):
        raise ValueError(f"M shape {M.shape} doesn't match {n} strategies")

    # Build initial player list from population dict
    initial_players = []
    for name in strategy_names:
        count = initial_population.get(name, 0)
        initial_players.extend([name] * count)

    if not initial_players:
        raise ValueError("initial_population must contain at least one player")

    fixation_counts = {name: 0 for name in strategy_names}

    # Use nashpy's Moran process. It returns the fixated type after termination.
    game = nash.Game(M)

    rng = np.random.default_rng(seed)

    for run_idx in range(n_runs):
        # We implement Moran ourselves for simpler control, since nashpy's API
        # is a bit awkward for our string-named strategies. Standard Moran:
        # Initialize counts.
        counts = np.array(
            [initial_population.get(name, 0) for name in strategy_names], dtype=float
        )
        N = counts.sum()

        run_rng = np.random.default_rng(seed * 10000 + run_idx)

        # Run until fixation (one strategy has count == N)
        max_steps = int(N * N * 100)  # Safety cap
        for step in range(max_steps):
            if (counts > 0).sum() == 1:
                break
            # Compute fitness of each strategy (avg payoff against current pop)
            # Probabilities for fitness-proportional reproduction
            x = counts / N
            fitness = M @ x  # Avg payoff of each strategy against current mix
            # Birth: pick a strategy weighted by count * fitness
            birth_weights = counts * fitness
            if birth_weights.sum() <= 0:
                break  # All extinct or zero fitness
            birth_probs = birth_weights / birth_weights.sum()
            birth = run_rng.choice(n, p=birth_probs)
            # Death: pick a strategy weighted by count alone (uniform individual)
            death_probs = counts / counts.sum()
            death = run_rng.choice(n, p=death_probs)
            counts[birth] += 1
            counts[death] -= 1

        # Find fixated strategy (the one with count == N, or the most common)
        fixated_idx = int(np.argmax(counts))
        fixation_counts[strategy_names[fixated_idx]] += 1

    return MoranResult(
        strategy_names=strategy_names,
        fixation_counts=fixation_counts,
        n_runs=n_runs,
    )
