"""
Main experiment runner.

Runs the full pipeline for a given experimental configuration:
  1. Round-robin tournament
  2. Compute behavioral fingerprints for each strategy
  3. Replicator dynamics from uniform initial distribution
  4. ESS invasion tests for each strategy
  5. Save data + plots

Used both for the validation run (classical strategies only) and for the
real experiment (LLMs + classical).

Usage:
    python -m run_experiment --mode classical
    python -m run_experiment --mode mock_llm
    python -m run_experiment --mode real_llm  (requires API keys)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

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
from llm.client import MockLLMClient
from llm.prompts import FramingCondition
from strategies.classical import (
    AlwaysCooperate,
    AlwaysDefect,
    GenerousTitForTat,
    GrimTrigger,
    Pavlov,
    TitForTat,
)
from strategies.llm_strategy import LLMStrategy
from tournament.match import MatchConfig
from tournament.tournament import (
    TournamentConfig,
    format_payoff_matrix,
    run_tournament,
    save_payoff_matrix,
)
from utils.plotting import (
    plot_fingerprint_radar,
    plot_fingerprint_table,
    plot_payoff_heatmap,
    plot_population_trajectory,
)

logger = logging.getLogger(__name__)


def get_classical_factories() -> dict:
    """
    Return the standard set of classical strategy factories.

    Uses axelrod-python's canonical implementations by default. Each
    factory returns a fresh axelrod.Player instance.
    """
    return dict(AXELROD_CLASSICAL_FACTORIES)


def get_classical_factories_custom() -> dict:
    """Use our hand-rolled classical strategies (for cross-validation)."""
    return {
        "TitForTat": TitForTat,
        "AlwaysDefect": AlwaysDefect,
        "AlwaysCooperate": AlwaysCooperate,
        "Pavlov": Pavlov,
        "GrimTrigger": GrimTrigger,
        "GenerousTFT": lambda: GenerousTitForTat(forgiveness=0.1),
    }


def get_mock_llm_factories() -> dict:
    """Return mock LLM factories — for cost-free pipeline testing."""

    def make_factory(coop_prob: float, framing: FramingCondition, name: str):
        def factory():
            client = MockLLMClient(
                coop_probability=coop_prob, model_id=f"mock-{name}"
            )
            return LLMStrategy(client, framing, custom_name=name)

        return factory

    return {
        # Three "Claude-like" mocks — high cooperation, varies by framing
        "MockClaude_Neutral": make_factory(0.85, FramingCondition.NEUTRAL, "MockClaude_Neutral"),
        "MockClaude_Rational": make_factory(0.65, FramingCondition.RATIONAL, "MockClaude_Rational"),
        "MockClaude_Coop": make_factory(0.92, FramingCondition.COOPERATIVE, "MockClaude_Coop"),
        # Three "GPT-like" mocks — moderate cooperation, more responsive to framing
        "MockGPT_Neutral": make_factory(0.70, FramingCondition.NEUTRAL, "MockGPT_Neutral"),
        "MockGPT_Rational": make_factory(0.45, FramingCondition.RATIONAL, "MockGPT_Rational"),
        "MockGPT_Coop": make_factory(0.85, FramingCondition.COOPERATIVE, "MockGPT_Coop"),
    }


def get_real_llm_factories() -> dict:
    """
    Return real LLM factories using actual API clients.

    NOTE: Requires ANTHROPIC_API_KEY and OPENAI_API_KEY environment variables.
    Each tournament call costs real money — see budget estimate in proposal.
    """
    from llm.anthropic_client import AnthropicClient
    from llm.openai_client import OpenAIClient

    def make_anthropic_factory(framing: FramingCondition, label: str):
        def factory():
            # Note: each LLMStrategy creates a fresh client, so we share one.
            # In practice we share the *client* but make a fresh strategy.
            # The client itself is stateless, so this is safe.
            return LLMStrategy(
                AnthropicClient(model="claude-sonnet-4-6"),
                framing,
                custom_name=label,
            )

        return factory

    def make_openai_factory(framing: FramingCondition, label: str):
        def factory():
            return LLMStrategy(
                OpenAIClient(model="gpt-4o"),
                framing,
                custom_name=label,
            )

        return factory

    return {
        "Claude_Neutral": make_anthropic_factory(FramingCondition.NEUTRAL, "Claude_Neutral"),
        "Claude_Rational": make_anthropic_factory(FramingCondition.RATIONAL, "Claude_Rational"),
        "Claude_Coop": make_anthropic_factory(FramingCondition.COOPERATIVE, "Claude_Coop"),
        "GPT_Neutral": make_openai_factory(FramingCondition.NEUTRAL, "GPT_Neutral"),
        "GPT_Rational": make_openai_factory(FramingCondition.RATIONAL, "GPT_Rational"),
        "GPT_Coop": make_openai_factory(FramingCondition.COOPERATIVE, "GPT_Coop"),
    }


def run_experiment(
    factories: dict,
    output_dir: Path,
    termination_prob: float = 0.10,
    max_rounds: int = 100,
    n_replicates: int = 5,
    n_generations: int = 500,
    seed: int = 42,
    label: str = "experiment",
    backend: str = "axelrod",
    mutation_rate: float = 0.01,
    n_moran_runs: int = 30,
    moran_population_size: int = 50,
) -> dict:
    """
    Run the full experimental pipeline.

    Args:
        backend: 'axelrod' (default, uses axelrod-python) or 'custom'
                 (uses our hand-rolled tournament).
        mutation_rate: rate for replicator-mutation analysis. Set 0 to skip.
        n_moran_runs: number of Moran process runs for stochastic analysis.
                      Set 0 to skip.
        moran_population_size: total population size for Moran process.

    Returns a dict with key findings + paths to saved artifacts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Tournament
    print(f"\n{'=' * 60}")
    print(f"Running tournament: {label}  [backend: {backend}]")
    print(f"{'=' * 60}")
    print(f"Strategies: {list(factories.keys())}")
    print(f"Termination prob: {termination_prob}, Max rounds: {max_rounds}")
    print(f"Replicates per pair: {n_replicates}")

    tournament_config = TournamentConfig(
        match_config=MatchConfig(
            termination_prob=termination_prob,
            max_rounds=max_rounds,
        ),
        n_replicates_per_pair=n_replicates,
        seed=seed,
    )
    if backend == "axelrod":
        result = run_tournament_axelrod(factories, tournament_config, verbose=True)
    elif backend == "custom":
        result = run_tournament(factories, tournament_config, verbose=True)
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'axelrod' or 'custom'.")

    print("\nPayoff matrix:")
    print(format_payoff_matrix(result))

    print("\nRanking by avg payoff:")
    for name, score in result.ranked_strategies():
        print(f"  {name:<24} {score:.3f}")

    save_payoff_matrix(result, str(output_dir / "payoff_matrix.json"))

    fig_payoff = plot_payoff_heatmap(result, title=f"{label}: payoff matrix")
    fig_payoff.savefig(output_dir / "payoff_matrix.png", dpi=150)
    print(f"\nSaved payoff heatmap to {output_dir / 'payoff_matrix.png'}")

    # 2. Fingerprints
    print("\nComputing behavioral fingerprints...")
    fingerprints = {
        name: compute_fingerprint(result.histories_by_strategy[name])
        for name in result.strategy_names
    }

    print("\nFingerprints:")
    for name, fp in fingerprints.items():
        print(f"  {name:<24} {fp}")

    # Save fingerprints to JSON
    fp_dict = {name: fp.to_dict() for name, fp in fingerprints.items()}
    with open(output_dir / "fingerprints.json", "w") as f:
        json.dump(fp_dict, f, indent=2)

    fig_radar = plot_fingerprint_radar(
        fingerprints, title=f"{label}: behavioral fingerprints"
    )
    fig_radar.savefig(output_dir / "fingerprints_radar.png", dpi=150, bbox_inches="tight")

    fig_table = plot_fingerprint_table(
        fingerprints, title=f"{label}: fingerprint metrics"
    )
    fig_table.savefig(output_dir / "fingerprints_table.png", dpi=150, bbox_inches="tight")
    print(f"Saved fingerprint plots to {output_dir}")

    # 3. Replicator dynamics (via nashpy — continuous-time)
    print("\nRunning replicator dynamics via nashpy from uniform initial dist...")
    n = len(result.strategy_names)
    initial = np.ones(n) / n
    rep_config = ReplicatorConfig(n_generations=n_generations)
    trajectory = simulate_replicator_nashpy(
        result.payoff_matrix, initial, result.strategy_names, rep_config
    )
    print(
        f"Converged: {trajectory.converged} after {trajectory.final_generation} "
        f"generations"
    )
    print("Final population shares:")
    for name, share in sorted(trajectory.final_distribution.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<24} {share:.4f}")

    fig_traj = plot_population_trajectory(
        trajectory, title=f"{label}: replicator dynamics (nashpy)"
    )
    fig_traj.savefig(output_dir / "replicator_trajectory.png", dpi=150)
    print(f"Saved trajectory plot to {output_dir / 'replicator_trajectory.png'}")

    # 3b. Replicator-mutation dynamics (robustness check)
    if mutation_rate > 0:
        print(f"\nRunning replicator-mutation dynamics (rate={mutation_rate})...")
        mut_trajectory = simulate_replicator_with_mutation(
            result.payoff_matrix,
            initial,
            result.strategy_names,
            mutation_rate=mutation_rate,
            config=rep_config,
        )
        print("Final shares with mutation:")
        for name, share in sorted(
            mut_trajectory.final_distribution.items(), key=lambda kv: -kv[1]
        ):
            print(f"  {name:<24} {share:.4f}")

        fig_mut = plot_population_trajectory(
            mut_trajectory,
            title=f"{label}: replicator-mutation dynamics (rate={mutation_rate})",
        )
        fig_mut.savefig(output_dir / "replicator_mutation.png", dpi=150)
        print(f"Saved mutation plot to {output_dir / 'replicator_mutation.png'}")

    # 3c. Moran process (finite-population stochastic robustness check)
    moran_summary = None
    if n_moran_runs > 0:
        print(f"\nRunning {n_moran_runs} Moran processes "
              f"(pop size {moran_population_size})...")
        # Distribute population uniformly across strategies
        per_strategy = moran_population_size // n
        remainder = moran_population_size - per_strategy * n
        initial_pop = {name: per_strategy for name in result.strategy_names}
        # Add remainder to first strategies for total = moran_population_size
        for i in range(remainder):
            initial_pop[result.strategy_names[i]] += 1

        moran_result = simulate_moran_process(
            result.payoff_matrix,
            result.strategy_names,
            initial_population=initial_pop,
            n_runs=n_moran_runs,
            seed=seed,
        )
        print("Moran process fixation probabilities:")
        for name, prob in sorted(
            moran_result.fixation_probabilities.items(), key=lambda kv: -kv[1]
        ):
            print(f"  {name:<24} {prob:.3f}")
        moran_summary = moran_result.fixation_probabilities

        with open(output_dir / "moran_results.json", "w") as f:
            json.dump(moran_result.fixation_probabilities, f, indent=2)

    # 4. ESS invasion tests for each strategy
    print("\nRunning ESS invasion tests...")
    ess_summary = {}
    for incumbent in result.strategy_names:
        ess_results = test_ess_against_all(
            result.payoff_matrix,
            result.strategy_names,
            incumbent=incumbent,
            invader_share=0.05,
        )
        n_resisted = sum(1 for r in ess_results.values() if r.is_stable)
        n_total = len(ess_results)
        ess_summary[incumbent] = {
            "n_resisted": n_resisted,
            "n_total": n_total,
            "is_global_ess": n_resisted == n_total,
            "details": {
                inv: {
                    "is_stable": r.is_stable,
                    "final_invader_share": r.final_invader_share,
                }
                for inv, r in ess_results.items()
            },
        }
        ess_label = "GLOBAL ESS" if n_resisted == n_total else f"{n_resisted}/{n_total} invaders resisted"
        print(f"  {incumbent:<24} {ess_label}")

    with open(output_dir / "ess_results.json", "w") as f:
        json.dump(ess_summary, f, indent=2)

    # 5. Done
    summary = {
        "label": label,
        "strategy_names": result.strategy_names,
        "ranking": [(n, s) for n, s in result.ranked_strategies()],
        "final_distribution": trajectory.final_distribution,
        "ess_summary": {k: v["is_global_ess"] for k, v in ess_summary.items()},
        "output_dir": str(output_dir),
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nExperiment complete. All artifacts in {output_dir}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run IPD evolutionary game theory experiment")
    parser.add_argument(
        "--mode",
        choices=["classical", "mock_llm", "mixed_mock", "real_llm"],
        default="classical",
        help="Which set of strategies to run",
    )
    parser.add_argument("--termination-prob", type=float, default=0.10)
    parser.add_argument("--max-rounds", type=int, default=100)
    parser.add_argument("--n-replicates", type=int, default=5)
    parser.add_argument("--n-generations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="data/output")
    parser.add_argument(
        "--backend",
        choices=["axelrod", "custom"],
        default="axelrod",
        help="Tournament backend implementation",
    )
    parser.add_argument(
        "--mutation-rate",
        type=float,
        default=0.01,
        help="Mutation rate for robustness check. Set 0 to skip.",
    )
    parser.add_argument(
        "--n-moran-runs",
        type=int,
        default=30,
        help="Number of Moran process runs. Set 0 to skip.",
    )
    parser.add_argument(
        "--moran-pop-size",
        type=int,
        default=50,
        help="Population size for Moran process",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.mode == "classical":
        factories = get_classical_factories()
        label = "classical_only"
    elif args.mode == "mock_llm":
        factories = get_mock_llm_factories()
        label = "mock_llm_only"
    elif args.mode == "mixed_mock":
        factories = {**get_classical_factories(), **get_mock_llm_factories()}
        label = "mixed_mock"
    elif args.mode == "real_llm":
        factories = {**get_classical_factories(), **get_real_llm_factories()}
        label = "real_llm"
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    output_dir = Path(args.output) / label
    summary = run_experiment(
        factories,
        output_dir,
        termination_prob=args.termination_prob,
        max_rounds=args.max_rounds,
        n_replicates=args.n_replicates,
        n_generations=args.n_generations,
        seed=args.seed,
        label=label,
        backend=args.backend,
        mutation_rate=args.mutation_rate,
        n_moran_runs=args.n_moran_runs,
        moran_population_size=args.moran_pop_size,
    )

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
