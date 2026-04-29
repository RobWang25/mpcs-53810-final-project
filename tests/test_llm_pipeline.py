"""
Test that LLMStrategy integrates with the rest of the pipeline using
a mock backend (so it's free and reproducible).

Run with: python -m tests.test_llm_pipeline
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fingerprints.metrics import compute_fingerprint
import llm.client as llm_client
from llm.client import MockLLMClient
from llm.prompts import FramingCondition
from strategies.classical import AlwaysCooperate, AlwaysDefect, TitForTat
from strategies.llm_strategy import LLMStrategy
from tournament.match import MatchConfig
from tournament.tournament import (
    TournamentConfig,
    format_payoff_matrix,
    run_tournament,
)


def test_mock_llm_strategy_in_match() -> None:
    """A mock LLM with coop_prob=1.0 should behave like AlwaysCooperate."""
    from tournament.match import play_match

    client = MockLLMClient(coop_probability=1.0, seed=0)
    llm_strat = LLMStrategy(client, FramingCondition.NEUTRAL)

    config = MatchConfig(termination_prob=0.0, max_rounds=10)
    result = play_match(llm_strat, AlwaysDefect(), config)

    # All cooperative LLM should get sucker payoff every round
    assert result.coop_rate_a == 1.0
    assert result.payoff_a == 0
    print(f"Mock LLM (coop_prob=1.0) behaves like AC")


def test_mock_llm_in_tournament() -> None:
    """Run a small tournament with mock LLMs and classical strategies."""

    def make_llm_cooperative() -> LLMStrategy:
        client = MockLLMClient(coop_probability=0.85, seed=None, model_id="mock-coop")
        return LLMStrategy(client, FramingCondition.COOPERATIVE, custom_name="MockLLM_Coop")

    def make_llm_rational() -> LLMStrategy:
        client = MockLLMClient(coop_probability=0.45, seed=None, model_id="mock-rational")
        return LLMStrategy(client, FramingCondition.RATIONAL, custom_name="MockLLM_Rational")

    factories = {
        "TitForTat": TitForTat,
        "AlwaysDefect": AlwaysDefect,
        "AlwaysCooperate": AlwaysCooperate,
        "MockLLM_Coop": make_llm_cooperative,
        "MockLLM_Rational": make_llm_rational,
    }

    config = TournamentConfig(
        match_config=MatchConfig(termination_prob=0.0, max_rounds=30),
        n_replicates_per_pair=3,
        seed=42,
    )

    print("Running mock LLM tournament...")
    result = run_tournament(factories, config, verbose=False)

    print("\nPayoff matrix:")
    print(format_payoff_matrix(result))

    print("\nRanking:")
    for name, score in result.ranked_strategies():
        print(f"  {name:<22} {score:.3f}")

    # Compute fingerprints for each strategy
    print("\nFingerprints:")
    for name in result.strategy_names:
        fp = compute_fingerprint(result.histories_by_strategy[name])
        print(f"  {name:<22}: coop={fp.cooperation_rate:.3f} "
              f"first_C={fp.first_move_coop:.3f} "
              f"retaliation={fp.retaliation_rate:.3f} "
              f"opportunism={fp.opportunism_rate:.3f}")

    # MockLLM_Coop should have higher cooperation rate than MockLLM_Rational
    fp_coop = compute_fingerprint(result.histories_by_strategy["MockLLM_Coop"])
    fp_rational = compute_fingerprint(result.histories_by_strategy["MockLLM_Rational"])
    assert fp_coop.cooperation_rate > fp_rational.cooperation_rate
    print(
        f"\nCooperative mock LLM has higher coop rate "
        f"({fp_coop.cooperation_rate:.3f}) than rational mock "
        f"({fp_rational.cooperation_rate:.3f})"
    )


def test_project_env_loading() -> None:
    """Project .env values should be read directly from the .env file."""
    original_cwd = Path.cwd()
    original_file = llm_client.__file__

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            llm_dir = tmp_path / "llm"
            llm_dir.mkdir()
            (tmp_path / ".env").write_text(
                "OPENAI_API_KEY=test-openai-key\n"
                "ANTHROPIC_API_KEY=test-anthropic-key\n"
            )
            fake_client_file = llm_dir / "client.py"
            fake_client_file.write_text("# placeholder\n")

            llm_client.__file__ = str(fake_client_file)
            assert llm_client.get_env_value("OPENAI_API_KEY") == "test-openai-key"
            assert llm_client.get_env_value("ANTHROPIC_API_KEY") == "test-anthropic-key"
            print(".env loader populates API keys")
    finally:
        os.chdir(original_cwd)
        llm_client.__file__ = original_file


if __name__ == "__main__":
    test_mock_llm_strategy_in_match()
    test_mock_llm_in_tournament()
    test_project_env_loading()
    print("\nLLM pipeline tests passed.")
