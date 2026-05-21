# LLM Strategic Fingerprints in Iterated Prisoner's Dilemma

Final project for Algorithmic Game Theory (Spring 2026)
Robert Wang & Charlotte Roth

## Overview

This project investigates whether LLM agent strategies in evolutionary
Prisoner's Dilemma settings converge toward classical cooperative behaviors
or exhibit a novel, distinct set of strategic tendencies — and whether those
tendencies are evolutionarily stable.

Two primary contributions:

1. Treating prompt framing as a systematic experimental variable
2. Operationalizing qualitative "strategic fingerprints" into quantitative
   behavioral metrics

Backed by two well-tested academic libraries:

- **axelrod-python** — canonical implementations of classical IPD strategies
  and tournament infrastructure (Vincent Knight et al.)
- **nashpy** — replicator dynamics, mutation dynamics, Moran processes,
  and Nash equilibrium computation (Vincent Knight et al.)

Our novel contributions live in our own code (LLM strategies, prompt framings,
behavioral fingerprint metrics).

## Strategies Used

The default tournament backend uses axelrod-python's canonical classical
implementations, plus our own wrappers for LLM-backed strategies.

Classical strategies:

- `TitForTat` — cooperate first, then mirror the opponent's previous move
- `AlwaysDefect` — defect every round
- `AlwaysCooperate` — cooperate every round
- `Pavlov` — win-stay, lose-shift
- `GrimTrigger` — cooperate until the opponent defects once, then defect forever
- `GenerousTFT` — Tit-for-Tat with occasional forgiveness
- `Joss` — Tit-for-Tat with occasional random defections
- `SuspiciousTFT` — Tit-for-Tat that defects on the first move
- `Tester` — probes the opponent for exploitability

LLM strategies:

- `Claude_Neutral`
- `Claude_Rational`
- `Claude_Coop`
- `GPT_Neutral`
- `GPT_Rational`
- `GPT_Coop`

Mock LLM strategies for cost-free testing:

- `MockClaude_Neutral`
- `MockClaude_Rational`
- `MockClaude_Coop`
- `MockGPT_Neutral`
- `MockGPT_Rational`
- `MockGPT_Coop`

Run modes use these rosters:

- `classical` — the 9 classical strategies
- `mock_llm` — the 6 mock LLM strategies
- `mixed_mock` — classical strategies plus mock LLM strategies
- `real_llm` — classical strategies plus real Claude/GPT strategies

## Project Structure

```
ipd_project/
├── utils/
│   ├── game.py             # Core types: Action, GameHistory, payoffs
│   └── plotting.py         # Visualization (heatmaps, radar, trajectories)
├── strategies/
│   ├── base.py             # Strategy abstract base class
│   ├── classical.py        # TFT, AD, AC, Pavlov, Grim, GenerousTFT
│   └── llm_strategy.py     # LLM-backed strategy wrapper (NOVEL)
├── llm/
│   ├── prompts.py          # Three framing conditions (NOVEL)
│   ├── client.py           # LLMClient protocol + MockLLMClient
│   ├── anthropic_client.py # Real Anthropic backend
│   └── openai_client.py    # Real OpenAI backend
├── tournament/
│   ├── match.py            # Single iterated match between two strategies
│   └── tournament.py       # Round-robin tournament + payoff matrix
├── fingerprints/
│   └── metrics.py          # Behavioral fingerprint computation (NOVEL)
├── evolution/
│   └── replicator.py       # Hand-rolled replicator dynamics + ESS testing
├── integrations/
│   ├── axelrod_adapter.py  # Adapter: our Strategy <-> axelrod Player
│   ├── axelrod_tournament.py # Tournament backend using axelrod-python
│   └── nashpy_dynamics.py  # Replicator + mutation + Moran via nashpy
├── tests/
│   ├── test_classical.py   # Classical strategies + match engine
│   ├── test_fingerprints.py # Fingerprint metrics
│   ├── test_pipeline.py    # End-to-end pipeline (custom backend)
│   ├── test_llm_pipeline.py # LLM strategy integration (mock)
│   └── test_integrations.py # axelrod + nashpy integrations
├── run_experiment.py       # Main runner script
└── data/                   # Output (gitignored)
```

## Usage

### Install dependencies

```bash
pip install numpy matplotlib nashpy axelrod
# For real LLM runs:
pip install anthropic openai
```

### Run all tests

```bash
python -m tests.test_classical
python -m tests.test_fingerprints
python -m tests.test_pipeline
python -m tests.test_llm_pipeline
python -m tests.test_integrations
```

### Run experiments

```bash
# Classical strategies via axelrod-python — useful for validating against Axelrod
python -m run_experiment --mode classical

# Mock LLMs only — pipeline test without API costs
python -m run_experiment --mode mock_llm

# Mixed mock LLMs + classical strategies
python -m run_experiment --mode mixed_mock

# Full real experiment (requires ANTHROPIC_API_KEY and OPENAI_API_KEY)
python -m run_experiment --mode real_llm \
    --termination-prob 0.10 \
    --max-rounds 100 \
    --n-replicates 5

# Switch tournament backend (axelrod is default)
python -m run_experiment --mode classical --backend custom

# Disable optional analyses
python -m run_experiment --mode classical --mutation-rate 0 --n-moran-runs 0
```

### Output

Each run creates an output directory with:

- `payoff_matrix.json` and `payoff_matrix.png` — pairwise payoffs
- `fingerprints.json`, `fingerprints_table.png`, `fingerprints_radar.png`
- `replicator_trajectory.png` — population shares over generations (nashpy)
- `replicator_mutation.png` — robustness check with mutation
- `moran_results.json` — finite-population fixation probabilities
- `ess_results.json` — invasion test summary
- `summary.json` — top-level findings

## Methodology

Following Payne & Alloui-Cros (2025):

- Iterated Prisoner's Dilemma with shadow-of-the-future termination
- Standard payoffs: T=5, R=3, P=1, S=0
- Round cap at 100 to bound API costs
- Round-robin tournament with K replicates per pair
- Replicator dynamics on the resulting payoff matrix
- ESS invasion testing with 5% invader share

Robustness extensions (via nashpy):

- Replicator dynamics with mutation (default 1% rate)
- Moran process for finite-population stochastic dynamics
- Continuous-time replicator ODE (smoother trajectories than discrete update)
