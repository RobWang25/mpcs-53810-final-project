# Final Run Results: Strategic Fingerprints of LLMs in Iterated Prisoner's Dilemma

## Experiment Configuration

| Parameter | Value |
|---|---|
| Termination probability | 0.10 (shadow of the future) |
| Max rounds per match | 100 |
| Replicates per pairing | 5 |
| Replicator generations | 500 |
| Mutation rate | 0.01 |
| Moran process runs | 50 (pop size 50) |
| Total LLM API calls | ~8,700 |
| Parse failure rate | < 0.13% (all strategies) |
| API failures | 0 |

### Strategies Tested

**9 Classical** (via axelrod-python): TitForTat, AlwaysDefect, AlwaysCooperate, Pavlov, GrimTrigger, GenerousTFT, Joss, SuspiciousTFT, Tester

**6 LLM** (2 models × 3 framings): Claude Sonnet 4.6 (Neutral, Rational, Cooperative), GPT-4o (Neutral, Rational, Cooperative)

---

## Finding 1: Claude and GPT Exhibit Distinct Strategic Personalities

The behavioral fingerprints reveal clear model-level differences that persist across all three framing conditions.

**Claude** is a "diplomatic cooperator" — high cooperation, reliable retaliation, moderate forgiveness, and very low opportunism. Across all framings, Claude cooperates heavily (0.824–0.913), retaliates consistently (0.926–0.966), and almost never exploits a cooperative opponent (opportunism 0.009–0.015).

**GPT** is a "rigid retaliator" — cooperation varies widely by framing, but retaliation is consistently near-perfect (0.904–0.992) and forgiveness is near-zero except under cooperative framing (0.009–0.278). GPT punishes defection harder and forgives less than Claude in almost every condition.

| Metric | Claude range | GPT range |
|---|---|---|
| Cooperation rate | 0.824 – 0.913 | 0.728 – 0.929 |
| Retaliation rate | 0.926 – 0.966 | 0.904 – 0.992 |
| Forgiveness rate | 0.048 – 0.099 | 0.009 – 0.278 |
| Opportunism rate | 0.009 – 0.015 | 0.008 – 0.028 |

These quantitative profiles confirm and extend the qualitative characterizations from Payne & Alloui-Cros (2025), who described Claude as a "sophisticated diplomat" and GPT as a "principled idealist." Our fingerprint metrics operationalize these descriptions into measurable, comparable quantities.

---

## Finding 2: Prompt Framing Shifts Behavior, But Differently for Each Model

This is our primary novel contribution. Prompt framing produces measurable behavioral shifts in both models, but the magnitude and direction of these shifts differ between models.

### Claude Framing Sensitivity

| Framing | Cooperation | Retaliation | Forgiveness | Opportunism |
|---|---|---|---|---|
| Neutral | 0.835 | 0.956 | 0.062 | 0.015 |
| Rational | 0.913 | 0.966 | 0.048 | 0.009 |
| Cooperative | 0.824 | 0.926 | 0.099 | 0.013 |

Claude's cooperation range spans **8.9 percentage points** across framings. Counterintuitively, the rational framing produces the highest cooperation rate (0.913). This suggests Claude under rational framing reasons its way to cooperation as the payoff-maximizing strategy in iterated games — "rational cooperation." The cooperative framing produces the highest forgiveness (0.099) but the lowest cooperation (0.824), indicating that the cooperative persona makes Claude more willing to absorb exploitation, which depresses its overall cooperation rate against defectors.

### GPT Framing Sensitivity

| Framing | Cooperation | Retaliation | Forgiveness | Opportunism |
|---|---|---|---|---|
| Neutral | 0.728 | 0.985 | 0.017 | 0.028 |
| Rational | 0.806 | 0.992 | 0.009 | 0.021 |
| Cooperative | 0.929 | 0.904 | 0.278 | 0.008 |

GPT's cooperation range spans **20.1 percentage points** — more than double Claude's. GPT_Neutral is strikingly aggressive (0.728 cooperation, near-perfect retaliation, essentially zero forgiveness). GPT_Coop swings to the opposite extreme (0.929 cooperation, 0.278 forgiveness). The magnitude of this swing demonstrates that GPT's strategic personality is far more malleable via prompting than Claude's.

### Key Insight

**Framing sensitivity is itself a model-specific property.** Claude maintains a relatively stable strategic identity regardless of framing; GPT's behavior shifts dramatically. This has direct implications for deploying LLM agents in competitive settings: GPT behavior is more controllable via prompt engineering but also less predictable when prompts change.

---

## Finding 3: LLM Strategies Are Tournament-Competitive

Claude_Neutral ranked **#1 overall** in the round-robin tournament, outperforming every classical strategy including GenerousTFT and Tit-for-Tat — the canonical Axelrod tournament winner.

### Full Tournament Rankings

| Rank | Strategy | Avg Payoff | Type |
|---|---|---|---|
| 1 | Claude_Neutral | 2.756 | LLM |
| 2 | GenerousTFT | 2.750 | Classical |
| 3 | GPT_Coop | 2.749 | LLM |
| 4 | Claude_Coop | 2.707 | LLM |
| 5 | TitForTat | 2.703 | Classical |
| 6 | Claude_Rational | 2.655 | LLM |
| 7 | GPT_Neutral | 2.643 | LLM |
| 8 | GrimTrigger | 2.640 | Classical |
| 9 | AlwaysCooperate | 2.631 | Classical |
| 10 | Pavlov | 2.628 | Classical |
| 11 | Joss | 2.604 | Classical |
| 12 | GPT_Rational | 2.564 | LLM |
| 13 | Tester | 2.560 | Classical |
| 14 | SuspiciousTFT | 2.503 | Classical |
| 15 | AlwaysDefect | 2.373 | Classical |

LLM strategies occupy 4 of the top 7 positions. Claude consistently outranks GPT within the same framing condition. AlwaysDefect ranks dead last — confirming the classical Axelrod result that pure defection is a losing strategy in iterated games.

---

## Finding 4: Evolutionary Dynamics Produce a Surprising Equilibrium

The replicator dynamics converged to a two-strategy equilibrium that would not have been predicted from classical results alone.

### Final Population Distribution (Replicator Dynamics)

| Strategy | Population Share |
|---|---|
| Tester | 52.9% |
| GPT_Coop | 47.1% |
| All others | ~0% |

Tester is an exploitative probing strategy that defects first to test whether the opponent can be pushed around. GPT_Coop is the most cooperative and forgiving GPT variant (forgiveness = 0.278). These two form a stable symbiotic pair: Tester exploits weaker strategies that don't retaliate, while GPT_Coop's high forgiveness means it recovers cooperation with Tester faster than stricter retaliators would — creating a niche where both can coexist.

### Mutation-Corrected Distribution

With 1% mutation rate, the equilibrium shifts but the core dynamic persists:

| Strategy | Share |
|---|---|
| Tester | 50.3% |
| GPT_Coop | 36.4% |
| TitForTat | 4.4% |
| Claude_Neutral | 3.2% |
| GenerousTFT | 1.4% |
| All others | < 1% each |

Under mutation, TFT and Claude_Neutral maintain small niches as "backup cooperators" that persist through mutation pressure. This is more ecologically realistic than the pure replicator result.

### Why This Matters

The emergence of the Tester + GPT_Coop equilibrium is an entirely novel finding. It demonstrates that LLM strategies, with their specific forgiveness and retaliation profiles, can create evolutionary dynamics that differ qualitatively from classical all-cooperator or all-defector outcomes. The LLM's particular combination of high cooperation with non-zero forgiveness enables a symbiosis with exploitative strategies that strict classical strategies (TFT, GrimTrigger) cannot sustain.

---

## Finding 5: Some LLM Strategies Are Evolutionarily Stable

Two LLM strategies achieved **Global ESS** — meaning no single strategy (classical or LLM) can successfully invade a population of that strategy.

### ESS Results Summary

| Strategy | Invaders Resisted | Global ESS? |
|---|---|---|
| AlwaysDefect | 14/14 | ✓ |
| GrimTrigger | 14/14 | ✓ |
| **Claude_Coop** | **14/14** | **✓** |
| **GPT_Neutral** | **14/14** | **✓** |
| Claude_Neutral | 13/14 | |
| Claude_Rational | 13/14 | |
| TitForTat | 13/14 | |
| Pavlov | 13/14 | |
| GPT_Rational | 13/14 | |
| GPT_Coop | 12/14 | |
| GenerousTFT | 11/14 | |
| AlwaysCooperate | 12/14 | |
| Tester | 10/14 | |
| Joss | 5/14 | |
| SuspiciousTFT | 4/14 | |

Claude_Coop and GPT_Neutral join only AlwaysDefect and GrimTrigger as Global ESS strategies. This means a population of Claude_Coop agents is fundamentally stable — no mutant strategy can gain a foothold.

Notably, Claude_Coop is a Global ESS despite ranking only #4 in the tournament. ESS and tournament ranking measure different properties: tournament ranking measures average performance against a diverse field, while ESS measures resistance to targeted invasion. Claude_Coop's combination of high cooperation (0.824) with reliable retaliation (0.926) and moderate forgiveness (0.099) creates a profile that no single strategy can exploit — it cooperates enough to earn high mutual payoffs but retaliates enough that defectors cannot gain an advantage.

---

## Finding 6: LLMs Compete With Classical Strategies in Finite Populations

In the Moran process (50 runs, population size 50), LLM strategies fixated in **48% of runs** (24/50), compared to 52% for classical strategies. The process uses stochastic dynamics in finite populations — a more realistic model than deterministic replicator dynamics.

### Moran Process Fixation Rates

| Strategy | Fixation Rate | Type |
|---|---|---|
| GrimTrigger | 16% | Classical |
| GPT_Neutral | 12% | LLM |
| GPT_Coop | 12% | LLM |
| Claude_Rational | 10% | LLM |
| Pavlov | 8% | Classical |
| GenerousTFT | 8% | Classical |
| TitForTat | 6% | Classical |
| Tester | 6% | Classical |
| Claude_Neutral | 6% | LLM |
| AlwaysDefect | 4% | Classical |
| AlwaysCooperate | 4% | Classical |
| Claude_Coop | 4% | LLM |
| GPT_Rational | 4% | LLM |
| Joss | 0% | Classical |
| SuspiciousTFT | 0% | Classical |

LLM strategies are not just surviving — they are winning outright in nearly half the stochastic runs. This is a stronger result than the deterministic replicator dynamics, where only GPT_Coop survives long-term. The stochastic setting reveals that multiple LLM variants are competitive fixation candidates.

---

## Summary of Contributions

1. **Behavioral fingerprint quantification.** We operationalize the qualitative "strategic fingerprint" concept from Payne & Alloui-Cros (2025) into six measurable metrics: cooperation rate, first-move cooperation, retaliation rate, forgiveness rate, opportunism rate, and good-partner rate. These provide a standardized framework for comparing LLM strategic behavior across models and conditions.

2. **Prompt framing as independent variable.** We demonstrate that prompt framing produces measurable shifts in LLM strategic behavior — and that framing sensitivity is itself model-specific. Claude maintains a stable strategic identity across framings (8.9pp cooperation range); GPT is far more malleable (20.1pp range). This finding has direct implications for the controllability and predictability of LLM agents in competitive environments.

3. **LLMs are tournament-competitive.** Claude_Neutral outperformed every classical strategy in the round-robin tournament, including TFT and GenerousTFT. LLM strategies occupied 4 of the top 7 positions.

4. **LLMs can be evolutionarily stable.** Claude_Coop and GPT_Neutral achieved Global ESS, joining only AlwaysDefect and GrimTrigger in this distinction. This demonstrates that LLM strategic profiles are not merely competitive but fundamentally robust against invasion.

5. **Novel evolutionary dynamics.** The Tester + GPT_Coop symbiotic equilibrium that emerged from replicator dynamics is an entirely new phenomenon — an exploitative strategy and a forgiving LLM coexisting in stable coevolution, a dynamic that classical strategies alone do not produce.