"""
Plotting utilities for visualizing tournament and evolution results.

All functions return matplotlib Figure objects so they can be saved or
embedded into the writeup.
"""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from evolution.replicator import ReplicatorTrajectory
from fingerprints.metrics import Fingerprint
from tournament.tournament import TournamentResult


# Consistent color palette
_COLORS = plt.cm.tab10.colors


def plot_payoff_heatmap(
    result: TournamentResult,
    title: str = "Pairwise payoff matrix (avg payoff per round)",
    figsize: tuple[float, float] = (8, 6),
) -> plt.Figure:
    """Plot the tournament payoff matrix as a heatmap."""
    fig, ax = plt.subplots(figsize=figsize)
    M = result.payoff_matrix
    im = ax.imshow(M, cmap="RdYlGn", aspect="auto", vmin=0, vmax=5)

    ax.set_xticks(range(len(result.strategy_names)))
    ax.set_yticks(range(len(result.strategy_names)))
    ax.set_xticklabels(result.strategy_names, rotation=45, ha="right")
    ax.set_yticklabels(result.strategy_names)
    ax.set_xlabel("Opponent strategy")
    ax.set_ylabel("Row strategy")
    ax.set_title(title)

    # Annotate cells
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(
                j, i, f"{M[i, j]:.2f}",
                ha="center", va="center",
                color="black", fontsize=9,
            )

    plt.colorbar(im, ax=ax, label="Avg payoff per round")
    plt.tight_layout()
    return fig


def plot_population_trajectory(
    trajectory: ReplicatorTrajectory,
    title: str = "Replicator dynamics — population shares over time",
    figsize: tuple[float, float] = (10, 6),
    log_scale: bool = False,
) -> plt.Figure:
    """Plot population shares over generations as line plot."""
    fig, ax = plt.subplots(figsize=figsize)

    n_gens = trajectory.final_generation + 1
    gens = np.arange(n_gens)

    for i, name in enumerate(trajectory.strategy_names):
        ax.plot(
            gens,
            trajectory.history[:n_gens, i],
            label=name,
            color=_COLORS[i % len(_COLORS)],
            linewidth=2,
        )

    ax.set_xlabel("Generation")
    ax.set_ylabel("Population share")
    ax.set_title(title)
    if log_scale:
        ax.set_yscale("log")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_fingerprint_radar(
    fingerprints: dict[str, Fingerprint],
    title: str = "Behavioral fingerprints",
    figsize: tuple[float, float] = (10, 10),
) -> plt.Figure:
    """
    Plot multiple strategies' fingerprints as overlaid radar charts.

    Each axis is one of the core metrics. -1.0 (n/a) values are mapped to 0
    for plotting only; this should be noted in any paper using these.
    """
    metrics = [
        "cooperation_rate",
        "first_move_coop",
        "retaliation_rate",
        "forgiveness_rate",
        "opportunism_rate",
        "good_partner_rate",
    ]
    n_metrics = len(metrics)

    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "polar"})

    for i, (name, fp) in enumerate(fingerprints.items()):
        values = []
        for metric in metrics:
            v = getattr(fp, metric)
            values.append(max(0.0, v))  # Clip n/a sentinels to 0
        values += values[:1]
        ax.plot(angles, values, label=name, color=_COLORS[i % len(_COLORS)], linewidth=2)
        ax.fill(angles, values, color=_COLORS[i % len(_COLORS)], alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(
        [m.replace("_", " ").title() for m in metrics],
        fontsize=10,
    )
    ax.set_ylim(0, 1)
    ax.set_title(title, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.1))
    plt.tight_layout()
    return fig


def plot_fingerprint_table(
    fingerprints: dict[str, Fingerprint],
    title: str = "Fingerprint metrics by strategy",
    figsize: tuple[float, float] = (12, None),
) -> plt.Figure:
    """Plot fingerprint metrics as a heatmap-style table."""
    metrics = [
        "cooperation_rate",
        "first_move_coop",
        "retaliation_rate",
        "forgiveness_rate",
        "opportunism_rate",
        "good_partner_rate",
    ]
    names = list(fingerprints.keys())

    data = np.zeros((len(names), len(metrics)))
    for i, name in enumerate(names):
        for j, metric in enumerate(metrics):
            v = getattr(fingerprints[name], metric)
            data[i, j] = v if v >= 0 else np.nan

    if figsize[1] is None:
        figsize = (figsize[0], 0.4 * len(names) + 2)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(
        [m.replace("_", " ").title() for m in metrics],
        rotation=30, ha="right",
    )
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_title(title)

    for i in range(len(names)):
        for j in range(len(metrics)):
            v = data[i, j]
            label = "n/a" if np.isnan(v) else f"{v:.2f}"
            ax.text(j, i, label, ha="center", va="center", color="black", fontsize=9)

    plt.colorbar(im, ax=ax, label="Metric value")
    plt.tight_layout()
    return fig
