"""Visualization utilities for the chess tutor project.

All plotting helpers apply :func:`plot_style.apply_journal_style` on entry
and draw from the fixed palette in :mod:`chess_tutor.utils.plot_style`, so
every figure in the project — whether produced by a script, a notebook, or
a downstream caller — shares the same look.
"""

import numpy as np
import matplotlib.pyplot as plt

from .plot_style import (
    apply_journal_style,
    policy_color,
    bracket_color,
    PRIMARY,
    ACCENT,
    CHANCE,
    GRAY,
    GRAY_DARK,
    BLUE_DARK,
    BLUE,
    BLUE_MID,
)

try:
    import chess.svg
    from IPython.display import SVG, display
    HAS_SVG = True
except ImportError:
    HAS_SVG = False


def render_board(board, last_move=None, arrows=None, size=400):
    """Render board with optional highlights."""
    if not HAS_SVG:
        print(board)
        return None
    kwargs = {"size": size}
    if last_move:
        kwargs["lastmove"] = last_move
    if arrows:
        kwargs["arrows"] = arrows
    return SVG(chess.svg.board(board, **kwargs))


def plot_cross_elo_matrix(matrix, elo_brackets, save_path=None):
    """Plot the cross-ELO accuracy heatmap (Blues cmap)."""
    apply_journal_style()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(elo_brackets)))
    ax.set_xticklabels(elo_brackets)
    ax.set_yticks(range(len(elo_brackets)))
    ax.set_yticklabels(elo_brackets)
    ax.set_xlabel("Test ELO bracket")
    ax.set_ylabel("Model trained on ELO bracket")
    ax.set_title("Cross-ELO move matching (Top-1)")
    ax.grid(False)

    vmax = matrix.max()
    for i in range(len(elo_brackets)):
        for j in range(len(elo_brackets)):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center",
                    color="white" if matrix[i, j] > vmax * 0.6 else "black",
                    fontsize=9)

    fig.colorbar(im, ax=ax, label="Top-1 accuracy")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


def plot_feature_importance(importances: dict[str, float], top_n: int = 15,
                            save_path=None):
    """Plot feature-importance bar chart (single primary colour)."""
    apply_journal_style()
    sorted_feats = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names, values = zip(*sorted_feats)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.barh(range(len(names)), values, color=PRIMARY, edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Gini importance")
    ax.set_title(f"Top {top_n} feature importances")
    ax.grid(axis="x", alpha=0.25, linestyle="--", linewidth=0.4)
    ax.grid(axis="y", alpha=0)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


def plot_kernel_weights(elo_brackets, bandwidths, save_path=None):
    """Plot Gaussian kernel weights for different bandwidths (bracket-coloured)."""
    apply_journal_style()
    from ..models.kernel_interpolation import NadarayaWatsonELO

    query_elos = np.linspace(900, 2100, 300)
    fig, axes = plt.subplots(1, len(bandwidths),
                              figsize=(4 * len(bandwidths), 3.2),
                              sharey=True)
    if len(bandwidths) == 1:
        axes = [axes]

    for ax, bw in zip(axes, bandwidths):
        nw = NadarayaWatsonELO(bandwidth=bw)
        for center in elo_brackets:
            weights = [nw.kernel(q, center) for q in query_elos]
            ax.plot(query_elos, weights, label=f"{center}",
                    color=bracket_color(center))
        ax.set_title(f"Bandwidth = {bw}")
        ax.set_xlabel("Query ELO")

    axes[0].set_ylabel("Kernel weight")
    axes[0].legend(title="Bracket", loc="upper right", fontsize=7,
                   title_fontsize=8)
    fig.suptitle("Gaussian kernel weights by bandwidth", y=1.04, fontsize=12)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


def plot_regret_curves(results: dict, save_path=None):
    """Plot cumulative regret curves for all policies (± 2 SEM band)."""
    apply_journal_style()
    fig, ax = plt.subplots(figsize=(9, 5))

    for name, res in results.items():
        curves = res["regret_curves"]
        mean_curve = curves.mean(axis=0)
        sem_curve = curves.std(axis=0) / np.sqrt(curves.shape[0])
        x = np.arange(len(mean_curve))
        c = policy_color(name)
        ax.plot(x, mean_curve, label=name, color=c)
        ax.fill_between(x, mean_curve - 2 * sem_curve,
                        mean_curve + 2 * sem_curve,
                        alpha=0.15, color=c, linewidth=0)

    ax.set_xlabel("Interaction")
    ax.set_ylabel("Cumulative regret")
    ax.set_title("Cumulative regret by policy (mean ± 2 SEM)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


def plot_arm_distribution(results: dict, feedback_names=None, save_path=None):
    """Plot arm-selection distribution per policy (policy-coloured)."""
    apply_journal_style()
    from ..config import FEEDBACK_TYPE_NAMES
    names = feedback_names or FEEDBACK_TYPE_NAMES

    policies = list(results.keys())
    n_arms = len(names)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(n_arms)
    width = 0.8 / len(policies)

    for i, policy in enumerate(policies):
        dist = np.asarray(results[policy]["arm_distribution"], dtype=float)
        dist = dist / dist.sum() if dist.sum() > 0 else dist
        ax.bar(x + i * width, dist[:n_arms], width, label=policy,
               color=policy_color(policy), edgecolor="black", linewidth=0.3)

    ax.set_xticks(x + width * (len(policies) - 1) / 2)
    ax.set_xticklabels(names, rotation=18, ha="right", fontsize=9)
    ax.set_ylabel("Selection frequency")
    ax.set_title("Feedback-type selection by policy")
    ax.legend(loc="upper right", ncol=2)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


def plot_elo_trajectories(results: dict, save_path=None):
    """Plot mean-ELO-gain bar chart across policies (policy-coloured)."""
    apply_journal_style()
    fig, ax = plt.subplots(figsize=(8, 4.5))

    names = list(results.keys())
    gains = [results[n].get("mean_elo_gain", 0) for n in names]
    stds  = [results[n].get("std_elo_gain", 0) for n in names]
    colors = [policy_color(n) for n in names]
    ax.bar(range(len(names)), gains, yerr=stds, color=colors,
           capsize=3, edgecolor="black", linewidth=0.5,
           error_kw={"lw": 0.8})
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=18, ha="right")
    ax.set_ylabel("Mean ELO gain ± std")
    ax.set_title("Teaching effectiveness: ELO gain by policy")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


def plot_roc_curve(y_true, y_scores, save_path=None):
    """Plot ROC curve with AUC annotated (single primary colour)."""
    apply_journal_style()
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color=PRIMARY, lw=1.7,
            label=f"ROC (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color=CHANCE, linestyle="--", lw=0.7,
            label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Blunder detection ROC")
    ax.legend(loc="lower right")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


def plot_bandwidth_cv(bandwidths, accuracies, best_bw=None, save_path=None):
    """Plot accuracy vs bandwidth (single primary line + accent marker)."""
    apply_journal_style()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(bandwidths, accuracies, "o-", color=PRIMARY,
            markersize=6, markerfacecolor="white", markeredgewidth=1.3)
    if best_bw is not None:
        ax.axvline(x=best_bw, color=ACCENT, linestyle="--", lw=0.8,
                   label=f"best = {best_bw}")
        ax.legend()
    ax.set_xlabel("Bandwidth (ELO points)")
    ax.set_ylabel("Top-1 accuracy")
    ax.set_title("Kernel bandwidth selection")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig
