"""Visualization utilities for the chess tutor project."""

import numpy as np
import matplotlib.pyplot as plt

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
    """Plot the cross-ELO accuracy heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(elo_brackets)))
    ax.set_xticklabels(elo_brackets)
    ax.set_yticks(range(len(elo_brackets)))
    ax.set_yticklabels(elo_brackets)
    ax.set_xlabel("Test ELO Bracket")
    ax.set_ylabel("Model Trained on ELO Bracket")
    ax.set_title("Cross-ELO Move Matching Accuracy")

    for i in range(len(elo_brackets)):
        for j in range(len(elo_brackets)):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max() * 0.7 else "black",
                    fontsize=9)

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


def plot_feature_importance(importances: dict[str, float], top_n: int = 15,
                            save_path=None):
    """Plot feature importance bar chart."""
    sorted_feats = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names, values = zip(*sorted_feats)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(names)), values, color="steelblue")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importances")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


def plot_kernel_weights(elo_brackets, bandwidths, save_path=None):
    """Plot Gaussian kernel weights for different bandwidths."""
    from ..models.kernel_interpolation import NadarayaWatsonELO

    query_elos = np.linspace(900, 2100, 200)
    fig, axes = plt.subplots(1, len(bandwidths), figsize=(4 * len(bandwidths), 4),
                              sharey=True)
    if len(bandwidths) == 1:
        axes = [axes]

    for ax, bw in zip(axes, bandwidths):
        nw = NadarayaWatsonELO(bandwidth=bw)
        for center in elo_brackets:
            weights = [nw.kernel(q, center) for q in query_elos]
            ax.plot(query_elos, weights, label=f"ELO {center}")
        ax.set_title(f"Bandwidth = {bw}")
        ax.set_xlabel("Query ELO")
        ax.legend(fontsize=7)

    axes[0].set_ylabel("Kernel Weight")
    fig.suptitle("Gaussian Kernel Weights by Bandwidth")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


def plot_regret_curves(results: dict, save_path=None):
    """Plot cumulative regret curves for all policies."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, res in results.items():
        curves = res["regret_curves"]
        mean_curve = curves.mean(axis=0)
        std_curve = curves.std(axis=0)
        x = np.arange(len(mean_curve))
        ax.plot(x, mean_curve, label=name)
        ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2)

    ax.set_xlabel("Interaction")
    ax.set_ylabel("Cumulative Regret")
    ax.set_title("Cumulative Regret by Policy")
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


def plot_arm_distribution(results: dict, feedback_names=None, save_path=None):
    """Plot arm selection distribution for each policy."""
    from ..config import FEEDBACK_TYPE_NAMES
    names = feedback_names or FEEDBACK_TYPE_NAMES

    policies = list(results.keys())
    n_arms = len(names)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(n_arms)
    width = 0.8 / len(policies)

    for i, policy in enumerate(policies):
        dist = results[policy]["arm_distribution"]
        dist = dist / dist.sum() if dist.sum() > 0 else dist
        ax.bar(x + i * width, dist[:n_arms], width, label=policy)

    ax.set_xticks(x + width * len(policies) / 2)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Selection Frequency")
    ax.set_title("Feedback Type Selection by Policy")
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


def plot_elo_trajectories(results: dict, save_path=None):
    """Plot ELO trajectories over teaching episodes."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, res in results.items():
        elo_gain = res.get("mean_elo_gain", 0)
        ax.bar(name, elo_gain)

    ax.set_ylabel("Mean ELO Gain")
    ax.set_title("Teaching Effectiveness: ELO Gain by Policy")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


def plot_roc_curve(y_true, y_scores, save_path=None):
    """Plot ROC curve with AUC annotated."""
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Blunder Detection ROC Curve")
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


def plot_bandwidth_cv(bandwidths, accuracies, best_bw=None, save_path=None):
    """Plot accuracy vs bandwidth from CV."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(bandwidths, accuracies, "o-", color="steelblue", lw=2, markersize=8)
    if best_bw is not None:
        ax.axvline(x=best_bw, color="red", linestyle="--", label=f"Best: {best_bw}")
        ax.legend()
    ax.set_xlabel("Bandwidth (ELO points)")
    ax.set_ylabel("Leave-One-Bracket-Out Accuracy")
    ax.set_title("Kernel Bandwidth Selection via CV")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig
