"""Continuous-ELO visualization: Architecture C's defining advantage.

Plots P(move | board, ELO) as a smooth function of ELO from 900 to 2100,
for a fixed board, contrasting Architecture A (per-bracket nearest-neighbor
lookup, step function) against Architecture C (kernel-interpolated, smooth
curve). Architecture B is excluded here because its ELO-feature usage
doesn't naturally map to the same "per-move probability vs. ELO" plot.

Output: results/plots/continuous_elo_interpolation.png

Goal of this visualization: show a concrete artifact that Architecture C
can produce and Architecture A cannot — continuous, smooth probability
curves across the full ELO range, including values (e.g. ELO 1640, 1750)
that fall between training brackets.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("MPLCONFIGDIR", os.path.abspath("results/.matplotlib"))

import pickle
import numpy as np
import pandas as pd
import chess
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from chess_tutor.config import ELO_BRACKETS, ELO_BRACKET_WIDTH
from chess_tutor.models.kernel_interpolation import NadarayaWatsonELO
from chess_tutor.utils.plot_style import apply_journal_style, PRIMARY, GRAY

apply_journal_style()


def load_models():
    with open("models/saved/models_a_candidate.pkl", "rb") as f:
        return pickle.load(f)


def build_query_row(board: chess.Board, move: chess.Move) -> np.ndarray:
    """Build the 40-dim candidate feature row for a specific (board, move) pair."""
    from chess_tutor.data.extract_features import extract_all_features
    return extract_all_features(board, move).reshape(1, -1)


def architecture_a_prob_vs_elo(models, query_row, elos):
    """Architecture A: nearest-bracket lookup — step function."""
    probs = []
    for e in elos:
        nearest = min(models.keys(), key=lambda b: abs(b - e))
        p = models[nearest].predict_proba(query_row)[0, 1]  # P(played | row)
        probs.append(p)
    return np.array(probs)


def architecture_c_prob_vs_elo(models, query_row, elos, bandwidth=100):
    """Architecture C: kernel-interpolated — smooth function."""
    nw = NadarayaWatsonELO(bandwidth=bandwidth)
    centers = sorted(models.keys())
    # Precompute per-bracket prob for this row
    per_bracket = {b: models[b].predict_proba(query_row)[0, 1] for b in centers}

    probs = []
    for e in elos:
        weights = np.array([nw.kernel(e, c) for c in centers])
        weights /= weights.sum()
        p = sum(w * per_bracket[c] for w, c in zip(weights, centers))
        probs.append(p)
    return np.array(probs)


def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/.matplotlib", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)
    models = load_models()

    # Choose a well-known tactical position (Italian game, after 3.Bc4)
    # Test moves that differentiate by skill level
    board = chess.Board(
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
    )

    moves_to_plot = [
        ("Nc3", chess.Move.from_uci("b1c3")),     # solid developing move
        ("d3", chess.Move.from_uci("d2d3")),      # quiet, solid
        ("Ng5", chess.Move.from_uci("f3g5")),     # aggressive tactical
        ("O-O", chess.Move.from_uci("e1g1")),     # castle
    ]

    # Filter to moves that are actually legal in this position
    legal_moves_to_plot = [(n, m) for n, m in moves_to_plot if m in board.legal_moves]
    if not legal_moves_to_plot:
        # Fallback: use the top-3 moves by bracket-1500 model
        legal = list(board.legal_moves)
        rows = np.vstack([build_query_row(board, m)[0] for m in legal])
        probs_1500 = models[1500].predict_proba(rows)[:, 1]
        top_idx = np.argsort(-probs_1500)[:4]
        legal_moves_to_plot = [(board.san(legal[i]), legal[i]) for i in top_idx]

    # Sweep ELOs, including both bracket centers and interstitial values
    elos = np.linspace(900, 2100, 241)  # step of 5
    bandwidth = 100  # default; per-bandwidth comparison lives in kernel_weights.png

    # Small-multiples grid: one panel per move, A (step) vs C (smooth)
    n = len(legal_moves_to_plot)
    n_cols = 2
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 3 * n_rows),
        sharex=True, sharey=True, squeeze=False,
    )

    for idx, (name, move) in enumerate(legal_moves_to_plot):
        r, c = divmod(idx, n_cols)
        ax = axes[r, c]
        query = build_query_row(board, move)
        probs_a = architecture_a_prob_vs_elo(models, query, elos)
        probs_c = architecture_c_prob_vs_elo(models, query, elos, bandwidth=bandwidth)

        ax.plot(elos, probs_a, label="Arch A (per-bracket)",
                color=GRAY, linestyle="--", lw=1.3, drawstyle="steps-mid")
        ax.plot(elos, probs_c, label=f"Arch C (NW, bw={bandwidth})",
                color=PRIMARY, lw=1.7)
        # Mark bracket centers
        for b in ELO_BRACKETS:
            ax.axvline(b, color=GRAY, linestyle=":", alpha=0.35, lw=0.6)

        ax.set_xlim(900, 2100)
        ax.set_title(f"Move: {name}")
        if r == n_rows - 1:
            ax.set_xlabel("Target ELO")
        if c == 0:
            ax.set_ylabel(r"$P(\mathrm{move} \mid \mathrm{ELO})$")

    # Hide any unused axes in the last row
    for idx in range(n, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r, c].axis("off")

    # Single legend (in the first panel)
    axes[0, 0].legend(loc="best", fontsize=8)
    fig.suptitle(
        "Continuous ELO interpolation — Arch A vs Arch C",
        y=1.00, fontsize=12,
    )
    plt.tight_layout()
    out = "results/plots/continuous_elo_interpolation.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved: {out}")

    # Also save the raw probability curves
    data = {"elo": elos}
    for name, move in legal_moves_to_plot:
        q = build_query_row(board, move)
        data[f"A_{name}"] = architecture_a_prob_vs_elo(models, q, elos)
        data[f"C_bw100_{name}"] = architecture_c_prob_vs_elo(models, q, elos, 100)
    pd.DataFrame(data).to_csv("results/continuous_elo_curves.csv", index=False)
    print("Saved: results/continuous_elo_curves.csv")


if __name__ == "__main__":
    main()
