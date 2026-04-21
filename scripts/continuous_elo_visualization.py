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

import pickle
import numpy as np
import pandas as pd
import chess
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from chess_tutor.config import ELO_BRACKETS, ELO_BRACKET_WIDTH
from chess_tutor.models.kernel_interpolation import NadarayaWatsonELO


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

    # Per-bandwidth comparison
    bandwidths = [50, 100, 300]

    fig, axes = plt.subplots(
        len(legal_moves_to_plot), len(bandwidths),
        figsize=(4.5 * len(bandwidths), 2.8 * len(legal_moves_to_plot)),
        sharex=True, squeeze=False,
    )

    for row_idx, (name, move) in enumerate(legal_moves_to_plot):
        query = build_query_row(board, move)
        probs_a = architecture_a_prob_vs_elo(models, query, elos)

        for col_idx, bw in enumerate(bandwidths):
            ax = axes[row_idx, col_idx]
            probs_c = architecture_c_prob_vs_elo(models, query, elos, bandwidth=bw)

            ax.plot(elos, probs_a, label="Arch A (nearest)", lw=1.5,
                    color="#e74c3c", alpha=0.7, drawstyle="steps-mid")
            ax.plot(elos, probs_c, label=f"Arch C (bw={bw})", lw=2.2,
                    color="#2ecc71")
            # Mark bracket centers
            for b in ELO_BRACKETS:
                ax.axvline(b, color="gray", linestyle=":", alpha=0.35, lw=0.8)

            ax.set_xlim(900, 2100)
            ax.set_ylabel(f"P({name} | board, ELO)" if col_idx == 0 else "")
            if row_idx == 0:
                ax.set_title(f"Kernel bandwidth h = {bw}", fontsize=11)
            if row_idx == len(legal_moves_to_plot) - 1:
                ax.set_xlabel("Target ELO")
            ax.grid(alpha=0.25)
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=9, loc="best")

    fig.suptitle(
        "Continuous-ELO move probability: Architecture C's defining advantage\n"
        "(Same board, four candidate moves; Arch A is a step function, Arch C is smooth)",
        fontsize=12, y=0.995,
    )
    plt.tight_layout()
    out = "results/plots/continuous_elo_interpolation.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
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
