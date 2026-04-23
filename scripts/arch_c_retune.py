"""Retune Architecture C with larger forests and wider bandwidths.

Checks whether Architecture C — whose apparent loss to Architecture B in
``results/ablation_table.csv`` motivated the honest "methodological, not
accuracy" framing — can actually reach or beat B with more compute.

Tries:
  - n_estimators: {500 (original), 1000}
  - max_depth: {20, 30}
  - kernel bandwidths: {50, 100, 150, 200, 300, 500, 1000}

Writes: results/arch_c_retune.csv
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from chess_tutor.config import ELO_BRACKETS, ELO_BRACKET_WIDTH
from chess_tutor.models.kernel_interpolation import NadarayaWatsonELO


def evaluate_ranking(bracket_preds, y_test, pos_idx_test):
    """Top-1 ranking accuracy on candidate data."""
    correct = 0
    total = 0
    for pos in np.unique(pos_idx_test):
        pm = pos_idx_test == pos
        if y_test[pm].sum() == 0:
            continue
        total += 1
        if y_test[pm][np.argmax(bracket_preds[pm])] == 1:
            correct += 1
    return correct / total if total > 0 else 0.0


def main():
    os.makedirs("results", exist_ok=True)
    print("Loading data...")
    X = np.load("data/processed/candidate_X.npy")
    y = np.load("data/processed/candidate_y.npy")
    elos = np.load("data/processed/candidate_elos.npy")
    pos_idx = np.load("data/processed/candidate_pos_idx.npy")

    # 80/20 split by position (same seed as train_and_evaluate.py)
    rng = np.random.RandomState(42)
    uniq = np.arange(pos_idx.max() + 1)
    rng.shuffle(uniq)
    split = int(len(uniq) * 0.8)
    train_pos = set(uniq[:split])
    test_pos = set(uniq[split:])
    train_mask = np.array([p in train_pos for p in pos_idx])
    test_mask = ~train_mask

    X_train, y_train, elos_train = X[train_mask], y[train_mask], elos[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    elos_test, pos_idx_test = elos[test_mask], pos_idx[test_mask]

    print(f"Train: {X_train.shape[0]:,} rows ({len(train_pos):,} positions)")
    print(f"Test:  {X_test.shape[0]:,} rows ({len(test_pos):,} positions)")

    # Train per-bracket models at different settings
    rows = []
    for n_est, max_depth in [(500, 20), (1000, 20), (1000, 30)]:
        print(f"\nTraining Architecture A with n_estimators={n_est}, max_depth={max_depth}...")
        models = {}
        for bracket in ELO_BRACKETS:
            mask = np.abs(elos_train - bracket) <= ELO_BRACKET_WIDTH
            if mask.sum() < 10:
                continue
            clf = RandomForestClassifier(
                n_estimators=n_est,
                max_depth=max_depth,
                min_samples_leaf=5,
                n_jobs=-1,
                random_state=42,
            )
            clf.fit(X_train[mask], y_train[mask])
            models[bracket] = clf

        # Score Architecture A at each bracket
        for bracket in ELO_BRACKETS:
            mask = np.abs(elos_test - bracket) <= ELO_BRACKET_WIDTH
            if mask.sum() == 0:
                continue
            preds = models[bracket].predict_proba(X_test[mask])[:, 1]
            acc = evaluate_ranking(preds, y_test[mask], pos_idx_test[mask])
            rows.append({
                "architecture": "A (per-bracket)",
                "n_estimators": n_est,
                "max_depth": max_depth,
                "bandwidth": None,
                "test_bracket": bracket,
                "top1": acc,
            })

        # Score Architecture C at each bracket for each bandwidth
        for bw in [50, 100, 150, 200, 300, 500, 1000]:
            nw = NadarayaWatsonELO(bandwidth=bw)
            for test_bracket in ELO_BRACKETS:
                mask = np.abs(elos_test - test_bracket) <= ELO_BRACKET_WIDTH
                if mask.sum() == 0:
                    continue
                # Interpolate from all brackets
                centers = sorted(models.keys())
                weights = np.array([nw.kernel(test_bracket, c) for c in centers])
                weights /= weights.sum()
                combined = None
                for w, c in zip(weights, centers):
                    p = models[c].predict_proba(X_test[mask])[:, 1]
                    combined = w * p if combined is None else combined + w * p
                acc = evaluate_ranking(combined, y_test[mask], pos_idx_test[mask])
                rows.append({
                    "architecture": "C (kernel)",
                    "n_estimators": n_est,
                    "max_depth": max_depth,
                    "bandwidth": bw,
                    "test_bracket": test_bracket,
                    "top1": acc,
                })

        print(f"  ... done n={n_est}, d={max_depth}")

    df = pd.DataFrame(rows)
    df.to_csv("results/arch_c_retune.csv", index=False)

    print("\n" + "=" * 72)
    print("MEAN TOP-1 ACROSS 5 BRACKETS:")
    print("=" * 72)
    summary = df.groupby(["architecture", "n_estimators", "max_depth", "bandwidth"],
                        dropna=False)["top1"].mean().reset_index()
    summary = summary.sort_values("top1", ascending=False)
    print(summary.to_string(index=False))

    print(f"\nSaved: results/arch_c_retune.csv")


if __name__ == "__main__":
    main()
