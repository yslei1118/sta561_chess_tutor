"""Ablation study suite for systematic evaluation."""

import numpy as np
import pandas as pd

from ..config import ELO_BRACKETS, KERNEL_BANDWIDTH_CANDIDATES


def run_ablation_suite(dataset, model_configs: list[dict] | None = None) -> pd.DataFrame:
    """Run all ablation experiments.

    Returns:
        DataFrame with columns: experiment, config, metric, value
    """
    from ..models.move_predictor import MovePredictor

    results = []

    X, y = dataset.get_features_and_labels()
    elos = dataset.positions["player_elo"].values

    # Split
    n = len(X)
    rng = np.random.RandomState(42)
    idx = rng.permutation(n)
    split = int(n * 0.8)
    train_idx, test_idx = idx[:split], idx[split:]
    X_train, y_train, elos_train = X[train_idx], y[train_idx], elos[train_idx]
    X_test, y_test, elos_test = X[test_idx], y[test_idx], elos[test_idx]

    # 1. Architecture comparison
    for arch in ["A", "B", "C"]:
        for clf in ["rf", "logistic"]:
            try:
                model = MovePredictor(architecture=arch, base_classifier=clf)
                model.fit(X_train, y_train, elos_train)
                scores = model.score(X_test, y_test, elos_test)
                results.append({
                    "experiment": "architecture",
                    "config": f"{arch}_{clf}",
                    "metric": "accuracy_top1",
                    "value": scores["accuracy_top1"],
                })
            except Exception as e:
                results.append({
                    "experiment": "architecture",
                    "config": f"{arch}_{clf}",
                    "metric": "accuracy_top1",
                    "value": float("nan"),
                })

    # 2. Feature subsets
    feature_subsets = {
        "material_only": slice(0, 12),
        "+mobility": slice(0, 14),
        "+king_safety": slice(0, 16),
        "+pawn_structure": slice(0, 21),
        "full_board": slice(0, 30),
    }
    for name, slc in feature_subsets.items():
        try:
            model = MovePredictor(architecture="A", base_classifier="rf")
            model.fit(X_train[:, slc], y_train, elos_train)
            # Score with same subset
            proba_preds = []
            for i in range(len(X_test)):
                p = model.predict_proba(X_test[i:i+1, slc], int(elos_test[i]))
                proba_preds.append(p.argmax(axis=1)[0])
            acc = float((np.array(proba_preds) == y_test).mean())
            results.append({
                "experiment": "feature_subset",
                "config": name,
                "metric": "accuracy_top1",
                "value": acc,
            })
        except Exception:
            pass

    # 3. Data size
    for frac in [0.1, 0.25, 0.5, 1.0]:
        n_sub = int(len(X_train) * frac)
        try:
            model = MovePredictor(architecture="A", base_classifier="rf")
            model.fit(X_train[:n_sub], y_train[:n_sub], elos_train[:n_sub])
            scores = model.score(X_test, y_test, elos_test)
            results.append({
                "experiment": "data_size",
                "config": f"{n_sub}",
                "metric": "accuracy_top1",
                "value": scores["accuracy_top1"],
            })
        except Exception:
            pass

    # 4. Bandwidth
    for bw in KERNEL_BANDWIDTH_CANDIDATES:
        try:
            model = MovePredictor(architecture="C", base_classifier="rf",
                                  kernel_bandwidth=bw)
            model.fit(X_train, y_train, elos_train)
            scores = model.score(X_test, y_test, elos_test)
            results.append({
                "experiment": "bandwidth",
                "config": f"bw={bw}",
                "metric": "accuracy_top1",
                "value": scores["accuracy_top1"],
            })
        except Exception:
            pass

    return pd.DataFrame(results)
