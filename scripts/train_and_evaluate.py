"""Train move predictors (A, B, C) using candidate-move binary classification.

For each position with N legal moves, we have N rows (features, label).
The model predicts P(human_played | board_features, move_features).
At eval time: rank moves by predicted probability, check if top-1/3/5 matches.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from chess_tutor.config import ELO_BRACKETS, ELO_BRACKET_WIDTH
import pickle

# Load data
X = np.load('data/processed/candidate_X.npy')
y = np.load('data/processed/candidate_y.npy')
elos = np.load('data/processed/candidate_elos.npy')
pos_idx = np.load('data/processed/candidate_pos_idx.npy')

print(f"Data: {X.shape[0]} rows, {X.shape[1]} features, {y.mean():.4f} positive rate")

n_positions = pos_idx.max() + 1

# Train/test split by position (not by row)
rng = np.random.RandomState(42)
all_pos = np.arange(n_positions)
rng.shuffle(all_pos)
split = int(n_positions * 0.8)
train_positions = set(all_pos[:split])
test_positions = set(all_pos[split:])

train_mask = np.array([p in train_positions for p in pos_idx])
test_mask = ~train_mask

X_train, y_train, elos_train, pidx_train = X[train_mask], y[train_mask], elos[train_mask], pos_idx[train_mask]
X_test, y_test, elos_test, pidx_test = X[test_mask], y[test_mask], elos[test_mask], pos_idx[test_mask]

print(f"Train: {X_train.shape[0]} rows ({len(train_positions)} positions)")
print(f"Test:  {X_test.shape[0]} rows ({len(test_positions)} positions)")


def evaluate_ranking(model, X_t, y_t, pidx_t, elos_t, target_elo, elo_tol=ELO_BRACKET_WIDTH):
    """Evaluate move ranking accuracy for a specific ELO bracket."""
    # Filter to target ELO
    elo_mask = np.abs(elos_t - target_elo) <= elo_tol
    if elo_mask.sum() == 0:
        return {'top1': 0, 'top3': 0, 'top5': 0, 'n_positions': 0}

    X_b, y_b, pidx_b = X_t[elo_mask], y_t[elo_mask], pidx_t[elo_mask]

    # Predict probabilities
    proba = model.predict_proba(X_b)[:, 1]  # P(human_played)

    # Group by position and check ranking
    unique_pos = np.unique(pidx_b)
    top1, top3, top5 = 0, 0, 0

    for p in unique_pos:
        mask = pidx_b == p
        p_proba = proba[mask]
        p_labels = y_b[mask]

        if p_labels.sum() == 0:
            continue

        # Rank by predicted probability
        ranked = np.argsort(p_proba)[::-1]
        true_idx = np.where(p_labels[ranked] == 1)[0]

        if len(true_idx) > 0:
            rank = true_idx[0]
            if rank < 1: top1 += 1
            if rank < 3: top3 += 1
            if rank < 5: top5 += 1

    n = len(unique_pos)
    return {
        'top1': top1 / n if n > 0 else 0,
        'top3': top3 / n if n > 0 else 0,
        'top5': top5 / n if n > 0 else 0,
        'n_positions': n,
    }


# ============================================================
# Architecture A: Per-bracket models
# ============================================================
print("\n" + "="*60)
print("Architecture A: Per-bracket RF")
print("="*60)

models_a = {}
for b in ELO_BRACKETS:
    elo_mask = np.abs(elos_train - b) <= ELO_BRACKET_WIDTH
    if elo_mask.sum() < 100:
        print(f"  Bracket {b}: too few samples, skipping")
        continue
    X_b, y_b = X_train[elo_mask], y_train[elo_mask]
    print(f"  Bracket {b}: training on {X_b.shape[0]} rows ({y_b.sum()} positives)...", flush=True)
    clf = RandomForestClassifier(n_estimators=200, max_depth=15, n_jobs=-1, random_state=42)
    clf.fit(X_b, y_b)
    models_a[b] = clf

    res = evaluate_ranking(clf, X_test, y_test, pidx_test, elos_test, b)
    print(f"    Top-1: {res['top1']:.4f}, Top-3: {res['top3']:.4f}, Top-5: {res['top5']:.4f} ({res['n_positions']} positions)")

# ============================================================
# Architecture B: Pooled model with ELO feature
# ============================================================
print("\n" + "="*60)
print("Architecture B: Pooled RF + ELO feature")
print("="*60)

# Add normalized ELO as feature
elo_feat_train = ((elos_train - 1100) / 800).reshape(-1, 1)
X_train_b = np.hstack([X_train, elo_feat_train])

print(f"  Training on {X_train_b.shape[0]} rows...", flush=True)
clf_b = RandomForestClassifier(n_estimators=200, max_depth=15, n_jobs=-1, random_state=42)
clf_b.fit(X_train_b, y_train)

elo_feat_test = ((elos_test - 1100) / 800).reshape(-1, 1)
X_test_b = np.hstack([X_test, elo_feat_test])

for b in ELO_BRACKETS:
    res = evaluate_ranking(clf_b, X_test_b, y_test, pidx_test, elos_test, b)
    print(f"  Bracket {b}: Top-1: {res['top1']:.4f}, Top-3: {res['top3']:.4f}, Top-5: {res['top5']:.4f}")

# ============================================================
# Architecture C: Kernel interpolation
# ============================================================
print("\n" + "="*60)
print("Architecture C: Kernel-smoothed (NW, bw=100)")
print("="*60)

from chess_tutor.models.kernel_interpolation import NadarayaWatsonELO

nw = NadarayaWatsonELO(bandwidth=100)

for target_elo in ELO_BRACKETS:
    # For each test position at target_elo, get predictions from each bracket model
    elo_mask = np.abs(elos_test - target_elo) <= ELO_BRACKET_WIDTH
    if elo_mask.sum() == 0:
        continue

    X_t = X_test[elo_mask]
    y_t = y_test[elo_mask]
    pidx_t = pidx_test[elo_mask]

    # Get predictions from each bracket model
    bracket_probas = {}
    for b, clf in models_a.items():
        bracket_probas[b] = clf.predict_proba(X_t)[:, 1]

    # Kernel-weighted combination
    weights = nw.kernel_weights(float(target_elo), list(models_a.keys()))
    combined_proba = np.zeros(len(X_t))
    for w, b in zip(weights, models_a.keys()):
        combined_proba += w * bracket_probas[b]

    # Evaluate ranking
    unique_pos = np.unique(pidx_t)
    top1, top3, top5 = 0, 0, 0
    for p in unique_pos:
        mask = pidx_t == p
        p_proba = combined_proba[mask]
        p_labels = y_t[mask]
        if p_labels.sum() == 0:
            continue
        ranked = np.argsort(p_proba)[::-1]
        true_idx = np.where(p_labels[ranked] == 1)[0]
        if len(true_idx) > 0:
            rank = true_idx[0]
            if rank < 1: top1 += 1
            if rank < 3: top3 += 1
            if rank < 5: top5 += 1

    n = len(unique_pos)
    print(f"  Bracket {target_elo}: Top-1: {top1/n:.4f}, Top-3: {top3/n:.4f}, Top-5: {top5/n:.4f} ({n} positions)")

# ============================================================
# Cross-ELO matrix (Arch A)
# ============================================================
print("\n" + "="*60)
print("Cross-ELO Matrix (Arch A)")
print("="*60)

brackets_list = sorted(models_a.keys())
matrix = np.zeros((len(brackets_list), len(brackets_list)))

for i, train_b in enumerate(brackets_list):
    for j, test_b in enumerate(brackets_list):
        res = evaluate_ranking(models_a[train_b], X_test, y_test, pidx_test, elos_test, test_b)
        matrix[i, j] = res['top1']

print("     " + "  ".join(f"{b:>6}" for b in brackets_list))
for i, b in enumerate(brackets_list):
    row_str = "  ".join(f"{matrix[i,j]:>6.4f}" for j in range(len(brackets_list)))
    print(f"{b}: {row_str}")

# Check diagonal dominance
print("\nDiagonal dominance check:")
for j in range(len(brackets_list)):
    diag = matrix[j, j]
    col_mean = matrix[:, j].mean()
    print(f"  {brackets_list[j]}: diag={diag:.4f} col_mean={col_mean:.4f} dominant={'✓' if diag >= col_mean else '✗'}")

# Feature importance
print("\n" + "="*60)
print("Feature Importance (bracket 1500)")
print("="*60)
from chess_tutor.config import BOARD_FEATURE_NAMES, MOVE_FEATURE_NAMES
names = BOARD_FEATURE_NAMES + MOVE_FEATURE_NAMES
if 1500 in models_a:
    imp = models_a[1500].feature_importances_
    top10 = sorted(zip(names, imp), key=lambda x: x[1], reverse=True)[:10]
    for name, val in top10:
        print(f"  {name}: {val:.4f}")

# Save models
os.makedirs('models/saved', exist_ok=True)
with open('models/saved/models_a_candidate.pkl', 'wb') as f:
    pickle.dump(models_a, f)
with open('models/saved/model_b_candidate.pkl', 'wb') as f:
    pickle.dump(clf_b, f)
print("\nModels saved.")
