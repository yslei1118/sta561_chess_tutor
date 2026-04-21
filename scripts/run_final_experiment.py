"""Final comprehensive experiment: larger scale, all plots, ablation table."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import pickle

from chess_tutor.config import (
    ELO_BRACKETS, ELO_BRACKET_WIDTH, N_CONTEXT_FEATURES, N_FEEDBACK_TYPES,
    FEEDBACK_TYPE_NAMES, BOARD_FEATURE_NAMES, MOVE_FEATURE_NAMES,
    KERNEL_BANDWIDTH_CANDIDATES,
)

os.makedirs('results/plots', exist_ok=True)

# ============================================================
# Large-scale bandit experiment (300 episodes, 30 interactions)
# ============================================================
print("=" * 60)
print("FINAL BANDIT EXPERIMENT (300 episodes × 30 interactions)")
print("=" * 60)

from chess_tutor.teaching.bandit import (
    LinearThompsonSampling, EpsilonGreedy, LinUCB, RandomPolicy, RuleBasedPolicy
)
from chess_tutor.simulation.student_simulator import StudentPopulation
from chess_tutor.simulation.runner import run_experiment

students = StudentPopulation.generate(n_students=50, random_state=42)

policies = {
    'Thompson Sampling': LinearThompsonSampling(n_arms=N_FEEDBACK_TYPES, context_dim=N_CONTEXT_FEATURES),
    'ε-Greedy (ε=0.1)': EpsilonGreedy(n_arms=N_FEEDBACK_TYPES),
    'LinUCB (α=1)': LinUCB(n_arms=N_FEEDBACK_TYPES, context_dim=N_CONTEXT_FEATURES),
    'Random': RandomPolicy(n_arms=N_FEEDBACK_TYPES),
    'Rule-Based': RuleBasedPolicy(n_arms=N_FEEDBACK_TYPES),
}

results = run_experiment(students, policies, n_episodes=1000, n_interactions_per_episode=50)

print(f"\n{'Policy':<25} {'Mean Reward':>12} {'Std':>8} {'ELO Gain':>10}")
print('-' * 60)
for name, res in results.items():
    print(f"{name:<25} {res['mean_cumulative_reward']:>12.3f} {res['std_cumulative_reward']:>8.3f} {res['mean_elo_gain']:>10.1f}")

# TS vs baselines
ts_r = results['Thompson Sampling']['mean_cumulative_reward']
for name in ['Random', 'ε-Greedy (ε=0.1)', 'Rule-Based']:
    other_r = results[name]['mean_cumulative_reward']
    impr = (ts_r - other_r) / other_r * 100 if other_r > 0 else 0
    print(f"  TS vs {name}: {impr:+.1f}%")

# Plot 6: Regret curves (with confidence bands)
fig, ax = plt.subplots(figsize=(10, 6))
colors = {'Thompson Sampling': '#2ecc71', 'ε-Greedy (ε=0.1)': '#3498db',
          'LinUCB (α=1)': '#9b59b6', 'Random': '#e74c3c', 'Rule-Based': '#f39c12'}
for name, res in results.items():
    curves = res['regret_curves']
    mean_curve = curves.mean(axis=0)
    std_curve = curves.std(axis=0) / np.sqrt(curves.shape[0])  # SEM
    x = np.arange(len(mean_curve))
    ax.plot(x, mean_curve, label=name, lw=2, color=colors.get(name, 'gray'))
    ax.fill_between(x, mean_curve - 2*std_curve, mean_curve + 2*std_curve, alpha=0.15,
                    color=colors.get(name, 'gray'))
ax.set_xlabel('Interaction', fontsize=12)
ax.set_ylabel('Cumulative Regret', fontsize=12)
ax.set_title('Cumulative Regret by Policy (1000 episodes, mean ± 2 SEM)', fontsize=13)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('results/plots/regret_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("Plot 6: Regret curves saved ✓")

# Sub-linear check
ts_regret = results['Thompson Sampling']['regret_curves'].mean(axis=0)
n = len(ts_regret)
first_half = ts_regret[n//2] / (n//2)
second_half = (ts_regret[-1] - ts_regret[n//2]) / (n - n//2)
print(f"  Sub-linear: first_half_rate={first_half:.4f}, second_half_rate={second_half:.4f} → {'✓' if second_half < first_half else '✗'}")

# Plot 8: Arm distribution
n_arms = len(FEEDBACK_TYPE_NAMES)
fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(n_arms)
width = 0.15
for i, (pname, res) in enumerate(results.items()):
    dist = res['arm_distribution']
    dist = dist / dist.sum() if dist.sum() > 0 else dist
    ax.bar(x + i * width, dist[:n_arms], width, label=pname, color=colors.get(pname, 'gray'))
ax.set_xticks(x + width * 2.5)
ax.set_xticklabels(FEEDBACK_TYPE_NAMES, rotation=45, ha='right', fontsize=10)
ax.set_ylabel('Selection Frequency', fontsize=12)
ax.set_title('Feedback Type Selection by Policy', fontsize=13)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('results/plots/arm_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("Plot 8: Arm distribution saved ✓")

# Plot 10: Teaching effectiveness
fig, ax = plt.subplots(figsize=(10, 6))
names_list = list(results.keys())
gains = [results[n]['mean_elo_gain'] for n in names_list]
stds = [results[n]['std_elo_gain'] for n in names_list]
bar_colors = [colors.get(n, 'gray') for n in names_list]
bars = ax.bar(range(len(names_list)), gains, yerr=stds, color=bar_colors,
              capsize=5, edgecolor='black', linewidth=0.5)
ax.set_xticks(range(len(names_list)))
ax.set_xticklabels(names_list, rotation=45, ha='right', fontsize=10)
ax.set_ylabel('Mean ELO Gain ± Std', fontsize=12)
ax.set_title('Teaching Effectiveness: ELO Gain by Policy', fontsize=13)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('results/plots/teaching_effectiveness.png', dpi=150, bbox_inches='tight')
plt.close()
print("Plot 10: Teaching effectiveness saved ✓")

# ============================================================
# Plot 7: ELO trajectories (per-policy simulation)
# ============================================================
import chess
from chess_tutor.simulation.student_simulator import StudentSimulator
from chess_tutor.teaching.context import build_context
from chess_tutor.student.model import StudentState
from chess_tutor.feedback import FeedbackType, FeedbackGenerator
import copy

fig, ax = plt.subplots(figsize=(10, 6))
n_steps = 300
fg = FeedbackGenerator()
positions = [chess.Board(fen) for fen in [
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
    "r1bqkb1r/pppppppp/2n5/4P3/8/5N2/PPPP1PPP/RNBQKB1R b KQkq - 0 3",
    "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N2N2/PP2BPPP/R1BQ1RK1 w - - 0 8",
    "8/8/4k3/8/8/4K3/4P3/8 w - - 0 1",
]]

for policy_name, color in [('Thompson Sampling', '#2ecc71'), ('LinUCB (α=1)', '#9b59b6'),
                             ('Random', '#e74c3c'), ('Rule-Based', '#f39c12')]:
    np.random.seed(42)
    student = StudentSimulator(elo=1200, learning_rate=0.03)
    policy = copy.deepcopy(policies[policy_name])
    elo_hist = [student.elo]

    for i in range(n_steps):
        board = positions[i % len(positions)].copy()
        from chess_tutor.data.extract_features import extract_board_features
        bf = extract_board_features(board)
        ctx = build_context(board, student.state, blunder_prob=0.3, complexity=0.5, board_features=bf)
        arm = policy.select_arm(ctx)
        student.respond_to_position(board, feedback_type=arm)
        cp = max(10, 80 - i * 0.15 + np.random.normal(0, 15))
        from chess_tutor.simulation.runner import _context_dependent_reward
        reward = _context_dependent_reward(ctx, arm, cp)
        policy.update(arm, ctx, reward)
        from chess_tutor.feedback.taxonomy import FEEDBACK_CONCEPT_MAP
        concepts = FEEDBACK_CONCEPT_MAP.get(FeedbackType(arm), [])
        student.update_state(arm, cp, concepts)
        elo_hist.append(student.elo)

    ax.plot(elo_hist, label=policy_name, lw=2, color=color)

ax.set_xlabel('Teaching Interaction', fontsize=12)
ax.set_ylabel('Student ELO', fontsize=12)
ax.set_title('Simulated Student ELO Trajectories by Policy', fontsize=13)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('results/plots/elo_trajectories.png', dpi=150, bbox_inches='tight')
plt.close()
print("Plot 7: ELO trajectories saved ✓")

# ============================================================
# Ablation Table (Plot 9)
# ============================================================
print("\n" + "=" * 60)
print("ABLATION: Architecture × Classifier")
print("=" * 60)

X_cand = np.load('data/processed/candidate_X.npy')
y_cand = np.load('data/processed/candidate_y.npy')
elos_cand = np.load('data/processed/candidate_elos.npy')
pos_idx_cand = np.load('data/processed/candidate_pos_idx.npy')

n_positions = pos_idx_cand.max() + 1
rng = np.random.RandomState(42)
all_pos = np.arange(n_positions)
rng.shuffle(all_pos)
split_pos = int(n_positions * 0.8)
train_positions = set(all_pos[:split_pos])
train_mask = np.array([p in train_positions for p in pos_idx_cand])

X_tr, y_tr, e_tr, p_tr = X_cand[train_mask], y_cand[train_mask], elos_cand[train_mask], pos_idx_cand[train_mask]
X_te, y_te, e_te, p_te = X_cand[~train_mask], y_cand[~train_mask], elos_cand[~train_mask], pos_idx_cand[~train_mask]

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

def train_and_eval_bracket(X_train, y_train, elos_train, X_test, y_test, elos_test, pos_test,
                           classifier_fn, bracket):
    mask_tr = np.abs(elos_train - bracket) <= ELO_BRACKET_WIDTH
    mask_te = np.abs(elos_test - bracket) <= ELO_BRACKET_WIDTH
    if mask_tr.sum() < 50 or mask_te.sum() < 50:
        return None

    clf = classifier_fn()
    clf.fit(X_train[mask_tr], y_train[mask_tr])
    proba = clf.predict_proba(X_test[mask_te])[:, 1]

    # Evaluate top-1
    pidx = pos_test[mask_te]
    y_b = y_test[mask_te]
    unique_pos = np.unique(pidx)
    top1 = 0
    for p in unique_pos:
        m = pidx == p
        ranked = np.argsort(proba[m])[::-1]
        if y_b[m][ranked[0]] == 1:
            top1 += 1
    return top1 / len(unique_pos)

ablation_results = []

# Architecture A × Classifiers
classifiers = {
    'RF': lambda: RandomForestClassifier(n_estimators=200, max_depth=15, n_jobs=-1, random_state=42),
    'GBT': lambda: GradientBoostingClassifier(n_estimators=100, max_depth=6, random_state=42),
    'LogReg': lambda: LogisticRegression(max_iter=1000, random_state=42),
}

for clf_name, clf_fn in classifiers.items():
    print(f"  Architecture A × {clf_name}...", flush=True)
    accs = []
    for b in ELO_BRACKETS:
        acc = train_and_eval_bracket(X_tr, y_tr, e_tr, X_te, y_te, e_te, p_te, clf_fn, b)
        if acc is not None:
            accs.append(acc)
    if accs:
        ablation_results.append({
            'Architecture': 'A (Per-bracket)',
            'Classifier': clf_name,
            'Top-1 Accuracy': f"{np.mean(accs):.4f}",
        })

# Architecture B × RF
print("  Architecture B × RF...", flush=True)
elo_feat = ((e_tr - 1100) / 800).reshape(-1, 1)
X_tr_b = np.hstack([X_tr, elo_feat])
elo_feat_te = ((e_te - 1100) / 800).reshape(-1, 1)
X_te_b = np.hstack([X_te, elo_feat_te])

clf_b = RandomForestClassifier(n_estimators=200, max_depth=15, n_jobs=-1, random_state=42)
clf_b.fit(X_tr_b, y_tr)
proba_b = clf_b.predict_proba(X_te_b)[:, 1]

accs_b = []
for b in ELO_BRACKETS:
    mask_te = np.abs(e_te - b) <= ELO_BRACKET_WIDTH
    if mask_te.sum() < 50:
        continue
    pidx = p_te[mask_te]
    y_b = y_te[mask_te]
    pb = proba_b[mask_te]
    unique_pos = np.unique(pidx)
    top1 = 0
    for p in unique_pos:
        m = pidx == p
        ranked = np.argsort(pb[m])[::-1]
        if y_b[m][ranked[0]] == 1:
            top1 += 1
    accs_b.append(top1 / len(unique_pos))

ablation_results.append({
    'Architecture': 'B (Pooled+ELO)',
    'Classifier': 'RF',
    'Top-1 Accuracy': f"{np.mean(accs_b):.4f}",
})

# Architecture C × RF (different bandwidths)
print("  Architecture C × RF (bandwidths)...", flush=True)

# Load pre-trained bracket models
with open('models/saved/models_a_candidate.pkl', 'rb') as f:
    models_a = pickle.load(f)

from chess_tutor.models.kernel_interpolation import NadarayaWatsonELO

for bw in [50, 75, 100, 150, 200]:
    nw = NadarayaWatsonELO(bandwidth=bw)
    accs_c = []
    for target_elo in ELO_BRACKETS:
        mask_te = np.abs(e_te - target_elo) <= ELO_BRACKET_WIDTH
        if mask_te.sum() < 50:
            continue
        X_t = X_te[mask_te]
        y_t = y_te[mask_te]
        pidx = p_te[mask_te]

        bracket_probas = {}
        for b, clf in models_a.items():
            bracket_probas[b] = clf.predict_proba(X_t)[:, 1]

        weights = nw.kernel_weights(float(target_elo), list(models_a.keys()))
        combined = np.zeros(len(X_t))
        for w, b in zip(weights, models_a.keys()):
            combined += w * bracket_probas[b]

        unique_pos = np.unique(pidx)
        top1 = 0
        for p in unique_pos:
            m = pidx == p
            ranked = np.argsort(combined[m])[::-1]
            if y_t[m][ranked[0]] == 1:
                top1 += 1
        accs_c.append(top1 / len(unique_pos))

    ablation_results.append({
        'Architecture': f'C (Kernel, bw={bw})',
        'Classifier': 'RF',
        'Top-1 Accuracy': f"{np.mean(accs_c):.4f}",
    })

# Print ablation table
abl_df = pd.DataFrame(ablation_results)
print("\n" + "=" * 60)
print("ABLATION TABLE (Plot 9)")
print("=" * 60)
print(abl_df.to_string(index=False))

# Save as CSV
abl_df.to_csv('results/ablation_table.csv', index=False)

# Plot ablation as heatmap-like table
fig, ax = plt.subplots(figsize=(10, 5))
ax.axis('off')
table = ax.table(cellText=abl_df.values, colLabels=abl_df.columns,
                 cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)
# Color the accuracy column
for i in range(len(abl_df)):
    acc = float(abl_df.iloc[i]['Top-1 Accuracy'])
    r = min(1, max(0, (acc - 0.10) / 0.10))
    table[i+1, 2].set_facecolor((1-r, 1, 1-r))
ax.set_title('Model Comparison: Architecture × Classifier', fontsize=13, pad=20)
plt.tight_layout()
plt.savefig('results/plots/ablation_table.png', dpi=150, bbox_inches='tight')
plt.close()
print("Plot 9: Ablation table saved ✓")

# ============================================================
# Bandit comparison table
# ============================================================
print("\n" + "=" * 60)
print("BANDIT COMPARISON TABLE")
print("=" * 60)

from chess_tutor.evaluation.metrics import arm_selection_entropy

bandit_table = []
for name, res in results.items():
    entropy = arm_selection_entropy(res['arm_distribution'])
    bandit_table.append({
        'Policy': name,
        'Mean Cum. Reward': f"{res['mean_cumulative_reward']:.3f}",
        'Std': f"{res['std_cumulative_reward']:.3f}",
        'Mean ELO Gain': f"{res['mean_elo_gain']:.1f}",
        'Arm Entropy': f"{entropy:.3f}",
    })

bandit_df = pd.DataFrame(bandit_table)
print(bandit_df.to_string(index=False))
bandit_df.to_csv('results/bandit_comparison.csv', index=False)

print("\n" + "=" * 60)
print("ALL DELIVERABLES COMPLETE")
print("=" * 60)
print("\nFiles generated:")
for root, dirs, files in os.walk('results/'):
    for f in sorted(files):
        path = os.path.join(root, f)
        size = os.path.getsize(path)
        print(f"  {path} ({size//1024}KB)")
