"""Hyperparameter sweep for the 5 bandit policies.

For each policy, run several hyperparameter configurations and report the
best one. Uses the empirical (Stockfish-based) reward via the updated
simulation/runner.py (no longer self-referential).

Outputs:
    results/hyperparam_sweep.csv
Writes a summary of best config per policy.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from chess_tutor.teaching.bandit import (
    LinearThompsonSampling, EpsilonGreedy, LinUCB, RandomPolicy, RuleBasedPolicy,
)
from chess_tutor.simulation.student_simulator import StudentPopulation
from chess_tutor.simulation.runner import run_experiment
from chess_tutor.config import N_CONTEXT_FEATURES, N_FEEDBACK_TYPES

np.random.seed(42)
os.makedirs("results", exist_ok=True)

N_EPISODES = 150
N_INTERACTIONS = 20
N_STUDENTS = 30

students = StudentPopulation.generate(n_students=N_STUDENTS, random_state=42)

configs = []

# TS hyperparameter: v (exploration scale)
for v in [0.5, 1.0, 2.0]:
    configs.append((f"TS (v={v})",
                    LinearThompsonSampling(N_FEEDBACK_TYPES, N_CONTEXT_FEATURES,
                                           exploration_v=v)))

# LinUCB hyperparameter: alpha (exploration scale)
for alpha in [0.1, 0.5, 1.0, 2.0, 5.0]:
    configs.append((f"LinUCB (α={alpha})",
                    LinUCB(N_FEEDBACK_TYPES, N_CONTEXT_FEATURES, alpha=alpha)))

# Epsilon-Greedy hyperparameter: epsilon
for eps in [0.05, 0.10, 0.20, 0.30]:
    configs.append((f"ε-Greedy (ε={eps})", EpsilonGreedy(N_FEEDBACK_TYPES, epsilon=eps)))

# Baselines (no hyperparameters to tune)
configs.append(("Random", RandomPolicy(N_FEEDBACK_TYPES)))
configs.append(("Rule-Based", RuleBasedPolicy(N_FEEDBACK_TYPES)))

policies = dict(configs)

print(f"Running {len(policies)} configs × {N_EPISODES} episodes × {N_INTERACTIONS} steps ...")
results = run_experiment(
    students, policies,
    n_episodes=N_EPISODES,
    n_interactions_per_episode=N_INTERACTIONS,
)

# Build summary table
rows = []
for name, res in results.items():
    rows.append({
        "Policy": name,
        "Mean Cum. Reward": res["mean_cumulative_reward"],
        "Std": res["std_cumulative_reward"],
        "Mean ELO Gain": res["mean_elo_gain"],
    })

df = pd.DataFrame(rows).sort_values("Mean Cum. Reward", ascending=False)
print("\n" + "=" * 75)
print(f"{'Policy':<25} {'Mean Reward':>12} {'Std':>8} {'ELO Gain':>10}")
print("-" * 75)
for _, r in df.iterrows():
    print(f"{r['Policy']:<25} {r['Mean Cum. Reward']:>12.3f} "
          f"{r['Std']:>8.3f} {r['Mean ELO Gain']:>10.2f}")

out_path = "results/hyperparam_sweep.csv"
df.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")

# Print best config per family
print("\n" + "=" * 75)
print("Best config per family:")
print("-" * 75)
for family in ["TS", "LinUCB", "ε-Greedy"]:
    family_rows = df[df["Policy"].str.startswith(family)]
    if not family_rows.empty:
        best = family_rows.iloc[0]
        print(f"  {family:10} → {best['Policy']:25} reward {best['Mean Cum. Reward']:.3f}")
