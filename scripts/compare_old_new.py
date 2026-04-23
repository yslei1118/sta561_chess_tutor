"""Programmatic comparison of old vs new results (post bug-fix refresh)."""

import os
import numpy as np
import pandas as pd

OLD = "results_old_snapshot"
NEW = "results"


def compare_bandit():
    print("=" * 72)
    print("A) BANDIT COMPARISON (bandit_comparison.csv)")
    print("=" * 72)
    old = pd.read_csv(f"{OLD}/bandit_comparison.csv")
    new = pd.read_csv(f"{NEW}/bandit_comparison.csv")

    merged = old.merge(new, on="Policy", suffixes=("_old", "_new"))
    numeric_cols = ["Mean Cum. Reward", "Std", "Mean ELO Gain", "Arm Entropy"]
    print(f"{'Policy':<22}", end="")
    for c in numeric_cols:
        print(f"{c[:10]:>11}_old {c[:10]:>11}_new {'delta':>9} {'%delta':>8}", end=" | ")
    print()

    rows = []
    for _, r in merged.iterrows():
        row = {"Policy": r["Policy"]}
        print(f"{r['Policy']:<22}", end="")
        for c in numeric_cols:
            o = float(r[f"{c}_old"])
            n = float(r[f"{c}_new"])
            d = n - o
            pct = (d / o * 100) if o != 0 else float("nan")
            print(f"{o:>14.3f} {n:>14.3f} {d:>9.3f} {pct:>7.2f}%", end=" | ")
            row[f"{c}_old"] = o
            row[f"{c}_new"] = n
            row[f"{c}_delta"] = d
            row[f"{c}_pct"] = pct
        print()
        rows.append(row)
    cmp_df = pd.DataFrame(rows)

    # Rankings
    print()
    print("Rankings by Mean Cum. Reward:")
    print("  OLD:", list(old.sort_values("Mean Cum. Reward", ascending=False)["Policy"]))
    print("  NEW:", list(new.sort_values("Mean Cum. Reward", ascending=False)["Policy"]))
    print("Rankings by Mean ELO Gain:")
    print("  OLD:", list(old.sort_values("Mean ELO Gain", ascending=False)["Policy"]))
    print("  NEW:", list(new.sort_values("Mean ELO Gain", ascending=False)["Policy"]))

    # Headline: TS/LinUCB vs Random
    rnd_new = new.loc[new["Policy"] == "Random", "Mean Cum. Reward"].iloc[0]
    ts_new = new.loc[new["Policy"] == "Thompson Sampling", "Mean Cum. Reward"].iloc[0]
    lin_new = new.loc[new["Policy"] == "LinUCB (α=1)", "Mean Cum. Reward"].iloc[0]
    print()
    print(f"Headline check (NEW):")
    print(f"  (TS - Random)/Random     = ({ts_new:.3f} - {rnd_new:.3f})/{rnd_new:.3f} = {(ts_new - rnd_new)/rnd_new*100:+.2f}%")
    print(f"  (LinUCB - Random)/Random = ({lin_new:.3f} - {rnd_new:.3f})/{rnd_new:.3f} = {(lin_new - rnd_new)/rnd_new*100:+.2f}%")

    rnd_old = old.loc[old["Policy"] == "Random", "Mean Cum. Reward"].iloc[0]
    ts_old = old.loc[old["Policy"] == "Thompson Sampling", "Mean Cum. Reward"].iloc[0]
    lin_old = old.loc[old["Policy"] == "LinUCB (α=1)", "Mean Cum. Reward"].iloc[0]
    print(f"Headline (OLD for reference):")
    print(f"  TS vs Random     = {(ts_old - rnd_old)/rnd_old*100:+.2f}%")
    print(f"  LinUCB vs Random = {(lin_old - rnd_old)/rnd_old*100:+.2f}%")
    return cmp_df


def compare_cp_losses():
    print()
    print("=" * 72)
    print("B) cp_loss distribution (NEW Stockfish run)")
    print("=" * 72)
    path_new = "data/processed/real_cp_losses.npy"
    if not os.path.exists(path_new):
        print(f"  (missing: {path_new})")
        return None
    cp = np.load(path_new)
    stats = {
        "n": len(cp),
        "mean": float(cp.mean()),
        "median": float(np.median(cp)),
        "p90": float(np.percentile(cp, 90)),
        "p99": float(np.percentile(cp, 99)),
        "blunder_rate": float((cp > 100).mean()),
    }
    print(f"  n={stats['n']:,}")
    print(f"  mean={stats['mean']:.1f}   (Appendix §3.3 claim: 50)")
    print(f"  median={stats['median']:.1f}   (claim: 17)")
    print(f"  p90={stats['p90']:.1f}   (claim: 140)")
    print(f"  p99={stats['p99']:.1f}   (claim: 560)")
    print(f"  blunder_rate={stats['blunder_rate']:.3f}   (claim: 0.14)")
    return stats


def compare_ablation():
    print()
    print("=" * 72)
    print("C) Ablation (ablation_table.csv) — old vs new")
    print("=" * 72)
    old = pd.read_csv(f"{OLD}/ablation_table.csv")
    new = pd.read_csv(f"{NEW}/ablation_table.csv")
    merged = old.merge(new, on=["Architecture", "Classifier"], suffixes=("_old", "_new"))
    for _, r in merged.iterrows():
        o = float(r["Top-1 Accuracy_old"])
        n = float(r["Top-1 Accuracy_new"])
        print(f"  {r['Architecture']:<22} {r['Classifier']:<7} old={o:.4f}  new={n:.4f}  Δ={n-o:+.4f}")
    return merged


def compare_other():
    print()
    print("=" * 72)
    print("D) hyperparam_sweep.csv / arch_c_retune.csv identity")
    print("=" * 72)
    for fname in ["hyperparam_sweep.csv", "arch_c_retune.csv", "continuous_elo_curves.csv"]:
        op = f"{OLD}/{fname}"
        np_ = f"{NEW}/{fname}"
        if not (os.path.exists(op) and os.path.exists(np_)):
            print(f"  {fname}: missing in one side")
            continue
        od = pd.read_csv(op)
        nd = pd.read_csv(np_)
        same_shape = od.shape == nd.shape
        print(f"  {fname}: old shape {od.shape} vs new shape {nd.shape}  same={same_shape}")


if __name__ == "__main__":
    cmp_df = compare_bandit()
    cmp_df.to_csv("results/bandit_comparison_delta.csv", index=False)
    compare_cp_losses()
    compare_ablation()
    compare_other()
    print()
    print("Saved: results/bandit_comparison_delta.csv")
