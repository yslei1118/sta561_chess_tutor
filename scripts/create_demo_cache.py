#!/usr/bin/env python3
"""Sample ~200 positions from the full processed dataset and save a small
demo cache that can be committed to the repo so the notebook runs offline."""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
SRC = PROJ / "data" / "processed"
DST = PROJ / "data" / "demo_cache"

N = 200  # number of positions to keep
SEED = 42


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)

    # ── features & labels ──────────────────────────────────────────────
    features = np.load(SRC / "chess_tutor_v1_features.npy")
    labels = np.load(SRC / "chess_tutor_v1_labels.npy")

    n_total = features.shape[0]
    idx = rng.choice(n_total, size=min(N, n_total), replace=False)
    idx.sort()

    np.save(DST / "chess_tutor_v1_features.npy", features[idx])
    np.save(DST / "chess_tutor_v1_labels.npy", labels[idx])
    print(f"Saved features {features[idx].shape} and labels {labels[idx].shape}")

    # ── positions parquet ──────────────────────────────────────────────
    positions = pd.read_parquet(SRC / "chess_tutor_v1_positions.parquet")
    positions_mini = positions.iloc[idx].reset_index(drop=True)
    positions_mini.to_parquet(DST / "chess_tutor_v1_positions.parquet", index=False)
    print(f"Saved positions parquet ({len(positions_mini)} rows)")

    # ── move vocab (small – just copy) ─────────────────────────────────
    shutil.copy2(
        SRC / "chess_tutor_v1_move_vocab.json",
        DST / "chess_tutor_v1_move_vocab.json",
    )
    print("Copied move_vocab.json")

    # ── candidate arrays (sample same N rows by pos_idx) ───────────────
    if (SRC / "candidate_X.npy").exists():
        cand_X = np.load(SRC / "candidate_X.npy")
        cand_y = np.load(SRC / "candidate_y.npy")
        cand_elos = np.load(SRC / "candidate_elos.npy")
        cand_pos = np.load(SRC / "candidate_pos_idx.npy")

        # keep candidates whose pos_idx is in our sampled set
        mask = np.isin(cand_pos, idx)
        # remap pos_idx to new consecutive indices
        old_to_new = {old: new for new, old in enumerate(idx)}
        new_pos = np.array([old_to_new[p] for p in cand_pos[mask]])

        np.save(DST / "candidate_X.npy", cand_X[mask])
        np.save(DST / "candidate_y.npy", cand_y[mask])
        np.save(DST / "candidate_elos.npy", cand_elos[mask])
        np.save(DST / "candidate_pos_idx.npy", new_pos)
        print(f"Saved candidate arrays ({mask.sum()} candidates)")

    print(f"\nDemo cache written to {DST}")


if __name__ == "__main__":
    main()
