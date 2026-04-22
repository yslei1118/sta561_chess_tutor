"""Label played moves as blunders using real Stockfish cp_loss.

Replaces the synthetic blunder labels (derived from hanging + mobility features)
with ground-truth labels computed by running Stockfish on each (board, move) pair.

Output: data/processed/real_cp_losses.npy  (float32, one per played move)
        data/processed/real_blunder_labels.npy  (int32, one per played move)

Usage:
    python scripts/label_real_blunders.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import shutil
import time
import numpy as np
import pandas as pd
import chess
import chess.engine
from multiprocessing import Pool


def _resolve_stockfish_path() -> str:
    """Return a usable Stockfish executable path.

    Resolution order:
      1. ``STOCKFISH_PATH`` env var
      2. ``stockfish`` on ``PATH`` (``/opt/homebrew/bin/stockfish`` on macOS,
         ``/usr/bin/stockfish`` on Linux, ``stockfish.exe`` on Windows)
    """
    env = os.environ.get("STOCKFISH_PATH")
    if env and os.path.exists(env):
        return env
    which = shutil.which("stockfish")
    if which:
        return which
    raise FileNotFoundError(
        "Stockfish executable not found. Install it (e.g. "
        "`brew install stockfish`, `apt install stockfish`, or download "
        "from https://stockfishchess.org/download/) and make sure it is "
        "on PATH, or set STOCKFISH_PATH=/abs/path/to/stockfish."
    )


DEPTH = 12  # depth 12 is sufficient for blunder detection (cp_loss > 100)
BLUNDER_THRESHOLD_CP = 100  # matches config.py
N_WORKERS = int(os.environ.get("STOCKFISH_WORKERS", max(1, os.cpu_count() or 1)))


def evaluate_chunk(args):
    """Evaluate a chunk of (fen, move_uci) pairs. Runs in worker process."""
    import asyncio, sys
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    chunk_id, pairs = args
    engine = chess.engine.SimpleEngine.popen_uci(_resolve_stockfish_path())
    results = []
    for fen, move_uci in pairs:
        try:
            board = chess.Board(fen)
            move = chess.Move.from_uci(move_uci)

            # Evaluate before move (from side-to-move perspective)
            info_before = engine.analyse(board, chess.engine.Limit(depth=DEPTH))
            score_before = info_before["score"].pov(board.turn)
            cp_before = score_before.score(mate_score=10000)

            # Push move, evaluate after (from opponent's perspective, so flip)
            board.push(move)
            info_after = engine.analyse(board, chess.engine.Limit(depth=DEPTH))
            score_after = info_after["score"].pov(not board.turn)  # from player who moved
            cp_after = score_after.score(mate_score=10000)

            cp_loss = max(0.0, cp_before - cp_after)
        except Exception:
            cp_loss = 0.0
        results.append(cp_loss)
    engine.quit()
    return chunk_id, results


def main():
    print(f"Using Stockfish at: {_resolve_stockfish_path()}", flush=True)
    print("Loading played-move positions and features ...", flush=True)
    df = pd.read_parquet("data/processed/parsed_positions.parquet")
    print(f"  Total rows: {len(df):,}")

    y = np.load("data/processed/candidate_y.npy")
    pos_idx = np.load("data/processed/candidate_pos_idx.npy")
    played_mask = y == 1
    played_pos = pos_idx[played_mask]
    print(f"  Played moves to label: {len(played_pos):,}")

    pairs = [(df.iloc[p]["fen"], df.iloc[p]["move_uci"]) for p in played_pos]

    chunk_size = (len(pairs) + N_WORKERS - 1) // N_WORKERS
    chunks = [
        (i, pairs[i * chunk_size : (i + 1) * chunk_size])
        for i in range(N_WORKERS)
    ]

    print(f"\nRunning Stockfish depth {DEPTH} on {N_WORKERS} processes ...")
    start = time.time()

    with Pool(N_WORKERS) as pool:
        chunk_results = pool.map(evaluate_chunk, chunks)

    # Reassemble in original order
    chunk_results.sort(key=lambda x: x[0])
    cp_losses = np.array(
        [v for _, chunk in chunk_results for v in chunk], dtype=np.float32
    )

    elapsed = time.time() - start
    print(f"Done in {elapsed/60:.1f} minutes ({elapsed/len(cp_losses)*1000:.0f} ms/eval avg)")

    blunder_labels = (cp_losses > BLUNDER_THRESHOLD_CP).astype(np.int32)

    out_cp = "data/processed/real_cp_losses.npy"
    out_lab = "data/processed/real_blunder_labels.npy"
    np.save(out_cp, cp_losses)
    np.save(out_lab, blunder_labels)

    print(f"\nSaved: {out_cp}")
    print(f"Saved: {out_lab}")
    print()
    print("=" * 50)
    print("Statistics:")
    print(f"  cp_loss  mean: {cp_losses.mean():.1f}")
    print(f"  cp_loss  p50:  {np.median(cp_losses):.1f}")
    print(f"  cp_loss  p90:  {np.percentile(cp_losses, 90):.1f}")
    print(f"  cp_loss  p99:  {np.percentile(cp_losses, 99):.1f}")
    print(f"  Blunder rate (cp_loss > 100): {blunder_labels.mean():.3f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
