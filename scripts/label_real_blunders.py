"""Label played moves as blunders using real Stockfish cp_loss.

Replaces the synthetic blunder labels (derived from hanging + mobility
features) with ground-truth labels computed by running Stockfish on each
played move in the sampled candidate dataset.

Outputs:
  * data/processed/real_cp_losses.npy
  * data/processed/real_blunder_labels.npy
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import shutil
import time
from multiprocessing import Pool

import chess
import chess.engine
import numpy as np
import pandas as pd


DEFAULT_PROCESSED_DIR = "data/processed"
DEFAULT_PARSED_PATH = os.path.join(DEFAULT_PROCESSED_DIR, "parsed_positions.parquet")
DEFAULT_CANDIDATE_Y_PATH = os.path.join(DEFAULT_PROCESSED_DIR, "candidate_y.npy")
DEFAULT_CANDIDATE_POS_IDX_PATH = os.path.join(DEFAULT_PROCESSED_DIR, "candidate_pos_idx.npy")
DEFAULT_CANDIDATE_SOURCE_ROW_PATH = os.path.join(
    DEFAULT_PROCESSED_DIR, "candidate_source_row_idx.npy"
)


def _resolve_stockfish_path() -> str:
    """Return a usable Stockfish executable path."""
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


DEPTH = 12
BLUNDER_THRESHOLD_CP = 100
N_WORKERS = int(os.environ.get("STOCKFISH_WORKERS", max(1, os.cpu_count() or 1)))


def _load_played_source_rows(
    candidate_y_path: str = DEFAULT_CANDIDATE_Y_PATH,
    candidate_source_row_path: str = DEFAULT_CANDIDATE_SOURCE_ROW_PATH,
    candidate_pos_idx_path: str = DEFAULT_CANDIDATE_POS_IDX_PATH,
) -> np.ndarray:
    """Return source parquet row indices for played moves.

    New candidate datasets save ``candidate_source_row_idx.npy`` so the
    relabeling step can map played moves back to the exact sampled rows.
    For backward compatibility we fall back to ``candidate_pos_idx.npy``,
    which was only safe when the sampled subset matched the parquet order.
    """
    y = np.load(candidate_y_path)
    played_mask = y == 1

    if os.path.exists(candidate_source_row_path):
        source_row_idx = np.load(candidate_source_row_path)
    else:
        source_row_idx = np.load(candidate_pos_idx_path)
        print(
            "Warning: candidate_source_row_idx.npy not found; falling back to "
            "candidate_pos_idx.npy. Regenerate the candidate dataset for exact "
            "Stockfish-label alignment.",
            flush=True,
        )

    if len(source_row_idx) != len(y):
        raise ValueError(
            f"Shape mismatch: {len(source_row_idx)} source rows vs {len(y)} labels"
        )
    return source_row_idx[played_mask].astype(np.int32)


def load_played_move_pairs(
    parsed_positions_path: str = DEFAULT_PARSED_PATH,
    candidate_y_path: str = DEFAULT_CANDIDATE_Y_PATH,
    candidate_source_row_path: str = DEFAULT_CANDIDATE_SOURCE_ROW_PATH,
    candidate_pos_idx_path: str = DEFAULT_CANDIDATE_POS_IDX_PATH,
) -> list[tuple[str, str]]:
    """Load the sampled played moves to relabel with Stockfish."""
    df = pd.read_parquet(parsed_positions_path)
    source_rows = _load_played_source_rows(
        candidate_y_path=candidate_y_path,
        candidate_source_row_path=candidate_source_row_path,
        candidate_pos_idx_path=candidate_pos_idx_path,
    )
    if len(source_rows) == 0:
        return []
    if source_rows.max(initial=0) >= len(df) or source_rows.min(initial=0) < 0:
        raise IndexError("Played source row indices are out of bounds for parsed parquet")
    return [(df.iloc[row]["fen"], df.iloc[row]["move_uci"]) for row in source_rows]


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

            info_before = engine.analyse(board, chess.engine.Limit(depth=DEPTH))
            score_before = info_before["score"].pov(board.turn)
            cp_before = score_before.score(mate_score=10000)

            board.push(move)
            info_after = engine.analyse(board, chess.engine.Limit(depth=DEPTH))
            score_after = info_after["score"].pov(not board.turn)
            cp_after = score_after.score(mate_score=10000)

            cp_loss = max(0.0, cp_before - cp_after)
        except Exception:
            cp_loss = 0.0
        results.append(cp_loss)
    engine.quit()
    return chunk_id, results


def label_real_blunders(processed_dir: str = DEFAULT_PROCESSED_DIR) -> tuple[np.ndarray, np.ndarray]:
    """Run Stockfish on all played moves in the sampled candidate dataset."""
    os.makedirs(processed_dir, exist_ok=True)
    parsed_positions_path = os.path.join(processed_dir, "parsed_positions.parquet")
    candidate_y_path = os.path.join(processed_dir, "candidate_y.npy")
    candidate_source_row_path = os.path.join(processed_dir, "candidate_source_row_idx.npy")
    candidate_pos_idx_path = os.path.join(processed_dir, "candidate_pos_idx.npy")

    print(f"Using Stockfish at: {_resolve_stockfish_path()}", flush=True)
    print("Loading played-move positions and features ...", flush=True)
    df = pd.read_parquet(parsed_positions_path)
    print(f"  Total rows: {len(df):,}")

    source_rows = _load_played_source_rows(
        candidate_y_path=candidate_y_path,
        candidate_source_row_path=candidate_source_row_path,
        candidate_pos_idx_path=candidate_pos_idx_path,
    )
    print(f"  Played moves to label: {len(source_rows):,}")

    pairs = [(df.iloc[row]["fen"], df.iloc[row]["move_uci"]) for row in source_rows]
    if not pairs:
        raise ValueError("No played moves found in candidate_y.npy")

    chunk_size = (len(pairs) + N_WORKERS - 1) // N_WORKERS
    chunks = [
        (i, pairs[i * chunk_size : (i + 1) * chunk_size])
        for i in range(N_WORKERS)
        if pairs[i * chunk_size : (i + 1) * chunk_size]
    ]

    print(f"\nRunning Stockfish depth {DEPTH} on {len(chunks)} processes ...")
    start = time.time()

    with Pool(len(chunks)) as pool:
        chunk_results = pool.map(evaluate_chunk, chunks)

    chunk_results.sort(key=lambda x: x[0])
    cp_losses = np.array(
        [v for _, chunk in chunk_results for v in chunk], dtype=np.float32
    )

    elapsed = time.time() - start
    print(
        f"Done in {elapsed/60:.1f} minutes "
        f"({elapsed/len(cp_losses)*1000:.0f} ms/eval avg)"
    )

    blunder_labels = (cp_losses > BLUNDER_THRESHOLD_CP).astype(np.int32)

    out_cp = os.path.join(processed_dir, "real_cp_losses.npy")
    out_lab = os.path.join(processed_dir, "real_blunder_labels.npy")
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

    return cp_losses, blunder_labels


def main():
    label_real_blunders()


if __name__ == "__main__":
    main()
