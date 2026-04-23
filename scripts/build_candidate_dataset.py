"""Build candidate-move dataset for proper move ranking.

For each sampled position, emit one row per legal move with:
  * 40 engineered features
  * y = 1 if the move was actually played, else 0

In addition to the model arrays, we also save the sampled position's
``source_row_idx`` from ``parsed_positions.parquet``. Stockfish relabeling
uses that mapping so the played-move rows remain aligned with the sampled
subset rather than being accidentally re-indexed against the full parquet.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess
import numpy as np
import pandas as pd

from chess_tutor.config import ELO_BRACKETS, ELO_BRACKET_WIDTH
from chess_tutor.data.extract_features import extract_board_features, extract_move_features


DEFAULT_PARSED_PATH = "data/processed/parsed_positions.parquet"
DEFAULT_OUTPUT_DIR = "data/processed"
MAX_POSITIONS_PER_BRACKET = 15000


def _sample_positions(df: pd.DataFrame) -> pd.DataFrame:
    """Return the sampled subset with stable source-row ids preserved."""
    records = []
    for bracket in ELO_BRACKETS:
        mask = (df["player_elo"] - bracket).abs() <= ELO_BRACKET_WIDTH
        bracket_df = df[mask]
        n = min(MAX_POSITIONS_PER_BRACKET, len(bracket_df))
        sampled = bracket_df.sample(n, random_state=42).copy()
        sampled["source_row_idx"] = sampled.index.astype(np.int32)
        records.append(sampled.reset_index(drop=True))
        print(f"  Bracket {bracket}: {n} positions sampled")
    return pd.concat(records, ignore_index=True)


def build_candidate_dataset(
    parsed_positions_path: str = DEFAULT_PARSED_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> dict[str, np.ndarray]:
    """Build and persist the candidate-ranking arrays."""
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_parquet(parsed_positions_path)
    print(f"Total positions in file: {len(df)}")

    sub_df = _sample_positions(df)
    print(f"Total sampled: {len(sub_df)}")

    all_X = []
    all_y = []
    all_elos = []
    all_pos_idx = []
    all_source_row_idx = []

    for pos_idx, row in enumerate(sub_df.itertuples(index=False)):
        if pos_idx % 2000 == 0:
            print(f"  Processing position {pos_idx}/{len(sub_df)}...", flush=True)

        board = chess.Board(row.fen)
        played_move = chess.Move.from_uci(row.move_uci)
        board_features = extract_board_features(board)

        for move in board.legal_moves:
            move_features = extract_move_features(board, move)
            all_X.append(np.concatenate([board_features, move_features]))
            all_y.append(1 if move == played_move else 0)
            all_elos.append(row.player_elo)
            all_pos_idx.append(pos_idx)
            all_source_row_idx.append(row.source_row_idx)

    arrays = {
        "candidate_X.npy": np.array(all_X, dtype=np.float32),
        "candidate_y.npy": np.array(all_y, dtype=np.int8),
        "candidate_elos.npy": np.array(all_elos, dtype=np.int16),
        "candidate_pos_idx.npy": np.array(all_pos_idx, dtype=np.int32),
        "candidate_source_row_idx.npy": np.array(all_source_row_idx, dtype=np.int32),
    }

    print(f"\nCandidate dataset: {arrays['candidate_X.npy'].shape[0]} rows, "
          f"{arrays['candidate_X.npy'].shape[1]} features")
    print(f"Positive rate: {arrays['candidate_y.npy'].mean():.4f}")

    for filename, array in arrays.items():
        np.save(os.path.join(output_dir, filename), array)
    print("Saved candidate dataset.")

    return arrays


def main():
    build_candidate_dataset()


if __name__ == "__main__":
    main()
