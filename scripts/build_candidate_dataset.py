"""Build candidate-move dataset for proper move ranking.

For each position, create (board_features + move_features) for each legal move.
Label = 1 if this was the move played, 0 otherwise.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess
import numpy as np
import pandas as pd
from chess_tutor.data.extract_features import extract_board_features, extract_move_features
from chess_tutor.config import ELO_BRACKETS, ELO_BRACKET_WIDTH

df = pd.read_parquet('data/processed/parsed_positions.parquet')
print(f"Total positions in file: {len(df)}")

# Subsample: 5000 positions per bracket
records = []
for b in ELO_BRACKETS:
    mask = (df['player_elo'] - b).abs() <= ELO_BRACKET_WIDTH
    bracket_df = df[mask]
    n = min(5000, len(bracket_df))
    bracket_df = bracket_df.sample(n, random_state=42)
    records.append(bracket_df)
    print(f"  Bracket {b}: {n} positions sampled")

sub_df = pd.concat(records, ignore_index=True)
print(f"Total sampled: {len(sub_df)}")

all_X = []
all_y = []
all_elos = []
all_pos_idx = []

for pos_idx, (_, row) in enumerate(sub_df.iterrows()):
    if pos_idx % 2000 == 0:
        print(f"  Processing position {pos_idx}/{len(sub_df)}...", flush=True)

    board = chess.Board(row['fen'])
    played_move = chess.Move.from_uci(row['move_uci'])
    bf = extract_board_features(board)

    for move in board.legal_moves:
        mf = extract_move_features(board, move)
        features = np.concatenate([bf, mf])
        all_X.append(features)
        all_y.append(1 if move == played_move else 0)
        all_elos.append(row['player_elo'])
        all_pos_idx.append(pos_idx)

X = np.array(all_X, dtype=np.float32)
y = np.array(all_y, dtype=np.int8)
elos = np.array(all_elos, dtype=np.int16)
pos_idx_arr = np.array(all_pos_idx, dtype=np.int32)

print(f"\nCandidate dataset: {X.shape[0]} rows, {X.shape[1]} features")
print(f"Positive rate: {y.mean():.4f}")

np.save('data/processed/candidate_X.npy', X)
np.save('data/processed/candidate_y.npy', y)
np.save('data/processed/candidate_elos.npy', elos)
np.save('data/processed/candidate_pos_idx.npy', pos_idx_arr)
print("Saved candidate dataset.")
