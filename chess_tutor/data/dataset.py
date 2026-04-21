"""Main dataset class for the chess tutor project."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import chess

from ..config import ELO_BRACKETS, ELO_BRACKET_WIDTH, BLUNDER_THRESHOLD_CP
from .parse_pgn import parse_pgn_file
from .extract_features import extract_board_features, extract_move_features


class ChessTutorDataset:
    """Main dataset class for the chess tutor project."""

    def __init__(self, data_dir: str = "data/processed/"):
        self.data_dir = Path(data_dir)
        self.positions: pd.DataFrame | None = None
        self.features: np.ndarray | None = None
        self.labels: np.ndarray | None = None
        self._move_vocab: dict[str, int] | None = None

    def build_from_pgn(
        self,
        pgn_paths: list[str],
        stockfish_path: str = "stockfish",
        use_hf_evals: bool = False,
    ) -> None:
        """Full pipeline: PGN -> parsed positions -> features -> save."""
        all_dfs = []
        for path in pgn_paths:
            print(f"Parsing {path} ...")
            df = parse_pgn_file(path)
            all_dfs.append(df)

        self.positions = pd.concat(all_dfs, ignore_index=True)
        print(f"Parsed {len(self.positions)} positions")

        # Extract features
        board_feats = []
        move_feats = []

        for _, row in self.positions.iterrows():
            board = chess.Board(row["fen"])
            move = chess.Move.from_uci(row["move_uci"])
            board_feats.append(extract_board_features(board))
            move_feats.append(extract_move_features(board, move))

        self.positions["board_features"] = board_feats
        self.positions["move_features"] = move_feats

        # Build move vocabulary from UCI strings
        all_moves = self.positions["move_uci"].unique()
        self._move_vocab = {m: i for i, m in enumerate(sorted(all_moves))}
        self.positions["move_idx"] = self.positions["move_uci"].map(self._move_vocab)

        # Build feature arrays
        self.features = np.array([
            np.concatenate([bf, mf])
            for bf, mf in zip(board_feats, move_feats)
        ])
        self.labels = self.positions["move_idx"].values

        # Add placeholder stockfish columns if not available
        if "stockfish_eval" not in self.positions.columns:
            self.positions["stockfish_eval"] = 0.0
            self.positions["best_move_uci"] = ""
            self.positions["cp_loss"] = 0.0
            self.positions["is_blunder"] = False

        if use_hf_evals:
            self._load_hf_evals()

        print(f"Dataset built: {self.n_positions} positions, "
              f"{self.features.shape[1]} features")

    def _load_hf_evals(self):
        """Try to load pre-computed evals from HuggingFace data."""
        eval_path = Path("data/raw/data/train-00000-of-00016.parquet")
        if not eval_path.exists():
            print("No HF evals found, using placeholder values")
            return

        try:
            evals_df = pd.read_parquet(eval_path)
            # Match by FEN
            eval_map = dict(zip(evals_df["fen"], evals_df["cp"]))
            self.positions["stockfish_eval"] = self.positions["fen"].map(
                lambda f: eval_map.get(f, 0.0)
            )
            print(f"Loaded HF evals for {(self.positions['stockfish_eval'] != 0).sum()} positions")
        except Exception as e:
            print(f"Failed to load HF evals: {e}")

    def add_stockfish_evals(self, stockfish_path: str = "stockfish") -> None:
        """Run Stockfish evaluation on all positions."""
        from .stockfish_eval import StockfishEvaluator

        with StockfishEvaluator(stockfish_path) as sf:
            evals = []
            best_moves = []
            cp_losses = []

            for i, row in self.positions.iterrows():
                board = chess.Board(row["fen"])
                result = sf.evaluate(board)
                evals.append(result["score_cp"])
                best_moves.append(result["best_move"])

                move = chess.Move.from_uci(row["move_uci"])
                move_result = sf.evaluate_move(board, move)
                cp_losses.append(move_result["cp_loss"])

                if (i + 1) % 1000 == 0:
                    print(f"Evaluated {i + 1}/{len(self.positions)} positions")

            self.positions["stockfish_eval"] = evals
            self.positions["best_move_uci"] = best_moves
            self.positions["cp_loss"] = cp_losses
            self.positions["is_blunder"] = [
                cl > BLUNDER_THRESHOLD_CP for cl in cp_losses
            ]

    def save(self, filename: str = "chess_tutor_dataset") -> None:
        """Save dataset to parquet files in data_dir."""
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Save positions without numpy arrays
        save_df = self.positions.copy()
        save_df["board_features"] = save_df["board_features"].apply(
            lambda x: x.tolist() if isinstance(x, np.ndarray) else x
        )
        save_df["move_features"] = save_df["move_features"].apply(
            lambda x: x.tolist() if isinstance(x, np.ndarray) else x
        )
        save_df.to_parquet(self.data_dir / f"{filename}_positions.parquet")

        np.save(self.data_dir / f"{filename}_features.npy", self.features)
        np.save(self.data_dir / f"{filename}_labels.npy", self.labels)

        # Save move vocab
        pd.Series(self._move_vocab).to_json(
            self.data_dir / f"{filename}_move_vocab.json"
        )

        print(f"Saved to {self.data_dir / filename}*")

    def load(self, filename: str = "chess_tutor_dataset") -> None:
        """Load dataset from parquet files."""
        self.positions = pd.read_parquet(
            self.data_dir / f"{filename}_positions.parquet"
        )
        self.features = np.load(self.data_dir / f"{filename}_features.npy")
        self.labels = np.load(self.data_dir / f"{filename}_labels.npy")

        # Restore numpy arrays in DataFrame
        self.positions["board_features"] = self.positions["board_features"].apply(
            lambda x: np.array(x) if isinstance(x, list) else x
        )
        self.positions["move_features"] = self.positions["move_features"].apply(
            lambda x: np.array(x) if isinstance(x, list) else x
        )

        import json
        vocab_path = self.data_dir / f"{filename}_move_vocab.json"
        if vocab_path.exists():
            with open(vocab_path) as f:
                self._move_vocab = json.load(f)

        print(f"Loaded {self.n_positions} positions")

    def get_bracket(self, elo: int, tolerance: int = ELO_BRACKET_WIDTH) -> pd.DataFrame:
        """Get positions for a specific ELO bracket."""
        mask = (self.positions["player_elo"] - elo).abs() <= tolerance
        return self.positions[mask]

    def train_test_split(
        self,
        test_month: int | None = None,
        test_frac: float = 0.2,
        random_state: int = 42,
    ) -> tuple:
        """Split into train/test.

        Returns:
            (X_train, y_train, meta_train, X_test, y_test, meta_test)
        """
        if test_month is not None:
            # Temporal split not implemented without date info
            pass

        n = len(self.positions)
        rng = np.random.RandomState(random_state)
        idx = rng.permutation(n)
        split = int(n * (1 - test_frac))

        train_idx = idx[:split]
        test_idx = idx[split:]

        X_train = self.features[train_idx]
        y_train = self.labels[train_idx]
        meta_train = self.positions.iloc[train_idx].reset_index(drop=True)

        X_test = self.features[test_idx]
        y_test = self.labels[test_idx]
        meta_test = self.positions.iloc[test_idx].reset_index(drop=True)

        return X_train, y_train, meta_train, X_test, y_test, meta_test

    def get_features_and_labels(
        self, elo_bracket: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get feature matrix X and label vector y."""
        if elo_bracket is not None:
            mask = (self.positions["player_elo"] - elo_bracket).abs() <= ELO_BRACKET_WIDTH
            indices = np.where(mask.values)[0]
            return self.features[indices], self.labels[indices]
        return self.features, self.labels

    @property
    def elo_brackets(self) -> list[int]:
        """Return sorted list of ELO brackets present in dataset."""
        if self.positions is None:
            return []
        brackets = []
        for b in ELO_BRACKETS:
            mask = (self.positions["player_elo"] - b).abs() <= ELO_BRACKET_WIDTH
            if mask.any():
                brackets.append(b)
        return sorted(brackets)

    @property
    def n_positions(self) -> int:
        """Total number of positions."""
        return len(self.positions) if self.positions is not None else 0

    @property
    def move_vocab(self) -> dict[str, int]:
        return self._move_vocab or {}

    def summary(self) -> dict:
        """Return summary statistics of the dataset."""
        if self.positions is None:
            return {"n_positions": 0}

        bracket_counts = {}
        for b in ELO_BRACKETS:
            mask = (self.positions["player_elo"] - b).abs() <= ELO_BRACKET_WIDTH
            bracket_counts[b] = int(mask.sum())

        return {
            "n_positions": self.n_positions,
            "n_games": self.positions["game_id"].nunique(),
            "n_unique_moves": len(self.move_vocab),
            "elo_range": (
                int(self.positions["player_elo"].min()),
                int(self.positions["player_elo"].max()),
            ),
            "bracket_counts": bracket_counts,
            "feature_dim": self.features.shape[1] if self.features is not None else 0,
        }
