"""Mock-based tests for the data pipeline.

These exercise the pipeline code paths without requiring a real Lichess PGN
or Stockfish binary. Specifically:

- ``parse_pgn_file``: fed a small in-memory PGN file.
- ``ChessTutorDataset``: feature extraction, vocabulary, split, save/load.
- ``extract_all_features``: end-to-end consistency.
- ``StockfishEvaluator``: mocked via a fake engine to verify the wrapper.
"""

import io
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import chess
import chess.pgn
import numpy as np
import pandas as pd
import pytest

from chess_tutor.data.parse_pgn import parse_pgn_file
from chess_tutor.data.dataset import ChessTutorDataset
from chess_tutor.data.extract_features import (
    extract_board_features, extract_move_features, extract_all_features,
)
from scripts.build_candidate_dataset import build_candidate_dataset
from scripts.label_real_blunders import load_played_move_pairs


MINIMAL_PGN = """[Event "Test"]
[Site "test"]
[Date "2013.01.01"]
[Round "1"]
[White "A"]
[Black "B"]
[Result "1-0"]
[WhiteElo "1500"]
[BlackElo "1500"]
[TimeControl "600+0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Nb8 10. d4 Nbd7 11. Nbd2 Bb7 12. Bc2 Re8 13. Nf1 Bf8 14. Ng3 g6 15. a4 c5 16. d5 c4 17. Bg5 h6 18. Be3 Nc5 19. Qd2 h5 20. Bxc5 dxc5 21. Nf5 gxf5 22. exf5 Bd6 23. a5 Bf8 24. h4 Bxf5 25. Bxf5 e4 26. Nd4 cxd4 27. cxd4 Bd6 28. Rxe4 Nxe4 29. Qe2 Nf6 30. Qxe8+ Qxe8 31. Re1 Qxe1# 0-1

"""

MULTI_GAME_PGN = MINIMAL_PGN + """\
[Event "Test2"]
[Site "test2"]
[White "C"]
[Black "D"]
[Result "1/2-1/2"]
[WhiteElo "1100"]
[BlackElo "1100"]

1. d4 d5 2. c4 c6 3. Nf3 Nf6 4. Nc3 e6 5. e3 Nbd7 6. Bd3 dxc4 7. Bxc4 b5 8. Bd3 Bb7 9. O-O a6 10. e4 c5 1/2-1/2

"""


@pytest.fixture
def pgn_tmpfile():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pgn", delete=False) as f:
        f.write(MULTI_GAME_PGN)
        path = f.name
    yield path
    os.unlink(path)


class TestParsePGN:
    def test_parse_returns_dataframe(self, pgn_tmpfile):
        df = parse_pgn_file(pgn_tmpfile)
        assert isinstance(df, pd.DataFrame)

    def test_parsed_positions_have_valid_fen(self, pgn_tmpfile):
        df = parse_pgn_file(pgn_tmpfile)
        assert len(df) > 0
        for fen in df["fen"]:
            chess.Board(fen)  # constructor rejects invalid FEN

    def test_elo_brackets_respected(self, pgn_tmpfile):
        df = parse_pgn_file(pgn_tmpfile, elo_brackets=[1500])
        if len(df) > 0:
            # All white_elo should match 1500 bracket (±50)
            assert (abs(df["white_elo"] - 1500) <= 50).all() or \
                   (abs(df["black_elo"] - 1500) <= 50).all()

    def test_move_number_filter(self, pgn_tmpfile):
        df = parse_pgn_file(pgn_tmpfile, min_move=5, max_move=20)
        if len(df) > 0:
            assert (df["move_number"] >= 5).all()
            assert (df["move_number"] <= 20).all()

    def test_player_elo_correct_for_side(self, pgn_tmpfile):
        df = parse_pgn_file(pgn_tmpfile)
        if len(df) == 0:
            return
        # When white to move, player_elo should be white_elo
        white_rows = df[df["side_to_move"] == "white"]
        if not white_rows.empty:
            assert (white_rows["player_elo"] == white_rows["white_elo"]).all()
        black_rows = df[df["side_to_move"] == "black"]
        if not black_rows.empty:
            assert (black_rows["player_elo"] == black_rows["black_elo"]).all()

    def test_empty_pgn(self):
        """Empty PGN file should yield empty DataFrame."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pgn", delete=False) as f:
            path = f.name
        try:
            df = parse_pgn_file(path)
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 0
        finally:
            os.unlink(path)

    def test_game_with_missing_elo_is_skipped(self):
        pgn = """[Event "Test"]
[White "A"]
[Black "B"]
[Result "1-0"]
[WhiteElo "?"]
[BlackElo "?"]

1. e4 e5 1-0
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pgn", delete=False) as f:
            f.write(pgn)
            path = f.name
        try:
            df = parse_pgn_file(path)
            assert len(df) == 0
        finally:
            os.unlink(path)


class TestDataset:
    @pytest.fixture
    def small_dataset(self, pgn_tmpfile, tmp_path):
        ds = ChessTutorDataset(data_dir=str(tmp_path))
        ds.build_from_pgn([pgn_tmpfile])
        return ds

    def test_build_produces_features_matrix(self, small_dataset):
        assert small_dataset.features is not None
        assert small_dataset.features.shape[1] == 40  # 30 + 10

    def test_build_produces_labels_vector(self, small_dataset):
        assert small_dataset.labels is not None
        assert len(small_dataset.labels) == len(small_dataset.positions)

    def test_move_vocab_consistent(self, small_dataset):
        # Every label should index into the vocab
        max_label = int(small_dataset.labels.max())
        assert max_label < len(small_dataset.move_vocab)
        # Vocab keys should all be valid UCI strings
        for uci in small_dataset.move_vocab.keys():
            chess.Move.from_uci(uci)  # raises on invalid

    def test_save_and_load_roundtrip(self, small_dataset, tmp_path):
        small_dataset.save("rt")
        ds2 = ChessTutorDataset(data_dir=str(tmp_path))
        ds2.load("rt")
        assert ds2.n_positions == small_dataset.n_positions
        np.testing.assert_array_equal(ds2.features, small_dataset.features)
        np.testing.assert_array_equal(ds2.labels, small_dataset.labels)

    def test_train_test_split_disjoint(self, small_dataset):
        X_tr, y_tr, mt_tr, X_te, y_te, mt_te = small_dataset.train_test_split(
            test_frac=0.3, random_state=42,
        )
        # Union covers full dataset
        assert X_tr.shape[0] + X_te.shape[0] == small_dataset.features.shape[0]
        # Test fraction roughly correct
        n = small_dataset.features.shape[0]
        assert abs(X_te.shape[0] / n - 0.3) < 0.1

    def test_summary_well_formed(self, small_dataset):
        s = small_dataset.summary()
        assert "n_positions" in s and s["n_positions"] > 0
        assert "elo_range" in s
        assert "feature_dim" in s and s["feature_dim"] == 40


class TestCandidateDatasetReproducibility:
    def test_build_candidate_dataset_saves_source_row_idx(self, pgn_tmpfile, tmp_path):
        parsed = parse_pgn_file(pgn_tmpfile)
        parsed_path = tmp_path / "parsed_positions.parquet"
        parsed.to_parquet(parsed_path)

        arrays = build_candidate_dataset(
            parsed_positions_path=str(parsed_path),
            output_dir=str(tmp_path),
        )

        assert "candidate_source_row_idx.npy" in arrays
        source_rows = arrays["candidate_source_row_idx.npy"]
        y = arrays["candidate_y.npy"]
        pos_idx = arrays["candidate_pos_idx.npy"]
        assert len(source_rows) == len(y) == len(pos_idx)

        for pos in np.unique(pos_idx):
            mask = pos_idx == pos
            assert len(np.unique(source_rows[mask])) == 1
            assert y[mask].sum() == 1

    def test_labeling_uses_source_row_idx_not_sampled_position_idx(self, tmp_path):
        parsed = pd.DataFrame(
            [
                {"fen": chess.Board().fen(), "move_uci": "e2e4"},
                {"fen": chess.Board().fen(), "move_uci": "d2d4"},
                {"fen": chess.Board().fen(), "move_uci": "g1f3"},
            ]
        )
        parsed_path = tmp_path / "parsed_positions.parquet"
        parsed.to_parquet(parsed_path)

        np.save(tmp_path / "candidate_y.npy", np.array([1, 0, 0, 1], dtype=np.int8))
        np.save(tmp_path / "candidate_pos_idx.npy", np.array([0, 0, 1, 1], dtype=np.int32))
        np.save(
            tmp_path / "candidate_source_row_idx.npy",
            np.array([2, 2, 0, 0], dtype=np.int32),
        )

        pairs = load_played_move_pairs(
            parsed_positions_path=str(parsed_path),
            candidate_y_path=str(tmp_path / "candidate_y.npy"),
            candidate_source_row_path=str(tmp_path / "candidate_source_row_idx.npy"),
            candidate_pos_idx_path=str(tmp_path / "candidate_pos_idx.npy"),
        )

        assert pairs == [
            (parsed.iloc[2]["fen"], "g1f3"),
            (parsed.iloc[0]["fen"], "e2e4"),
        ]


class TestExtractAllFeatures:
    def test_shape(self):
        board = chess.Board()
        move = chess.Move.from_uci("e2e4")
        feat = extract_all_features(board, move)
        assert feat.shape == (40,)

    def test_with_stockfish_evals(self):
        board = chess.Board()
        move = chess.Move.from_uci("e2e4")
        feat_no_eval = extract_all_features(board, move)
        feat_with_eval = extract_all_features(
            board, move, sf_eval_before=50.0, sf_eval_after=30.0
        )
        # The cp_loss feature (index 38) should differ
        assert feat_no_eval[38] != feat_with_eval[38]
        # It should be positive when eval drops
        # sign = 1 (white), eval_diff = 1*(30-50)=-20, features[8] = max(0, 20) = 20
        assert feat_with_eval[38] == 20.0

    def test_cp_loss_zero_when_no_evals(self):
        board = chess.Board()
        move = chess.Move.from_uci("e2e4")
        feat = extract_all_features(board, move)
        assert feat[38] == 0.0  # cp_loss column at index 38 (30 + 8)


class TestStockfishEvaluatorMock:
    """Mock-based tests of the Stockfish wrapper — no real binary needed."""

    def _build_mock_engine(self, cp=50, best_move_uci="e2e4"):
        engine = MagicMock()
        info = {
            "score": MagicMock(),
            "pv": [chess.Move.from_uci(best_move_uci)],
        }
        score = info["score"]
        score.white.return_value = MagicMock()
        score.white.return_value.score.return_value = cp
        score.white.return_value.is_mate.return_value = False
        score.white.return_value.mate.return_value = None
        engine.analyse.return_value = info
        return engine

    def test_evaluate_returns_expected_dict(self):
        from chess_tutor.data.stockfish_eval import StockfishEvaluator
        sf = StockfishEvaluator(stockfish_path="dummy")
        sf.engine = self._build_mock_engine(cp=42)
        result = sf.evaluate(chess.Board())
        assert result["score_cp"] == 42.0
        assert result["best_move"] == "e2e4"
        assert result["is_mate"] is False

    def test_evaluate_move_computes_cp_loss(self):
        from chess_tutor.data.stockfish_eval import StockfishEvaluator
        sf = StockfishEvaluator(stockfish_path="dummy")
        # Fake engine: returns 100 cp before, 50 cp after
        engine = MagicMock()
        call_count = {"n": 0}

        def mock_analyse(board, limit, **kw):
            call_count["n"] += 1
            info = {
                "score": MagicMock(),
                "pv": [chess.Move.from_uci("e2e4")],
            }
            cp = 100 if call_count["n"] == 1 else 50
            info["score"].white.return_value = MagicMock()
            info["score"].white.return_value.score.return_value = cp
            info["score"].white.return_value.is_mate.return_value = False
            info["score"].white.return_value.mate.return_value = None
            return info

        engine.analyse.side_effect = mock_analyse
        sf.engine = engine

        board = chess.Board()
        move = chess.Move.from_uci("e2e4")
        result = sf.evaluate_move(board, move)
        # White moved, eval dropped from 100 → 50 from white's view
        # cp_loss = max(0, 100 - 50) = 50
        assert result["cp_loss"] == 50.0
        assert result["is_blunder"] is False  # 50 <= 100 threshold
