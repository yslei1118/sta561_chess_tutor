"""Tests for chess_tutor.teaching.context — 20-dim bandit context builder."""

import numpy as np
import pytest

from chess_tutor.teaching.context import build_context
from chess_tutor.student.model import StudentState
from chess_tutor.config import N_CONTEXT_FEATURES


class TestContextShape:
    def test_context_dimension(self, starting_board):
        ctx = build_context(
            board=starting_board, student_state=StudentState(elo=1500),
            blunder_prob=0.3, complexity=0.5,
        )
        assert ctx.shape == (N_CONTEXT_FEATURES,)


class TestContextContents:
    def test_context_finite(self, starting_board):
        ctx = build_context(
            board=starting_board, student_state=StudentState(elo=1500),
            blunder_prob=0.3, complexity=0.5,
        )
        assert np.isfinite(ctx).all()

    def test_complexity_in_range(self, starting_board):
        ctx = build_context(
            board=starting_board, student_state=StudentState(elo=1500),
            blunder_prob=0.3, complexity=1.5,  # over 1
        )
        assert 0.0 <= ctx[4] <= 1.0  # complexity clipped

    def test_blunder_prob_in_range(self, starting_board):
        ctx = build_context(
            board=starting_board, student_state=StudentState(elo=1500),
            blunder_prob=2.0, complexity=0.5,  # over 1
        )
        assert 0.0 <= ctx[5] <= 1.0  # blunder_prob clipped

    def test_student_elo_normalized(self, starting_board):
        # ELO 1500 is middle of [800, 2200] → 0.5
        ctx = build_context(
            board=starting_board, student_state=StudentState(elo=1500),
            blunder_prob=0.3, complexity=0.5,
        )
        assert ctx[6] == pytest.approx(0.5, abs=0.01)

    def test_low_elo_near_zero(self, starting_board):
        ctx = build_context(
            board=starting_board, student_state=StudentState(elo=800),
            blunder_prob=0.3, complexity=0.5,
        )
        assert ctx[6] == pytest.approx(0.0, abs=0.01)

    def test_high_elo_near_one(self, starting_board):
        ctx = build_context(
            board=starting_board, student_state=StudentState(elo=2200),
            blunder_prob=0.3, complexity=0.5,
        )
        assert ctx[6] == pytest.approx(1.0, abs=0.01)


class TestTrendOneHot:
    def test_trend_one_hot_sums_to_one(self, starting_board):
        for trend in ['improving', 'stable', 'declining']:
            state = StudentState(elo=1500)
            state.trend = trend
            ctx = build_context(
                board=starting_board, student_state=state,
                blunder_prob=0.3, complexity=0.5,
            )
            # Indices 14, 15, 16 are trend one-hot
            assert ctx[14] + ctx[15] + ctx[16] == pytest.approx(1.0)

    def test_trend_improving_correct_index(self, starting_board):
        state = StudentState(elo=1500)
        state.trend = 'improving'
        ctx = build_context(
            board=starting_board, student_state=state,
            blunder_prob=0.3, complexity=0.5,
        )
        assert ctx[14] == 1.0  # improving
        assert ctx[15] == 0.0  # stable
        assert ctx[16] == 0.0  # declining


class TestWeaknessProfile:
    def test_weakness_values_in_zero_one(self, starting_board):
        ctx = build_context(
            board=starting_board, student_state=StudentState(elo=1500),
            blunder_prob=0.3, complexity=0.5,
        )
        # Indices 8, 9, 10 are tactics/strategy/endgame weakness
        for i in [8, 9, 10]:
            assert 0.0 <= ctx[i] <= 1.0


class TestRevivedDimensions:
    """dims 17, 18, 19 used to be zero fallbacks or 'reserved' placeholders.
    These tests guard against regression to dead features.
    """

    def test_game_result_prob_varies_with_material(self):
        import chess
        # Even material
        ctx_even = build_context(
            board=chess.Board(), student_state=StudentState(elo=1500),
            blunder_prob=0.3, complexity=0.5,
        )
        # Heavy material advantage for white
        b_advantage = chess.Board(
            "rnb1kbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1"
        )
        ctx_advantage = build_context(
            board=b_advantage, student_state=StudentState(elo=1500),
            blunder_prob=0.3, complexity=0.5,
        )
        assert ctx_even[17] == pytest.approx(0.5, abs=1e-3)
        assert ctx_advantage[17] > 0.5, "game_result_prob should rise with material"

    def test_phase_one_hot_indicators(self):
        import chess
        # Opening position (all material on board, move 1)
        ctx_opening = build_context(
            board=chess.Board(), student_state=StudentState(elo=1500),
            blunder_prob=0.3, complexity=0.5,
        )
        assert ctx_opening[18] == 1.0  # phase_opening
        assert ctx_opening[19] == 0.0  # phase_endgame

        # Endgame (K+P vs K)
        b_endgame = chess.Board("8/8/4k3/8/3KP3/8/8/8 w - - 0 1")
        ctx_endgame = build_context(
            board=b_endgame, student_state=StudentState(elo=1500),
            blunder_prob=0.3, complexity=0.5,
        )
        assert ctx_endgame[18] == 0.0
        assert ctx_endgame[19] == 1.0

    def test_no_dim_permanently_zero_across_diverse_inputs(self):
        """Every dimension except 12 (time_pressure, deliberately
        un-instrumented in simulation) must vary across realistic inputs.
        """
        import chess, numpy as np

        boards = [
            chess.Board(),
            chess.Board(
                "r1bqkb1r/pppp1ppp/2n5/4N3/2B1P3/8/PPPP1PPP/RNBQK2R b KQkq - 0 4"
            ),
            chess.Board("8/8/4k3/8/3KP3/8/8/8 w - - 0 1"),
            chess.Board(
                "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N2N2/PP2BPPP/R1BQ1RK1 w - - 0 8"
            ),
        ]
        rng = np.random.RandomState(0)
        samples = []
        for b in boards:
            for elo in [900, 1400, 1900]:
                for trend in ["improving", "stable", "declining"]:
                    state = StudentState(elo=elo)
                    state.trend = trend
                    state.weakness_profile = {k: float(rng.rand())
                                              for k in state.weakness_profile}
                    state.recent_cp_losses = list(rng.exponential(50, 12))
                    state.blunder_count = int(rng.randint(0, 10))
                    ctx = build_context(b, state, blunder_prob=float(rng.rand()),
                                        complexity=float(rng.rand()))
                    samples.append(ctx)
        M = np.array(samples)
        for i in range(20):
            if i == 12:
                # Time pressure is intentionally not instrumented
                continue
            assert M[:, i].var() > 1e-12, (
                f"dim {i} has zero variance across diverse inputs — dead feature"
            )
