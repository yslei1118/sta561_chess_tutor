"""End-to-end integration tests for the Section 10 interactive loop.

These tests exercise the full pipeline used by the notebook's interactive
play-against-bot section, in Python (no widgets). They are slower than unit
tests and marked accordingly.
"""

import chess
import numpy as np
import pytest

from chess_tutor.bot.player import ChessTutorBot
from chess_tutor.bot.commentary import CommentaryGenerator
from chess_tutor.feedback import FeedbackType, FeedbackGenerator
from chess_tutor.data.extract_features import extract_board_features
from chess_tutor.teaching.bandit import LinearThompsonSampling
from chess_tutor.teaching.context import build_context
from chess_tutor.student.model import StudentState
from chess_tutor.config import N_FEEDBACK_TYPES, N_CONTEXT_FEATURES


pytestmark = pytest.mark.integration


class TestSection10Loop:
    """Reproduce the Section 10 interactive-play loop end-to-end."""

    def _simulate_n_turns(self, n_turns, bot_elo, your_elo=1200, seed=42):
        np.random.seed(seed)
        board = chess.Board()
        bot = ChessTutorBot()
        cg = CommentaryGenerator()
        fg = FeedbackGenerator()
        bandit = LinearThompsonSampling(N_FEEDBACK_TYPES, N_CONTEXT_FEATURES)
        student_state = StudentState(elo=your_elo)

        trace = []
        # Use a fixed set of reasonable user moves
        user_moves = ['e2e4', 'g1f3', 'f1c4', 'b1c3', 'd2d3']

        for i, uci in enumerate(user_moves[:n_turns]):
            if board.is_game_over(claim_draw=False):
                break
            user_move = chess.Move.from_uci(uci)
            if user_move not in board.legal_moves:
                break

            # Student commentary + bandit-selected feedback
            cg.comment_on_student_move(board, user_move, your_elo)
            board_feats = extract_board_features(board)
            complexity = min(board.legal_moves.count() / 40.0, 1.0)
            context = build_context(
                board, student_state, blunder_prob=0.3,
                complexity=complexity, board_features=board_feats,
            )
            arm = bandit.select_arm(context)
            ft = FeedbackType(arm)
            feedback = fg.generate(board, your_elo, ft, student_move=user_move)
            bandit.update(arm, context, reward=0.5)
            board.push(user_move)

            if not board.is_game_over(claim_draw=False):
                bot_move = bot.play_move(board, target_elo=bot_elo)
                bot_comment = cg.comment_on_bot_move(board, bot_move, bot_elo)
                board.push(bot_move)
                trace.append({
                    'user': uci,
                    'bot_move': bot_move.uci(),
                    'bot_comment': bot_comment,
                    'feedback_type': ft.name,
                    'feedback': feedback,
                })

        return trace

    def test_loop_completes_without_error(self):
        trace = self._simulate_n_turns(n_turns=3, bot_elo=1500)
        assert len(trace) == 3

    def test_bot_moves_differ_by_elo(self):
        """Section 10's main visible property: bot plays differently at different ELOs."""
        trace_low = self._simulate_n_turns(n_turns=5, bot_elo=1100, seed=42)
        trace_high = self._simulate_n_turns(n_turns=5, bot_elo=1900, seed=42)

        # At least one bot move should differ between the two runs
        differ = sum(
            1 for a, b in zip(trace_low, trace_high)
            if a['bot_move'] != b['bot_move']
        )
        assert differ >= 2, (
            f"Only {differ}/5 bot moves differed between ELO 1100 and 1900 "
            "runs — ELO differentiation is too weak"
        )

    def test_feedback_types_vary(self):
        """Over several turns, the bandit should pick multiple feedback types."""
        trace = self._simulate_n_turns(n_turns=5, bot_elo=1500, seed=42)
        unique_types = {t['feedback_type'] for t in trace}
        assert len(unique_types) >= 2

    def test_commentary_non_trivial(self):
        trace = self._simulate_n_turns(n_turns=5, bot_elo=1500)
        for entry in trace:
            assert len(entry['bot_comment']) > 10  # not just 'Ok.' etc.
            assert len(entry['feedback']) > 10


class TestPipelineDataShape:
    """Pipeline data flow: features → context → bandit → arm → feedback."""

    def test_feature_context_bandit_pipeline(self, starting_board):
        from chess_tutor.config import N_BOARD_FEATURES

        features = extract_board_features(starting_board)
        assert features.shape == (N_BOARD_FEATURES,)

        state = StudentState(elo=1500)
        context = build_context(
            starting_board, state, blunder_prob=0.3,
            complexity=0.5, board_features=features,
        )
        assert context.shape == (N_CONTEXT_FEATURES,)

        bandit = LinearThompsonSampling(N_FEEDBACK_TYPES, N_CONTEXT_FEATURES)
        arm = bandit.select_arm(context)
        assert 0 <= arm < N_FEEDBACK_TYPES

        ft = FeedbackType(arm)
        fg = FeedbackGenerator()
        text = fg.generate(starting_board, 1500, ft)
        assert len(text) > 0
