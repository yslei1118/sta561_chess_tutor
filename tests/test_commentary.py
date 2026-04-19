"""Tests for chess_tutor.bot.commentary — commentary varies by piece type.

The commentary generator was rewritten to produce piece-specific, non-repeating
output. These tests guard against regression to the old boilerplate ("to
develop my pieces and control the center" repeated for every move).
"""

import chess
import numpy as np
import pytest

from chess_tutor.bot.commentary import CommentaryGenerator


class TestCommentaryDiversity:
    @pytest.fixture
    def cg(self):
        return CommentaryGenerator()

    def test_different_pieces_produce_different_commentary(self, cg, starting_board):
        """Moving a knight vs a pawn should produce different commentary."""
        knight_move = chess.Move.from_uci("g1f3")
        pawn_move = chess.Move.from_uci("e2e4")
        c_knight = cg.comment_on_bot_move(starting_board, knight_move, target_elo=1500)
        c_pawn = cg.comment_on_bot_move(starting_board, pawn_move, target_elo=1500)
        assert c_knight != c_pawn, (
            "Pawn and knight moves produced identical commentary — "
            "likely fell back to generic boilerplate"
        )

    def test_commentary_not_constant_across_moves(self, cg, starting_board):
        """Over several bot moves, we should see a variety of commentary."""
        moves = [
            chess.Move.from_uci("e2e4"),
            chess.Move.from_uci("g1f3"),
            chess.Move.from_uci("b1c3"),
            chess.Move.from_uci("d2d4"),
        ]
        comments = set()
        for m in moves:
            if m in starting_board.legal_moves:
                comments.add(cg.comment_on_bot_move(starting_board, m, 1500))
        # At least 3 distinct comments from 4 varied moves
        assert len(comments) >= 3

    def test_castling_commentary_mentions_safety(self, cg):
        """Castling should yield a commentary that acknowledges king safety."""
        # A position where kingside castling is legal
        board = chess.Board(
            "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
        )
        castle = chess.Move.from_uci("e1g1")
        if castle in board.legal_moves:
            comment = cg.comment_on_bot_move(board, castle, target_elo=1500)
            # Should mention king / safety / castle / something related
            assert any(kw in comment.lower() for kw in
                       ["king", "castle", "safe", "safety"])


class TestStudentMoveCommentary:
    @pytest.fixture
    def cg(self):
        return CommentaryGenerator()

    def test_student_commentary_returns_string(self, cg, starting_board):
        move = chess.Move.from_uci("e2e4")
        comment = cg.comment_on_student_move(starting_board, move, student_elo=1500)
        assert isinstance(comment, str)
        assert len(comment) > 0

    def test_student_commentary_across_elos(self, cg, starting_board):
        """Commentary should not crash on any ELO level."""
        move = chess.Move.from_uci("e2e4")
        for elo in [800, 1100, 1500, 1900, 2200]:
            comment = cg.comment_on_student_move(starting_board, move, elo)
            assert isinstance(comment, str)
            assert len(comment) > 0


class TestEloAdaptiveCommentary:
    """Commentary vocabulary must genuinely shift with ELO tier.

    These regression tests guard against future removal of the ELO branching
    that was added to comment_on_bot_move / comment_on_student_move.
    """

    @pytest.fixture
    def cg(self):
        return CommentaryGenerator()

    def _pool(self, cg_fn, board, move, elo, n_samples=30):
        import random
        outputs = set()
        for seed in range(n_samples):
            random.seed(seed)
            outputs.add(cg_fn(board, move, elo))
        return outputs

    def test_bot_commentary_differs_beginner_vs_advanced(self, cg, starting_board):
        """At ELO 1100 vs 1900, bot commentary on the same move should
        produce distinct vocabulary — no overlap between the two tiers
        on a variety-aware sample."""
        move = chess.Move.from_uci("g1f3")  # Knight move = 3 random reasons per tier
        beginner = self._pool(cg.comment_on_bot_move, starting_board, move, 1100)
        advanced = self._pool(cg.comment_on_bot_move, starting_board, move, 1900)
        intersect = beginner & advanced
        assert not intersect, (
            f"ELO 1100 and ELO 1900 bot commentary share identical strings: {intersect}"
        )

    def test_student_commentary_differs_beginner_vs_advanced(self, cg, starting_board):
        move = chess.Move.from_uci("e2e4")
        beginner = self._pool(cg.comment_on_student_move, starting_board, move, 1100)
        advanced = self._pool(cg.comment_on_student_move, starting_board, move, 1900)
        assert beginner != advanced, (
            "Beginner and advanced student commentary are identical — "
            "ELO differentiation is missing."
        )

    def test_advanced_commentary_uses_technical_vocabulary(self, cg, starting_board):
        """At ELO 1900, commentary should occasionally mention advanced
        chess concepts (outpost, prophylactic, diagonal, initiative, etc.).
        """
        import random
        move = chess.Move.from_uci("g1f3")  # knight — 3 random reasons
        found_technical = False
        technical_terms = [
            "outpost", "prophylactic", "reroute", "long diagonal",
            "initiative", "structural", "heavy piece", "pawn break",
            "tempo", "refutation", "bishop pair",
        ]
        for seed in range(50):
            random.seed(seed)
            comment = cg.comment_on_bot_move(starting_board, move, target_elo=1900)
            if any(t in comment.lower() for t in technical_terms):
                found_technical = True
                break
        assert found_technical, (
            "Advanced-tier commentary never used any technical chess vocabulary"
        )

    def test_beginner_commentary_avoids_technical_jargon(self, cg, starting_board):
        """At ELO 1100, commentary should never use advanced jargon."""
        import random
        move = chess.Move.from_uci("g1f3")
        jargon = ["prophylactic", "outpost", "refutation", "tempo"]
        for seed in range(30):
            random.seed(seed)
            comment = cg.comment_on_bot_move(starting_board, move, target_elo=1100)
            lower = comment.lower()
            hit = [j for j in jargon if j in lower]
            assert not hit, f"Beginner commentary used jargon {hit}: {comment}"
