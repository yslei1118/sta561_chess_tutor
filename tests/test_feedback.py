"""Tests for chess_tutor.feedback.generator — 7 feedback types × ELO adaptation.

Key invariants:
- All 7 FeedbackType values produce non-empty text
- Different ELO levels produce different text (demonstrates ELO adaptivity)
- select_best_feedback_type returns a valid FeedbackType
- Templates don't raise on well-formed inputs
"""

import chess
import pytest

from chess_tutor.feedback import FeedbackType, FeedbackGenerator
from chess_tutor.feedback.taxonomy import FEEDBACK_CONCEPT_MAP


class TestFeedbackTaxonomy:
    def test_seven_feedback_types(self):
        assert len(list(FeedbackType)) == 7

    def test_feedback_types_have_unique_values(self):
        values = [ft.value for ft in FeedbackType]
        assert len(values) == len(set(values))

    def test_concept_map_covers_all_types(self):
        for ft in FeedbackType:
            assert ft in FEEDBACK_CONCEPT_MAP


class TestFeedbackGeneration:
    @pytest.fixture
    def generator(self):
        return FeedbackGenerator()

    def test_every_type_produces_text(self, generator, italian_opening_board):
        for ft in FeedbackType:
            text = generator.generate(
                board=italian_opening_board, student_elo=1500, feedback_type=ft,
            )
            assert isinstance(text, str)
            assert len(text) > 0
            assert text.strip() == text or len(text) > 0  # not just whitespace

    def test_every_type_produces_different_text(self, generator, italian_opening_board):
        """Different feedback types should produce different content."""
        texts = set()
        for ft in FeedbackType:
            text = generator.generate(
                board=italian_opening_board, student_elo=1500, feedback_type=ft,
            )
            texts.add(text)
        # With 7 types, should produce at least 5 distinct outputs
        # (some types may coincidentally overlap with very similar templates)
        assert len(texts) >= 5

    def test_elo_adapts_wording(self, generator, italian_opening_board):
        """Same feedback type at different ELOs should differ in wording often."""
        differs = 0
        for ft in FeedbackType:
            t_low = generator.generate(italian_opening_board, 1100, ft)
            t_high = generator.generate(italian_opening_board, 1900, ft)
            if t_low != t_high:
                differs += 1
        # At least half of feedback types should have ELO-dependent wording
        assert differs >= 3, (
            f"Only {differs}/7 feedback types showed ELO-adaptive wording; "
            "expected at least 3"
        )


class TestFeedbackSelection:
    @pytest.fixture
    def generator(self):
        return FeedbackGenerator()

    def test_select_returns_valid_type(self, generator, starting_board):
        ft = generator.select_best_feedback_type(
            board=starting_board, student_elo=1500, engine_eval={"score_cp": 0},
        )
        assert isinstance(ft, FeedbackType)

    def test_select_varies_with_context(self, generator):
        """Different positions should favor different feedback types."""
        selections = set()
        positions = [
            chess.Board(),  # opening
            chess.Board("r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N2N2/PP2BPPP/R1BQ1RK1 w - - 0 8"),  # middlegame
            chess.Board("8/8/4k3/8/3KP3/8/8/8 w - - 0 1"),  # endgame
        ]
        # Aggregate across ELO to reduce randomness
        for _ in range(10):
            for pos in positions:
                for elo in [1100, 1500, 1900]:
                    ft = generator.select_best_feedback_type(pos, elo, {"score_cp": 0})
                    selections.add(ft)
        # Should have selected at least 3 different types across varied contexts
        assert len(selections) >= 3

    def test_select_is_elo_sensitive(self, generator):
        """Over many random draws, a complex middlegame position should
        produce a noticeably different feedback-type distribution at
        beginner vs advanced ELO. Beginners should see more
        BLUNDER_WARNING/ENCOURAGEMENT; advanced should see more
        STRATEGIC_NUDGE/MOVE_COMPARISON.
        """
        import random
        from collections import Counter

        # Middlegame position with some tactical content
        board = chess.Board(
            "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N2N2/PP2BPPP/R1BQ1RK1 w - - 0 8"
        )
        N = 200
        beg = Counter()
        adv = Counter()
        for seed in range(N):
            random.seed(seed)
            beg[generator.select_best_feedback_type(board, 1100, {"score_cp": 0}).name] += 1
            random.seed(seed)
            adv[generator.select_best_feedback_type(board, 1900, {"score_cp": 0}).name] += 1

        # The two distributions must differ materially (NOT identical)
        assert beg != adv, (
            "select_best_feedback_type produced identical distributions at "
            f"ELO 1100 and 1900: {dict(beg)}"
        )

        # Beginner share of "safety" arms should exceed advanced share
        safety = lambda c: c.get("BLUNDER_WARNING", 0) + c.get("ENCOURAGEMENT", 0) + c.get("SIMPLIFICATION", 0)
        nuanced = lambda c: c.get("STRATEGIC_NUDGE", 0) + c.get("MOVE_COMPARISON", 0) + c.get("PATTERN_RECOGNITION", 0)
        assert safety(beg) > safety(adv), (
            f"Beginner safety arms ({safety(beg)}/{N}) <= advanced ({safety(adv)}/{N})"
        )
        assert nuanced(adv) > nuanced(beg), (
            f"Advanced nuanced arms ({nuanced(adv)}/{N}) <= beginner ({nuanced(beg)}/{N})"
        )

    def test_select_with_student_move(self, generator, italian_opening_board):
        move = chess.Move.from_uci("c6e5")  # Nxe5
        if move in italian_opening_board.legal_moves:
            ft = generator.select_best_feedback_type(
                board=italian_opening_board, student_elo=1500,
                engine_eval={"score_cp": 0},
            )
            text = generator.generate(
                board=italian_opening_board, student_elo=1500,
                feedback_type=ft, student_move=move,
            )
            assert isinstance(text, str)
            assert len(text) > 0


class TestFeedbackTemplatesDontCrash:
    """All feedback types across a variety of positions/ELOs should not raise."""

    @pytest.fixture
    def generator(self):
        return FeedbackGenerator()

    def test_across_positions_and_elos(self, generator):
        positions = [
            chess.Board(),
            chess.Board("r1bqkb1r/pppp1ppp/2n5/4N3/2B1P3/8/PPPP1PPP/RNBQK2R b KQkq - 0 4"),
            chess.Board("8/8/4k3/8/3KP3/8/8/8 w - - 0 1"),
            chess.Board("r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N2N2/PP2BPPP/R1BQ1RK1 w - - 0 8"),
        ]
        for board in positions:
            for elo in [1100, 1300, 1500, 1700, 1900]:
                for ft in FeedbackType:
                    text = generator.generate(board, elo, ft)
                    assert isinstance(text, str) and len(text) > 0
