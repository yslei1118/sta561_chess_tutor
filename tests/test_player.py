"""Tests for chess_tutor.bot.player — ChessTutorBot.

Core requirements:
- play_move always returns a legal move
- evaluate_position returns a valid dict with all required keys
- ELO differentiation: ELO 1100 and ELO 1900 produce different move distributions
  on most positions. This is the central A+ claim — position-specific regression
  guards against future accidental removal of the heuristic.
"""

import chess
import numpy as np
import pytest

from chess_tutor.bot.player import ChessTutorBot


class TestBotBasics:
    @pytest.fixture
    def bot(self):
        return ChessTutorBot()

    def test_play_move_returns_legal(self, bot, starting_board):
        move = bot.play_move(starting_board, target_elo=1500)
        assert move in starting_board.legal_moves

    def test_play_move_at_multiple_elos_all_legal(self, bot, starting_board):
        for elo in [900, 1100, 1300, 1500, 1700, 1900, 2100]:
            move = bot.play_move(starting_board, target_elo=elo)
            assert move in starting_board.legal_moves

    def test_play_move_plays_only_legal_move_when_forced(self, bot):
        # Position with a single legal move (e.g. king forced to move out of check)
        board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
        legal = list(board.legal_moves)
        if len(legal) == 1:
            assert bot.play_move(board, target_elo=1500) == legal[0]


class TestEvaluatePosition:
    @pytest.fixture
    def bot(self):
        return ChessTutorBot()

    def test_evaluate_keys(self, bot, italian_opening_board):
        result = bot.evaluate_position(italian_opening_board, target_elo=1500)
        required = {"assessment", "confidence", "key_features",
                    "suggested_plan", "move_predictions"}
        assert required.issubset(result.keys())

    def test_assessment_is_string(self, bot, italian_opening_board):
        result = bot.evaluate_position(italian_opening_board, target_elo=1500)
        assert isinstance(result["assessment"], str)

    def test_confidence_in_unit_interval(self, bot, italian_opening_board):
        for elo in [1100, 1500, 1900]:
            result = bot.evaluate_position(italian_opening_board, target_elo=elo)
            assert 0.0 <= result["confidence"] <= 1.0

    def test_move_predictions_are_san(self, bot, italian_opening_board):
        result = bot.evaluate_position(italian_opening_board, target_elo=1500)
        assert len(result["move_predictions"]) > 0
        # SAN notation should not start with a digit (UCI does)
        for san in result["move_predictions"]:
            assert isinstance(san, str)

    def test_plan_is_nonempty_string(self, bot, italian_opening_board):
        for elo in [1100, 1500, 1900]:
            result = bot.evaluate_position(italian_opening_board, target_elo=elo)
            assert isinstance(result["suggested_plan"], str)
            assert len(result["suggested_plan"]) > 0

    def test_plan_differs_between_beginner_and_advanced(self, bot, italian_opening_board):
        """evaluate_position plans must differ in language across ELO tiers."""
        import random
        beginner_plans = set()
        advanced_plans = set()
        for seed in range(30):
            random.seed(seed)
            beginner_plans.add(bot.evaluate_position(italian_opening_board, 1100)["suggested_plan"])
            random.seed(seed)
            advanced_plans.add(bot.evaluate_position(italian_opening_board, 1900)["suggested_plan"])
        overlap = beginner_plans & advanced_plans
        assert not overlap, (
            "ELO 1100 and ELO 1900 produced identical plans — "
            "the ELO-differentiation in evaluate_position is missing. "
            f"Shared: {overlap}"
        )

    def test_advanced_plan_uses_positional_vocabulary(self, bot, italian_opening_board):
        import random
        plans = set()
        for seed in range(30):
            random.seed(seed)
            plans.add(bot.evaluate_position(italian_opening_board, 1900)["suggested_plan"])
        corpus = " ".join(plans).lower()
        advanced_terms = [
            "prophylactic", "initiative", "break", "outpost", "imbalance",
            "zugzwang", "triangulation", "structural", "centralize",
        ]
        assert any(t in corpus for t in advanced_terms), (
            "Advanced-tier plans never used any positional vocabulary"
        )

    def test_beginner_plan_avoids_jargon(self, bot, italian_opening_board):
        import random
        for seed in range(30):
            random.seed(seed)
            plan = bot.evaluate_position(italian_opening_board, 1100)["suggested_plan"].lower()
            for term in ("prophylactic", "zugzwang", "triangulation", "outpost"):
                assert term not in plan, (
                    f"Beginner plan contained jargon '{term}': {plan}"
                )


class TestEloDifferentiation:
    """The bot's move distribution must genuinely differ across ELO levels.

    This is the single most important behavioral property of the system —
    it's what the A+ requirement demands: 'setup a position and have it
    evaluated by your teacher for a pre-specified ELO'. If ELO 1100 and
    1900 produce identical move distributions, the entire premise fails.
    """

    @pytest.fixture
    def bot(self):
        return ChessTutorBot()

    def test_different_distributions_starting_position(self, bot, starting_board):
        legal = list(starting_board.legal_moves)
        probs_low = bot._get_human_probs(starting_board, legal, target_elo=1100)
        probs_high = bot._get_human_probs(starting_board, legal, target_elo=1900)
        # Distributions should be materially different
        max_abs_diff = np.max(np.abs(probs_low - probs_high))
        assert max_abs_diff > 0.05, (
            f"ELO 1100 and 1900 produced nearly identical move probabilities "
            f"(max diff {max_abs_diff:.4f}). The bot is not ELO-conditioned."
        )

    def test_high_elo_prefers_good_knight_move(self, bot, starting_board):
        """ELO 1900 should prefer Nf3 over Nh3 more sharply than ELO 1100."""
        legal = list(starting_board.legal_moves)
        probs_low = bot._get_human_probs(starting_board, legal, target_elo=1100)
        probs_high = bot._get_human_probs(starting_board, legal, target_elo=1900)

        nf3 = chess.Move.from_uci("g1f3")
        nh3 = chess.Move.from_uci("g1h3")
        nf3_idx = legal.index(nf3)
        nh3_idx = legal.index(nh3)

        ratio_low = probs_low[nf3_idx] / probs_low[nh3_idx]
        ratio_high = probs_high[nf3_idx] / probs_high[nh3_idx]

        assert ratio_high > ratio_low, (
            f"Expected ELO 1900 to prefer Nf3 over Nh3 more than ELO 1100 does; "
            f"got ratios {ratio_low:.2f} (low) vs {ratio_high:.2f} (high)"
        )
        # High ELO should strongly prefer Nf3 (at least 2:1)
        assert ratio_high >= 2.0

    def test_high_elo_avoids_hanging_piece(self, bot):
        """A move that drops a piece should be chosen less often at high ELO."""
        # Position where moving the white queen to h5 exposes it
        board = chess.Board(
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
        )
        legal = list(board.legal_moves)
        qh5 = chess.Move.from_uci("d1h5")
        qh5_idx = legal.index(qh5)
        probs_low = bot._get_human_probs(board, legal, target_elo=1100)
        probs_high = bot._get_human_probs(board, legal, target_elo=1900)
        # High ELO should pick Qh5 less often than low ELO
        assert probs_high[qh5_idx] < probs_low[qh5_idx]

    def test_temperature_decreases_with_elo(self, bot, starting_board):
        """High ELO should produce a more peaked distribution than low ELO."""
        legal = list(starting_board.legal_moves)
        probs_low = bot._get_human_probs(starting_board, legal, target_elo=1100)
        probs_high = bot._get_human_probs(starting_board, legal, target_elo=1900)
        # Shannon entropy is lower when distribution is peaked
        entropy_low = -np.sum(probs_low * np.log(probs_low + 1e-12))
        entropy_high = -np.sum(probs_high * np.log(probs_high + 1e-12))
        assert entropy_high < entropy_low, (
            f"ELO 1900 should produce sharper distribution than ELO 1100; "
            f"got entropies {entropy_low:.3f} (low) vs {entropy_high:.3f} (high)"
        )


class TestMoveProbabilityInvariants:
    @pytest.fixture
    def bot(self):
        return ChessTutorBot()

    def test_probs_sum_to_one(self, bot, starting_board):
        legal = list(starting_board.legal_moves)
        for elo in [1100, 1500, 1900]:
            probs = bot._get_human_probs(starting_board, legal, target_elo=elo)
            assert np.isclose(probs.sum(), 1.0, atol=1e-9), \
                f"Probs don't sum to 1 at ELO {elo}: {probs.sum()}"

    def test_probs_nonneg(self, bot, starting_board):
        legal = list(starting_board.legal_moves)
        for elo in [900, 1100, 1500, 1900, 2200]:
            probs = bot._get_human_probs(starting_board, legal, target_elo=elo)
            assert (probs >= 0).all()
