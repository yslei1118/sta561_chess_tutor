"""Tests for chess_tutor.teaching.reward — the analytic reward function.

The reward signal drives every bandit update, so its invariants matter:
- bounded in [0, 1] (required by all bandit implementations we use)
- monotone in cp_improvement (less cp_loss → more reward)
- positive contribution from blunder_avoided and continued_play
"""

import pytest

from chess_tutor.teaching.reward import compute_reward
from chess_tutor.config import REWARD_WEIGHTS


pytestmark = pytest.mark.math


class TestRewardBounds:
    def test_reward_in_unit_interval_all_zeros(self):
        r = compute_reward(
            cp_loss_before_feedback=0, cp_loss_after_feedback=0,
            blunder_avoided=False, continued_play=False,
        )
        assert 0.0 <= r <= 1.0

    def test_reward_in_unit_interval_all_best(self):
        r = compute_reward(
            cp_loss_before_feedback=500, cp_loss_after_feedback=0,
            blunder_avoided=True, continued_play=True,
        )
        assert 0.0 <= r <= 1.0

    def test_reward_in_unit_interval_all_worst(self):
        r = compute_reward(
            cp_loss_before_feedback=0, cp_loss_after_feedback=500,
            blunder_avoided=False, continued_play=False,
        )
        assert 0.0 <= r <= 1.0


class TestRewardMonotonicity:
    def test_better_improvement_gives_better_reward(self):
        r_small = compute_reward(
            cp_loss_before_feedback=100, cp_loss_after_feedback=90,
            blunder_avoided=False, continued_play=False,
        )
        r_big = compute_reward(
            cp_loss_before_feedback=100, cp_loss_after_feedback=10,
            blunder_avoided=False, continued_play=False,
        )
        assert r_big > r_small

    def test_blunder_avoided_increases_reward(self):
        r_no = compute_reward(
            cp_loss_before_feedback=100, cp_loss_after_feedback=50,
            blunder_avoided=False, continued_play=True,
        )
        r_yes = compute_reward(
            cp_loss_before_feedback=100, cp_loss_after_feedback=50,
            blunder_avoided=True, continued_play=True,
        )
        assert r_yes > r_no

    def test_continued_play_increases_reward(self):
        r_no = compute_reward(
            cp_loss_before_feedback=100, cp_loss_after_feedback=50,
            blunder_avoided=False, continued_play=False,
        )
        r_yes = compute_reward(
            cp_loss_before_feedback=100, cp_loss_after_feedback=50,
            blunder_avoided=False, continued_play=True,
        )
        assert r_yes > r_no


class TestRewardWeights:
    def test_weights_sum_to_one(self):
        total = sum(REWARD_WEIGHTS.values())
        assert total == pytest.approx(1.0)

    def test_all_weights_nonneg(self):
        assert all(w >= 0 for w in REWARD_WEIGHTS.values())


class TestCustomWeights:
    def test_custom_weights_override_default(self):
        weights = {'cp_improvement': 1.0, 'blunder_avoided': 0.0, 'continued_play': 0.0}
        r1 = compute_reward(
            cp_loss_before_feedback=100, cp_loss_after_feedback=50,
            blunder_avoided=False, continued_play=False, weights=weights,
        )
        r2 = compute_reward(
            cp_loss_before_feedback=100, cp_loss_after_feedback=50,
            blunder_avoided=True, continued_play=True, weights=weights,
        )
        # With zero weight on bonuses, adding them shouldn't change reward
        assert r1 == pytest.approx(r2)
