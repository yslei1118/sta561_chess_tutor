"""Tests for chess_tutor.teaching.bandit — 5 bandit policies.

Each policy must:
- Return a valid arm index in [0, n_arms)
- Update without raising on reasonable inputs
- Track pull counts correctly
- Be deterministic given a seed (or explicitly document non-determinism)

We also verify a 'learning' property for TS: after many updates where the
true best arm is arm 0, TS should pull arm 0 the most.
"""

import numpy as np
import pytest

from chess_tutor.teaching.bandit import (
    LinearThompsonSampling, EpsilonGreedy, LinUCB, RandomPolicy,
    RuleBasedPolicy, AlwaysTactical,
)
from chess_tutor.config import N_FEEDBACK_TYPES, N_CONTEXT_FEATURES


@pytest.fixture
def random_context():
    return np.abs(np.random.randn(N_CONTEXT_FEATURES))


@pytest.fixture
def all_policies():
    return {
        'ts': LinearThompsonSampling(N_FEEDBACK_TYPES, N_CONTEXT_FEATURES),
        'eps': EpsilonGreedy(N_FEEDBACK_TYPES),
        'ucb': LinUCB(N_FEEDBACK_TYPES, N_CONTEXT_FEATURES),
        'rand': RandomPolicy(N_FEEDBACK_TYPES),
        'rule': RuleBasedPolicy(N_FEEDBACK_TYPES),
        'alt': AlwaysTactical(N_FEEDBACK_TYPES),
    }


class TestPolicyInterface:
    """Each policy must implement select_arm/update/total_pulls contract."""

    def test_all_policies_return_valid_arm(self, all_policies, random_context):
        for name, policy in all_policies.items():
            arm = policy.select_arm(random_context)
            assert 0 <= arm < N_FEEDBACK_TYPES, f"{name} returned invalid arm {arm}"

    def test_all_policies_accept_update(self, all_policies, random_context):
        for name, policy in all_policies.items():
            policy.update(0, random_context, reward=0.5)
            assert policy.total_pulls[0] == 1, f"{name} missed the update"

    def test_cumulative_reward_tracked(self, all_policies, random_context):
        for name, policy in all_policies.items():
            policy.update(0, random_context, reward=0.5)
            policy.update(1, random_context, reward=0.3)
            assert policy.cumulative_reward == pytest.approx(0.8)


class TestThompsonSampling:
    def test_posterior_covariance_stays_positive_definite(self, random_context):
        ts = LinearThompsonSampling(N_FEEDBACK_TYPES, N_CONTEXT_FEATURES)
        for _ in range(50):
            ts.update(0, random_context, reward=np.random.rand())
        # B must be positive definite for sampling to work
        eig = np.linalg.eigvalsh(ts.B[0])
        assert (eig > 0).all(), "TS precision matrix became singular"

    def test_posterior_mean_finite(self, random_context):
        ts = LinearThompsonSampling(N_FEEDBACK_TYPES, N_CONTEXT_FEATURES)
        for _ in range(20):
            ts.update(0, random_context, reward=np.random.rand())
        assert np.isfinite(ts.mu_hat[0]).all()

    def test_reset_clears_state(self, random_context):
        ts = LinearThompsonSampling(N_FEEDBACK_TYPES, N_CONTEXT_FEATURES)
        for _ in range(10):
            ts.update(0, random_context, reward=0.5)
        ts.reset()
        assert ts.total_pulls.sum() == 0
        assert ts.cumulative_reward == 0.0

    @pytest.mark.slow
    def test_ts_learns_best_arm(self):
        """After many rounds with a constant best arm, TS should pull it most."""
        ts = LinearThompsonSampling(n_arms=3, context_dim=5)
        best_arm = 0
        rng = np.random.RandomState(123)
        for _ in range(500):
            ctx = np.abs(rng.randn(5))
            arm = ts.select_arm(ctx)
            # Arm 0 gives reward 0.9; arms 1, 2 give reward 0.1
            reward = 0.9 if arm == best_arm else 0.1
            ts.update(arm, ctx, reward + rng.normal(0, 0.05))
        # Best arm should have the most pulls
        assert np.argmax(ts.total_pulls) == best_arm
        # Best arm should have received >50% of pulls
        assert ts.total_pulls[best_arm] / ts.total_pulls.sum() > 0.5


class TestEpsilonGreedy:
    def test_exploration_rate_respected(self):
        eps = EpsilonGreedy(n_arms=3, epsilon=1.0)  # always explore
        ctx = np.zeros(5)
        arms = [eps.select_arm(ctx) for _ in range(100)]
        # With eps=1.0, should see all arms
        assert len(set(arms)) == 3

    def test_eps_zero_exploits(self):
        eps = EpsilonGreedy(n_arms=3, epsilon=0.0)
        ctx = np.zeros(5)
        # Pull each arm once so argmax is defined
        eps.update(0, ctx, reward=0.1)
        eps.update(1, ctx, reward=0.9)  # best
        eps.update(2, ctx, reward=0.3)
        # With eps=0, should always pick arm 1
        for _ in range(20):
            assert eps.select_arm(ctx) == 1


class TestLinUCB:
    def test_linucb_upper_confidence_bonus_decreases(self, random_context):
        """After repeated pulls of an arm with the same context, LinUCB's
        upper-confidence bonus for that arm should shrink monotonically.

        UCB bonus = alpha * sqrt(x^T A^-1 x). Each update adds x x^T to A,
        so A^-1 shrinks in the x direction and the quadratic form decreases.
        """
        ucb = LinUCB(N_FEEDBACK_TYPES, N_CONTEXT_FEATURES)
        d = N_CONTEXT_FEATURES

        def current_bonus(arm: int) -> float:
            A_inv = np.linalg.solve(ucb.A[arm], np.eye(d))
            return float(ucb.alpha * np.sqrt(random_context @ A_inv @ random_context))

        bonuses = [current_bonus(0)]
        for _ in range(20):
            ucb.update(0, random_context, reward=0.5)
            bonuses.append(current_bonus(0))

        # Strictly decreasing: each update must reduce the bonus
        for a, b in zip(bonuses, bonuses[1:]):
            assert b < a, f"UCB bonus increased after an update: {a:.6f} -> {b:.6f}"
        # Large cumulative shrinkage (more than 2x drop)
        assert bonuses[-1] < bonuses[0] / 2

        # Arms that were NEVER pulled should retain their initial bonus
        untouched_bonus = current_bonus(5)
        assert untouched_bonus > bonuses[-1], (
            "An un-pulled arm's UCB bonus should exceed a heavily-pulled arm's bonus"
        )


class TestRandomPolicy:
    def test_random_covers_all_arms(self):
        rp = RandomPolicy(n_arms=N_FEEDBACK_TYPES)
        ctx = np.zeros(N_CONTEXT_FEATURES)
        pulled = {rp.select_arm(ctx) for _ in range(200)}
        # Should have pulled every arm at least once
        assert pulled == set(range(N_FEEDBACK_TYPES))


class TestRuleBasedPolicy:
    """Rule-based policy uses context[4]=complexity, context[5]=blunder_prob."""

    def test_high_blunder_prob_selects_blunder_warning(self):
        rule = RuleBasedPolicy(N_FEEDBACK_TYPES)
        ctx = np.zeros(N_CONTEXT_FEATURES)
        ctx[5] = 0.9  # high blunder prob
        assert rule.select_arm(ctx) == 2  # BLUNDER_WARNING

    def test_high_complexity_selects_simplification(self):
        rule = RuleBasedPolicy(N_FEEDBACK_TYPES)
        ctx = np.zeros(N_CONTEXT_FEATURES)
        ctx[4] = 0.8  # high complexity
        ctx[5] = 0.1  # low blunder prob
        assert rule.select_arm(ctx) == 6  # SIMPLIFICATION

    def test_default_branch_strategic_nudge(self):
        rule = RuleBasedPolicy(N_FEEDBACK_TYPES)
        ctx = np.zeros(N_CONTEXT_FEATURES)
        ctx[4] = 0.1  # low complexity
        ctx[5] = 0.1  # low blunder prob
        assert rule.select_arm(ctx) == 1  # STRATEGIC_NUDGE


class TestAlwaysTactical:
    def test_always_returns_zero(self):
        at = AlwaysTactical(N_FEEDBACK_TYPES)
        for ctx in [np.zeros(10), np.ones(5), np.random.randn(3)]:
            assert at.select_arm(ctx) == 0
