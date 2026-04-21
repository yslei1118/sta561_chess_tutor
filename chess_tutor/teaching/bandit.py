"""Contextual bandit policies for adaptive feedback selection."""

import numpy as np

from ..config import (
    N_FEEDBACK_TYPES, N_CONTEXT_FEATURES,
    TS_EXPLORATION_V, TS_PRIOR_VARIANCE,
    LINUCB_ALPHA, EPSILON_GREEDY_EPS,
)


class LinearThompsonSampling:
    """Thompson Sampling for Linear Contextual Bandits (Agrawal & Goyal, ICML 2013)."""

    def __init__(self, n_arms: int = N_FEEDBACK_TYPES,
                 context_dim: int = N_CONTEXT_FEATURES,
                 exploration_v: float = TS_EXPLORATION_V,
                 prior_variance: float = TS_PRIOR_VARIANCE):
        self.n_arms = n_arms
        self.d = context_dim
        self.v = exploration_v

        self.B = [np.eye(context_dim) / prior_variance for _ in range(n_arms)]
        self.f = [np.zeros(context_dim) for _ in range(n_arms)]
        self.mu_hat = [np.zeros(context_dim) for _ in range(n_arms)]
        self.n_pulls = np.zeros(n_arms, dtype=int)
        self._cumulative_reward = 0.0

    def select_arm(self, context: np.ndarray) -> int:
        scores = np.zeros(self.n_arms)
        for a in range(self.n_arms):
            cov = self.v ** 2 * np.linalg.solve(
                self.B[a], np.eye(self.d)
            )
            # Make symmetric for numerical stability
            cov = (cov + cov.T) / 2
            cov += 1e-6 * np.eye(self.d)
            theta_sample = np.random.multivariate_normal(self.mu_hat[a], cov)
            scores[a] = theta_sample @ context
        return int(np.argmax(scores))

    def update(self, arm: int, context: np.ndarray, reward: float) -> None:
        self.B[arm] += np.outer(context, context)
        self.f[arm] += reward * context
        self.mu_hat[arm] = np.linalg.solve(self.B[arm], self.f[arm])
        self.n_pulls[arm] += 1
        self._cumulative_reward += reward

    def get_posterior(self, arm: int) -> tuple[np.ndarray, np.ndarray]:
        cov = self.v ** 2 * np.linalg.solve(self.B[arm], np.eye(self.d))
        return self.mu_hat[arm].copy(), cov

    def get_arm_scores(self, context: np.ndarray) -> np.ndarray:
        return np.array([self.mu_hat[a] @ context for a in range(self.n_arms)])

    def reset(self) -> None:
        for a in range(self.n_arms):
            self.B[a] = np.eye(self.d)
            self.f[a] = np.zeros(self.d)
            self.mu_hat[a] = np.zeros(self.d)
        self.n_pulls[:] = 0
        self._cumulative_reward = 0.0

    @property
    def total_pulls(self) -> np.ndarray:
        return self.n_pulls.copy()

    @property
    def cumulative_reward(self) -> float:
        return self._cumulative_reward


class EpsilonGreedy:
    """ε-Greedy baseline policy."""

    def __init__(self, n_arms: int = N_FEEDBACK_TYPES,
                 epsilon: float = EPSILON_GREEDY_EPS):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.arm_rewards = np.zeros(n_arms)
        self.n_pulls = np.zeros(n_arms, dtype=int)
        self._cumulative_reward = 0.0

    def select_arm(self, context: np.ndarray) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_arms)
        if self.n_pulls.sum() == 0:
            return np.random.randint(self.n_arms)
        avg = np.where(self.n_pulls > 0, self.arm_rewards / np.maximum(self.n_pulls, 1), 0)
        return int(np.argmax(avg))

    def update(self, arm: int, context: np.ndarray, reward: float) -> None:
        self.arm_rewards[arm] += reward
        self.n_pulls[arm] += 1
        self._cumulative_reward += reward

    @property
    def total_pulls(self) -> np.ndarray:
        return self.n_pulls.copy()

    @property
    def cumulative_reward(self) -> float:
        return self._cumulative_reward


class LinUCB:
    """LinUCB baseline (Li et al., WWW 2010)."""

    def __init__(self, n_arms: int = N_FEEDBACK_TYPES,
                 context_dim: int = N_CONTEXT_FEATURES,
                 alpha: float = LINUCB_ALPHA):
        self.n_arms = n_arms
        self.d = context_dim
        self.alpha = alpha

        self.A = [np.eye(context_dim) for _ in range(n_arms)]
        self.b = [np.zeros(context_dim) for _ in range(n_arms)]
        self.n_pulls = np.zeros(n_arms, dtype=int)
        self._cumulative_reward = 0.0

    def select_arm(self, context: np.ndarray) -> int:
        scores = np.zeros(self.n_arms)
        for a in range(self.n_arms):
            A_inv = np.linalg.solve(self.A[a], np.eye(self.d))
            theta = A_inv @ self.b[a]
            ucb = theta @ context + self.alpha * np.sqrt(context @ A_inv @ context)
            scores[a] = ucb
        return int(np.argmax(scores))

    def update(self, arm: int, context: np.ndarray, reward: float) -> None:
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context
        self.n_pulls[arm] += 1
        self._cumulative_reward += reward

    @property
    def total_pulls(self) -> np.ndarray:
        return self.n_pulls.copy()

    @property
    def cumulative_reward(self) -> float:
        return self._cumulative_reward


class RandomPolicy:
    """Uniform random baseline."""

    def __init__(self, n_arms: int = N_FEEDBACK_TYPES):
        self.n_arms = n_arms
        self.n_pulls = np.zeros(n_arms, dtype=int)
        self._cumulative_reward = 0.0

    def select_arm(self, context: np.ndarray) -> int:
        arm = np.random.randint(self.n_arms)
        return arm

    def update(self, arm: int, context: np.ndarray, reward: float) -> None:
        self.n_pulls[arm] += 1
        self._cumulative_reward += reward

    @property
    def total_pulls(self) -> np.ndarray:
        return self.n_pulls.copy()

    @property
    def cumulative_reward(self) -> float:
        return self._cumulative_reward


class RuleBasedPolicy:
    """Hand-designed heuristic baseline."""

    def __init__(self, n_arms: int = N_FEEDBACK_TYPES):
        self.n_arms = n_arms
        self.n_pulls = np.zeros(n_arms, dtype=int)
        self._cumulative_reward = 0.0

    def select_arm(self, context: np.ndarray) -> int:
        """Rules based on context features:
        context[5] = blunder_prob
        context[4] = complexity
        """
        if context[5] > 0.5:
            return 2  # BLUNDER_WARNING
        if context[4] > 0.7:
            return 6  # SIMPLIFICATION
        if context[4] > 0.5:
            return 0  # TACTICAL_ALERT
        return 1  # STRATEGIC_NUDGE

    def update(self, arm: int, context: np.ndarray, reward: float) -> None:
        self.n_pulls[arm] += 1
        self._cumulative_reward += reward

    @property
    def total_pulls(self) -> np.ndarray:
        return self.n_pulls.copy()

    @property
    def cumulative_reward(self) -> float:
        return self._cumulative_reward


class AlwaysTactical:
    """Always selects tactical alert feedback."""

    def __init__(self, n_arms: int = N_FEEDBACK_TYPES):
        self.n_arms = n_arms
        self.n_pulls = np.zeros(n_arms, dtype=int)
        self._cumulative_reward = 0.0

    def select_arm(self, context: np.ndarray) -> int:
        return 0  # TACTICAL_ALERT

    def update(self, arm: int, context: np.ndarray, reward: float) -> None:
        self.n_pulls[arm] += 1
        self._cumulative_reward += reward

    @property
    def total_pulls(self) -> np.ndarray:
        return self.n_pulls.copy()

    @property
    def cumulative_reward(self) -> float:
        return self._cumulative_reward
