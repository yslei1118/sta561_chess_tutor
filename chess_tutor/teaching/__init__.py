"""Teaching engine module."""
from .bandit import (
    LinearThompsonSampling, EpsilonGreedy, LinUCB,
    RandomPolicy, RuleBasedPolicy, AlwaysTactical,
)
from .context import build_context
from .reward import compute_reward
