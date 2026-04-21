"""Reward computation for teaching interactions."""

import numpy as np
from ..config import REWARD_WEIGHTS
from ..utils.helpers import sigmoid


def compute_reward(
    cp_loss_before_feedback: float,
    cp_loss_after_feedback: float,
    blunder_avoided: bool,
    continued_play: bool,
    weights: dict | None = None,
) -> float:
    """Compute reward for a teaching interaction.

    Formula:
        r = α · sigmoid(cp_before - cp_after) + β · blunder_avoided + γ · continued_play
        normalized to [0, 1]
    """
    w = weights or REWARD_WEIGHTS
    alpha = w.get("cp_improvement", 0.5)
    beta = w.get("blunder_avoided", 0.3)
    gamma = w.get("continued_play", 0.2)

    cp_improvement = sigmoid((cp_loss_before_feedback - cp_loss_after_feedback) / 100.0)

    reward = (
        alpha * cp_improvement
        + beta * float(blunder_avoided)
        + gamma * float(continued_play)
    )

    return float(np.clip(reward, 0.0, 1.0))
