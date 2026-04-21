"""Evaluation metrics for the chess tutor system."""

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score


def move_matching_accuracy(y_true: np.ndarray, y_pred: np.ndarray,
                           top_k: int = 1) -> float:
    """Fraction of positions where true move is in top-K predictions."""
    if y_pred.ndim == 1:
        return float((y_true == y_pred).mean())
    # y_pred is (n, k) with top-k predictions
    k = min(top_k, y_pred.shape[1])
    matches = np.array([y_true[i] in y_pred[i, :k] for i in range(len(y_true))])
    return float(matches.mean())


def centipawn_loss_correlation(predicted_losses: np.ndarray,
                               actual_losses: np.ndarray) -> float:
    """Pearson correlation between predicted and actual centipawn losses."""
    if len(predicted_losses) < 2:
        return 0.0
    return float(np.corrcoef(predicted_losses, actual_losses)[0, 1])


def cumulative_regret(rewards: np.ndarray, oracle_reward: float = 1.0) -> np.ndarray:
    """Cumulative regret: Σ(oracle_reward - reward_t)."""
    return np.cumsum(oracle_reward - rewards)


def arm_selection_entropy(arm_counts: np.ndarray) -> float:
    """Shannon entropy of arm selection distribution."""
    probs = arm_counts / arm_counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def elo_gain_rate(elo_trajectory: list[int]) -> float:
    """Rate of ELO improvement (slope of linear fit)."""
    if len(elo_trajectory) < 2:
        return 0.0
    x = np.arange(len(elo_trajectory))
    coeffs = np.polyfit(x, elo_trajectory, 1)
    return float(coeffs[0])


def blunder_detection_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """AUC for blunder detection."""
    if len(np.unique(y_true)) < 2:
        return 0.0
    return float(roc_auc_score(y_true, y_scores))
