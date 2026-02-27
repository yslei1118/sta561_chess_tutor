"""Context builder for the contextual bandit."""

import numpy as np
import chess

from ..config import N_CONTEXT_FEATURES
from ..utils.helpers import normalize_elo, cp_to_win_prob


def build_context(
    board: chess.Board,
    student_state,
    blunder_prob: float,
    complexity: float,
    board_features: np.ndarray | None = None,
) -> np.ndarray:
    """Build context vector for the contextual bandit.

    Returns:
        np.ndarray of shape (20,)
    """
    ctx = np.zeros(N_CONTEXT_FEATURES, dtype=np.float64)

    # [0:4] Board features subset
    if board_features is not None:
        ctx[0] = board_features[12] / 1000.0  # material balance normalized
        ctx[1] = board_features[13] / 40.0    # mobility normalized
        ctx[2] = board_features[14] / 3.0     # king safety shield
        ctx[3] = board_features[15] / 8.0     # king safety attacks
    else:
        from ..data.extract_features import extract_board_features
        bf = extract_board_features(board)
        ctx[0] = bf[12] / 1000.0
        ctx[1] = bf[13] / 40.0
        ctx[2] = bf[14] / 3.0
        ctx[3] = bf[15] / 8.0

    ctx[4] = min(complexity, 1.0)
    ctx[5] = min(blunder_prob, 1.0)

    # Student state
    ctx[6] = normalize_elo(student_state.elo)

    if student_state.recent_cp_losses:
        avg_cp = np.mean(student_state.recent_cp_losses[-10:])
        ctx[7] = min(avg_cp / 200.0, 1.0)

    wp = student_state.weakness_profile
    ctx[8] = wp.get("tactics", 0.5)
    ctx[9] = wp.get("strategy", 0.5)
    ctx[10] = wp.get("endgame", 0.5)

    ctx[11] = board.fullmove_number / 80.0

    # Time pressure (not available in simulation, default 0)
    ctx[12] = 0.0

    ctx[13] = min(student_state.blunder_count / 10.0, 1.0)

    # Trend one-hot
    trend = student_state.trend
    ctx[14] = float(trend == "improving")
    ctx[15] = float(trend == "stable")
    ctx[16] = float(trend == "declining")

    # Win probability from eval
    if board_features is not None:
        ctx[17] = cp_to_win_prob(board_features[12])
    else:
        ctx[17] = 0.5

    # [18:20] reserved
    ctx[18] = 0.0
    ctx[19] = 0.0

    return ctx
