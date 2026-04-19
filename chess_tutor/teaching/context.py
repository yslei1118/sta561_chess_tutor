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

    # Always compute board_features if not provided — every downstream
    # dimension that depends on the board should draw from a live vector,
    # never a zero fallback.
    if board_features is None:
        from ..data.extract_features import extract_board_features
        board_features = extract_board_features(board)

    # [0:4] Board features subset
    ctx[0] = board_features[12] / 1000.0  # material balance normalized
    ctx[1] = board_features[13] / 40.0    # mobility normalized
    ctx[2] = board_features[14] / 3.0     # king safety shield
    ctx[3] = board_features[15] / 8.0     # king safety attacks

    ctx[4] = min(complexity, 1.0)
    ctx[5] = min(blunder_prob, 1.0)

    # Student state
    ctx[6] = normalize_elo(student_state.elo)

    if student_state.recent_cp_losses:
        avg_cp = float(np.mean(student_state.recent_cp_losses[-10:]))
        ctx[7] = min(avg_cp / 200.0, 1.0)

    wp = student_state.weakness_profile
    ctx[8] = wp.get("tactics", 0.5)
    ctx[9] = wp.get("strategy", 0.5)
    ctx[10] = wp.get("endgame", 0.5)

    ctx[11] = board.fullmove_number / 80.0

    # Time pressure: not instrumented in the simulation (no clock), so
    # this dimension is left at 0 in simulation runs. It becomes live
    # only if a caller overrides it via ``student_state.time_pressure``
    # (optional attribute, not on the dataclass).
    ctx[12] = float(getattr(student_state, "time_pressure", 0.0))

    ctx[13] = min(student_state.blunder_count / 10.0, 1.0)

    # Trend one-hot
    trend = student_state.trend
    ctx[14] = float(trend == "improving")
    ctx[15] = float(trend == "stable")
    ctx[16] = float(trend == "declining")

    # Win probability from centipawn eval — now always live because
    # board_features is always present above.
    ctx[17] = cp_to_win_prob(board_features[12])

    # [18] phase_opening, [19] phase_endgame — replaces the former two
    # "reserved" zero features. Middlegame is implicit (both 0).
    # Drawn from the board's feature vector (indices 26 and 28 are the
    # phase one-hot columns set by ``data.extract_features``).
    ctx[18] = float(board_features[26])  # phase_opening
    ctx[19] = float(board_features[28])  # phase_endgame

    return ctx
