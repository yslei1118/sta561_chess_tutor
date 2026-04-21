"""Shared utilities for the chess tutor project."""

import numpy as np
import chess


def sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + np.exp(-x))
    else:
        ez = np.exp(x)
        return ez / (1.0 + ez)


def normalize_elo(elo: int, elo_min: int = 800, elo_max: int = 2200) -> float:
    """Normalize ELO to [0, 1] range."""
    return (elo - elo_min) / (elo_max - elo_min)


def cp_to_win_prob(cp: float) -> float:
    """Convert centipawn score to win probability using logistic model."""
    return sigmoid(cp / 400.0)


def fen_to_board(fen: str) -> chess.Board:
    """Convert FEN string to chess.Board."""
    return chess.Board(fen)


def move_to_uci(move: chess.Move) -> str:
    """Convert chess.Move to UCI string."""
    return move.uci()


def uci_to_move(uci: str) -> chess.Move:
    """Convert UCI string to chess.Move."""
    return chess.Move.from_uci(uci)


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Compute softmax with temperature."""
    x = np.asarray(x, dtype=np.float64)
    x = x / temperature
    x = x - x.max()
    exp_x = np.exp(x)
    return exp_x / exp_x.sum()


def sample_from_distribution(probs: np.ndarray, rng: np.random.RandomState | None = None) -> int:
    """Sample an index from a probability distribution."""
    if rng is None:
        return int(np.random.choice(len(probs), p=probs))
    return int(rng.choice(len(probs), p=probs))
