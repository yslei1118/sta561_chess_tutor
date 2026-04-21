"""Shared pytest fixtures for the chess_tutor test suite."""

import sys
import os
import numpy as np
import pytest
import chess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(autouse=True)
def fixed_seed():
    """Every test runs with a fixed RNG seed for reproducibility."""
    np.random.seed(42)
    yield
    np.random.seed()


@pytest.fixture
def starting_board():
    """Standard chess starting position."""
    return chess.Board()


@pytest.fixture
def italian_opening_board():
    """Italian game after 1.e4 e5 2.Nf3 Nc6 3.Bc4."""
    return chess.Board(
        "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
    )


@pytest.fixture
def endgame_board():
    """K+P vs K endgame."""
    return chess.Board("8/8/4k3/8/3KP3/8/8/8 w - - 0 1")


@pytest.fixture
def hanging_piece_board():
    """White knight on e5 that can be captured by black knight on c6."""
    return chess.Board(
        "r1bqkb1r/pppp1ppp/2n5/4N3/2B1P3/8/PPPP1PPP/RNBQK2R b KQkq - 0 4"
    )


@pytest.fixture
def elo_brackets():
    """The 5 canonical ELO brackets used throughout the project."""
    return [1100, 1300, 1500, 1700, 1900]
