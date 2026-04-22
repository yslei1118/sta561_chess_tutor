"""Interactive interfaces: FEN-based position evaluation and human-vs-bot play."""

from .position_evaluator import evaluate_fen, evaluate_position, format_report
from .game import InteractiveGame, play_cli

__all__ = [
    "evaluate_fen",
    "evaluate_position",
    "format_report",
    "InteractiveGame",
    "play_cli",
]
