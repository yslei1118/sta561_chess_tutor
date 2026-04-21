"""FEN-based position evaluation at a pre-specified ELO.

Satisfies the A+ requirement:
    "the user should be able to setup a position and have it evaluated by the
     teacher for a pre-specified ELO."

Usage:
    from chess_tutor.interactive import evaluate_fen
    report = evaluate_fen(
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        target_elo=1400,
    )
    print(report["feedback"])
"""

from __future__ import annotations

import chess

from ..bot.player import ChessTutorBot
from ..feedback.generator import FeedbackGenerator
from ..feedback.taxonomy import FeedbackType


def _pick_feedback_type(bot_report: dict, blunder_prob: float) -> FeedbackType:
    """Choose the most informative feedback arm for the position.

    Rules of thumb — no learned policy is needed here because the user
    has explicitly asked for a position analysis (not a teaching arm):
      * high blunder probability → BLUNDER_WARNING
      * winning/losing material imbalance → MOVE_COMPARISON
      * otherwise → STRATEGIC_NUDGE
    """
    if blunder_prob >= 0.5:
        return FeedbackType.BLUNDER_WARNING
    if bot_report.get("assessment") in ("winning", "losing"):
        return FeedbackType.MOVE_COMPARISON
    return FeedbackType.STRATEGIC_NUDGE


def evaluate_position(
    board: chess.Board,
    target_elo: int,
    move_predictor=None,
    feedback_generator: FeedbackGenerator | None = None,
    bot: ChessTutorBot | None = None,
) -> dict:
    """Evaluate a `chess.Board` for a player at `target_elo`.

    Returns a dict with:
        assessment            — 'winning' | 'losing' | 'equal' | 'unclear'
        key_features          — list[str] of concrete observations
        suggested_plan        — natural-language plan, ELO-adapted
        top_moves             — 3 moves the tutor thinks a ``target_elo``
                                player should consider (SAN)
        blunder_probability   — float in [0,1]
        feedback              — full ELO-appropriate feedback paragraph
        feedback_type         — the FeedbackType chosen
    """
    bot = bot or ChessTutorBot(move_predictor=move_predictor)
    fb_gen = feedback_generator or FeedbackGenerator()

    bot_report = bot.evaluate_position(board, target_elo)

    # Position analysis (blunder prob, complexity) comes from the feedback
    # generator's internal analysis path — reuse it so the numbers match.
    analysis = fb_gen.analyze_position_for_feedback(board, engine_eval={})
    blunder_prob = float(analysis.get("blunder_prob", 0.0))

    ft = _pick_feedback_type(bot_report, blunder_prob)
    feedback_text = fb_gen.generate(board, target_elo, ft)

    return {
        "assessment": bot_report["assessment"],
        "confidence": bot_report["confidence"],
        "key_features": bot_report["key_features"],
        "suggested_plan": bot_report["suggested_plan"],
        "top_moves": bot_report["move_predictions"],
        "blunder_probability": blunder_prob,
        "feedback": feedback_text,
        "feedback_type": ft.name,
    }


def evaluate_fen(
    fen: str,
    target_elo: int,
    move_predictor=None,
    feedback_generator: FeedbackGenerator | None = None,
    bot: ChessTutorBot | None = None,
) -> dict:
    """Same as ``evaluate_position`` but takes a FEN string.

    Raises ``ValueError`` if ``fen`` is not a valid FEN.
    """
    try:
        board = chess.Board(fen)
    except ValueError as exc:
        raise ValueError(f"Invalid FEN: {fen!r}") from exc
    return evaluate_position(
        board,
        target_elo,
        move_predictor=move_predictor,
        feedback_generator=feedback_generator,
        bot=bot,
    )


def format_report(report: dict) -> str:
    """Pretty-print a report dict as a human-readable block."""
    lines = [
        f"Assessment: {report['assessment']} (confidence {report['confidence']:.2f})",
        f"Blunder probability: {report['blunder_probability']:.2f}",
    ]
    if report["key_features"]:
        lines.append("Key features:")
        lines.extend(f"  - {f}" for f in report["key_features"])
    lines.append(f"Top moves (for this ELO): {', '.join(report['top_moves'])}")
    lines.append(f"Plan: {report['suggested_plan']}")
    lines.append("")
    lines.append("Feedback:")
    lines.append(f"  {report['feedback']}")
    return "\n".join(lines)
