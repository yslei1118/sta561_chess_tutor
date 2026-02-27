"""Move-by-move commentary generation for interactive play."""

import chess
from ..data.extract_features import extract_board_features, detect_game_phase


class CommentaryGenerator:
    """Generate move-by-move commentary for interactive play."""

    def __init__(self, feedback_generator=None, move_predictor=None,
                 position_evaluator=None):
        self.feedback_generator = feedback_generator
        self.move_predictor = move_predictor
        self.position_evaluator = position_evaluator

    def comment_on_student_move(self, board_before: chess.Board,
                                 student_move: chess.Move,
                                 student_elo: int,
                                 engine_eval: dict | None = None) -> str:
        """Generate commentary on the student's move."""
        san = board_before.san(student_move)
        is_capture = board_before.is_capture(student_move)
        gives_check = board_before.gives_check(student_move)

        parts = [f"You played {san}"]

        # Describe the move
        piece = board_before.piece_at(student_move.from_square)
        piece_name = chess.piece_name(piece.piece_type) if piece else "piece"

        if gives_check:
            parts.append("— a forcing check!")
        elif is_capture:
            captured = board_before.piece_at(student_move.to_square)
            if captured:
                parts.append(f"— capturing the {chess.piece_name(captured.piece_type)}.")
            else:
                parts.append("— a capture.")
        else:
            parts.append(f"— a {piece_name} move.")

        # Compare with engine if available
        if engine_eval and "best_move" in engine_eval and engine_eval["best_move"]:
            best_uci = engine_eval["best_move"]
            if student_move.uci() == best_uci:
                parts.append("This is the engine's top choice — excellent!")
            else:
                try:
                    best_san = board_before.san(chess.Move.from_uci(best_uci))
                    if student_elo < 1400:
                        parts.append(f"The engine slightly prefers {best_san}, but your move is reasonable for your level.")
                    else:
                        cp_loss = engine_eval.get("cp_loss", 0)
                        if cp_loss and cp_loss < 30:
                            parts.append(f"Very close to the engine's {best_san}. Barely any difference.")
                        elif cp_loss and cp_loss < 100:
                            parts.append(f"The engine prefers {best_san}. The difference is about {cp_loss:.0f} centipawns.")
                        else:
                            parts.append(f"The engine strongly prefers {best_san} here.")
                except Exception:
                    pass

        return " ".join(parts)

    def comment_on_bot_move(self, board_before: chess.Board,
                            bot_move: chess.Move,
                            target_elo: int,
                            engine_eval: dict | None = None) -> str:
        """Explain why the bot played this move."""
        san = board_before.san(bot_move)
        piece = board_before.piece_at(bot_move.from_square)
        piece_name = chess.piece_name(piece.piece_type) if piece else "piece"

        parts = [f"I played {san}"]

        if board_before.is_capture(bot_move):
            parts.append("because I saw a chance to win material.")
        elif board_before.gives_check(bot_move):
            parts.append("to put pressure on your king.")
        else:
            phase = detect_game_phase(board_before)
            if phase == "opening":
                parts.append("to develop my pieces and control the center.")
            elif phase == "endgame":
                parts.append("to improve my king position in the endgame.")
            else:
                parts.append("to improve my position.")

        if target_elo < 1300:
            parts.append("At my level, I often play this kind of move in these positions.")
        elif target_elo < 1700:
            parts.append("This felt like the most natural move to me.")

        return " ".join(parts)

    def game_summary(self, moves: list[chess.Move],
                     evaluations: list[dict],
                     student_elo: int) -> str:
        """Generate end-of-game summary with teaching points."""
        n_moves = len(moves)
        blunders = sum(1 for e in evaluations if e.get("cp_loss", 0) > 100)
        mistakes = sum(1 for e in evaluations if 50 < e.get("cp_loss", 0) <= 100)
        good_moves = sum(1 for e in evaluations if e.get("cp_loss", 0) < 10)

        lines = [
            f"Game Summary ({n_moves} moves):",
            f"  Good moves: {good_moves}",
            f"  Mistakes: {mistakes}",
            f"  Blunders: {blunders}",
            "",
        ]

        if blunders == 0:
            lines.append("Great game! You avoided all blunders.")
        elif blunders <= 2:
            lines.append("Decent game, but watch out for those blunders.")
        else:
            lines.append("Several blunders occurred. Focus on checking for opponent threats before each move.")

        if student_elo < 1300:
            lines.append("\nTip: Always check if your pieces are safe before moving!")
        elif student_elo < 1700:
            lines.append("\nTip: Try to think about your opponent's plan before choosing your move.")
        else:
            lines.append("\nTip: Focus on concrete calculation in tactical positions.")

        return "\n".join(lines)
