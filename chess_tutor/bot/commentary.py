"""Move-by-move commentary generation for interactive play."""

import random
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
        piece = board_before.piece_at(student_move.from_square)
        piece_name = chess.piece_name(piece.piece_type) if piece else "piece"
        to_name = chess.square_name(student_move.to_square)

        parts = [f"You played {san}"]

        if gives_check:
            parts.append("— a forcing check!")
        elif is_capture:
            captured = board_before.piece_at(student_move.to_square)
            if captured:
                parts.append(f"— capturing the {chess.piece_name(captured.piece_type)}.")
            else:
                parts.append("— a capture.")
        elif piece and piece.piece_type == chess.PAWN:
            center = {chess.D4, chess.D5, chess.E4, chess.E5}
            if student_move.to_square in center:
                parts.append("— occupying the center with a pawn. Good idea!")
            else:
                parts.append(f"— advancing the pawn to {to_name}.")
        elif piece and piece.piece_type == chess.KNIGHT:
            center_area = {chess.C3, chess.C6, chess.F3, chess.F6, chess.D4, chess.D5, chess.E4, chess.E5}
            if student_move.to_square in center_area:
                parts.append(f"— a knight developing to a strong central square.")
            else:
                parts.append(f"— moving the knight to {to_name}.")
        elif piece and piece.piece_type == chess.BISHOP:
            parts.append(f"— developing the bishop to {to_name}.")
        elif piece and piece.piece_type == chess.ROOK:
            # Check if moving to open file
            file = chess.square_file(student_move.to_square)
            pawns_on_file = sum(1 for sq in chess.SQUARES if chess.square_file(sq) == file
                               and board_before.piece_at(sq) and board_before.piece_at(sq).piece_type == chess.PAWN)
            if pawns_on_file == 0:
                parts.append(f"— placing the rook on the open {chr(ord('a')+file)}-file.")
            else:
                parts.append(f"— a rook move to {to_name}.")
        elif piece and piece.piece_type == chess.KING:
            if board_before.is_castling(student_move):
                parts.append("— castling to safety. Smart!")
            else:
                parts.append(f"— a king move to {to_name}.")
        else:
            parts.append(f"— a {piece_name} move to {to_name}.")

        # Compare with engine if available
        if engine_eval and "best_move" in engine_eval and engine_eval["best_move"]:
            best_uci = engine_eval["best_move"]
            if student_move.uci() == best_uci:
                parts.append("This is the engine's top choice!")
            else:
                try:
                    best_san = board_before.san(chess.Move.from_uci(best_uci))
                    if student_elo < 1400:
                        parts.append(f"The engine slightly prefers {best_san}, but your move is fine.")
                    else:
                        cp_loss = engine_eval.get("cp_loss", 0)
                        if cp_loss and cp_loss < 30:
                            parts.append(f"Very close to the engine's {best_san}.")
                        elif cp_loss and cp_loss < 100:
                            parts.append(f"The engine prefers {best_san} ({cp_loss:.0f}cp better).")
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
        to_name = chess.square_name(bot_move.to_square)

        parts = [f"I played {san}"]

        if board_before.is_capture(bot_move):
            captured = board_before.piece_at(bot_move.to_square)
            if captured:
                parts.append(f"to take your {chess.piece_name(captured.piece_type)}.")
            else:
                parts.append("to win material.")
        elif board_before.gives_check(bot_move):
            parts.append("to put pressure on your king.")
        elif board_before.is_castling(bot_move):
            parts.append("to castle and get my king to safety.")
        elif piece and piece.piece_type == chess.PAWN:
            center = {chess.D4, chess.D5, chess.E4, chess.E5}
            if bot_move.to_square in center:
                parts.append("to fight for central control.")
            else:
                reasons = ["to gain space.", "to support my pieces.", "to create a pawn chain."]
                parts.append(random.choice(reasons))
        elif piece and piece.piece_type == chess.KNIGHT:
            reasons = [
                f"to bring my knight to the active square {to_name}.",
                f"to develop my knight towards the center.",
                f"to reposition my knight to {to_name}.",
            ]
            parts.append(random.choice(reasons))
        elif piece and piece.piece_type == chess.BISHOP:
            reasons = [
                f"to develop my bishop to the {to_name} diagonal.",
                f"to activate my bishop on {to_name}.",
                f"to place my bishop on a useful diagonal.",
            ]
            parts.append(random.choice(reasons))
        elif piece and piece.piece_type == chess.ROOK:
            file = chess.square_file(bot_move.to_square)
            reasons = [
                f"to activate my rook on the {chr(ord('a')+file)}-file.",
                f"to double rooks or connect them.",
                f"to place my rook on a more active square.",
            ]
            parts.append(random.choice(reasons))
        elif piece and piece.piece_type == chess.QUEEN:
            reasons = [
                f"to bring my queen into the game on {to_name}.",
                f"to centralize my queen.",
                f"to create threats with my queen.",
            ]
            parts.append(random.choice(reasons))
        else:
            parts.append("to improve my position.")

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
