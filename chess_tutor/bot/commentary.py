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
        """Generate commentary on the student's move, in ELO-appropriate vocabulary.

        Beginner (< 1300): direct, encouraging, simple nouns only.
        Intermediate (1300–1700): names concepts (development, center, file).
        Advanced (>= 1700): positional language (outpost, initiative, prophylaxis).
        """
        san = board_before.san(student_move)
        is_capture = board_before.is_capture(student_move)
        gives_check = board_before.gives_check(student_move)
        piece = board_before.piece_at(student_move.from_square)
        piece_name = chess.piece_name(piece.piece_type) if piece else "piece"
        to_name = chess.square_name(student_move.to_square)

        if student_elo < 1300:
            tier = "beginner"
        elif student_elo < 1700:
            tier = "intermediate"
        else:
            tier = "advanced"

        parts = [f"You played {san}"]

        if gives_check:
            if tier == "beginner":
                parts.append("— nice, you gave check!")
            elif tier == "intermediate":
                parts.append("— a forcing check that limits my options.")
            else:
                parts.append("— a forcing move; checks gain tempo and can be the start of a combination.")
        elif is_capture:
            captured = board_before.piece_at(student_move.to_square)
            if captured:
                kind = chess.piece_name(captured.piece_type)
                if tier == "beginner":
                    parts.append(f"— capturing the {kind}. Nice!")
                elif tier == "intermediate":
                    parts.append(f"— capturing the {kind}. Check if the recapture leaves you with a good trade.")
                else:
                    parts.append(f"— taking the {kind}. Always evaluate the follow-up and whether the trade improves your structure.")
            else:
                parts.append("— a capture.")
        elif piece and piece.piece_type == chess.PAWN:
            center = {chess.D4, chess.D5, chess.E4, chess.E5}
            if student_move.to_square in center:
                if tier == "beginner":
                    parts.append("— good, a pawn in the center!")
                elif tier == "intermediate":
                    parts.append("— occupying the center limits my pieces.")
                else:
                    parts.append("— a central pawn move — these stake a long-term claim and open lines for the bishops.")
            else:
                if tier == "beginner":
                    parts.append(f"— you moved a pawn to {to_name}.")
                elif tier == "intermediate":
                    parts.append(f"— advancing the pawn to {to_name} to gain space on that side.")
                else:
                    parts.append(f"— a flank pawn move; keep an eye on how this affects your king-side structure.")
        elif piece and piece.piece_type == chess.KNIGHT:
            center_area = {chess.C3, chess.C6, chess.F3, chess.F6, chess.D4, chess.D5, chess.E4, chess.E5}
            in_center = student_move.to_square in center_area
            if in_center:
                if tier == "beginner":
                    parts.append("— your knight went to a good central square!")
                elif tier == "intermediate":
                    parts.append("— a classical knight development to a central square.")
                else:
                    parts.append("— the knight establishes itself near the center, where it attacks the most squares.")
            else:
                if tier == "beginner":
                    parts.append(f"— you moved your knight to {to_name}.")
                elif tier == "intermediate":
                    parts.append(f"— a knight move to {to_name}; centralizing pieces is usually preferred.")
                else:
                    parts.append(f"— a rook-pawn-adjacent knight move; consider whether {to_name} is a genuine outpost or just a rim square.")
        elif piece and piece.piece_type == chess.BISHOP:
            if tier == "beginner":
                parts.append(f"— you developed your bishop.")
            elif tier == "intermediate":
                parts.append(f"— developing the bishop to {to_name} — good coordination with your other pieces.")
            else:
                parts.append(f"— the bishop takes its diagonal; watch whether the trade of the bishop pair later is worth it.")
        elif piece and piece.piece_type == chess.ROOK:
            file = chess.square_file(student_move.to_square)
            pawns_on_file = sum(1 for sq in chess.SQUARES if chess.square_file(sq) == file
                               and board_before.piece_at(sq) and board_before.piece_at(sq).piece_type == chess.PAWN)
            open_file = pawns_on_file == 0
            if open_file:
                if tier == "beginner":
                    parts.append(f"— a rook on the {chr(ord('a')+file)}-file, nice!")
                elif tier == "intermediate":
                    parts.append(f"— putting your rook on the open {chr(ord('a')+file)}-file.")
                else:
                    parts.append(f"— claiming the {chr(ord('a')+file)}-file — this is usually worth doubling rooks on.")
            else:
                if tier == "beginner":
                    parts.append(f"— a rook move.")
                elif tier == "intermediate":
                    parts.append(f"— a rook move to {to_name}; try to find an open or semi-open file.")
                else:
                    parts.append(f"— a prophylactic rook repositioning.")
        elif piece and piece.piece_type == chess.KING:
            if board_before.is_castling(student_move):
                if tier == "beginner":
                    parts.append("— castling, good job!")
                elif tier == "intermediate":
                    parts.append("— castling gets your king safe and activates your rook.")
                else:
                    parts.append("— castling; priorities now shift to central control and the pawn breaks.")
            else:
                if tier == "beginner":
                    parts.append("— a king move.")
                elif tier == "intermediate":
                    parts.append(f"— a king move to {to_name}; avoid walking into threats.")
                else:
                    parts.append(f"— a king walk; in the endgame the king becomes a fighting piece, but in the middlegame prefer safety.")
        else:
            parts.append(f"— a {piece_name} move to {to_name}.")

        # Compare with engine if available (tier-aware phrasing)
        if engine_eval and "best_move" in engine_eval and engine_eval["best_move"]:
            best_uci = engine_eval["best_move"]
            if student_move.uci() == best_uci:
                parts.append("This is the engine's top choice!")
            else:
                try:
                    best_san = board_before.san(chess.Move.from_uci(best_uci))
                    cp_loss = engine_eval.get("cp_loss", 0)
                    if tier == "beginner":
                        parts.append(f"The engine slightly prefers {best_san}, but your move is fine.")
                    elif tier == "intermediate":
                        if cp_loss and cp_loss < 30:
                            parts.append(f"Very close to the engine's {best_san}.")
                        elif cp_loss and cp_loss < 100:
                            parts.append(f"The engine prefers {best_san} ({cp_loss:.0f}cp better).")
                        else:
                            parts.append(f"The engine strongly prefers {best_san} here.")
                    else:  # advanced
                        if cp_loss and cp_loss < 30:
                            parts.append(f"Close to the top choice {best_san} — within noise.")
                        elif cp_loss and cp_loss < 100:
                            parts.append(f"The engine's {best_san} loses less ({cp_loss:.0f}cp); check the critical line.")
                        else:
                            parts.append(f"{best_san} was the main line ({cp_loss:.0f}cp difference) — calculate the refutation.")
                except Exception:
                    pass

        return " ".join(parts)

    def comment_on_bot_move(self, board_before: chess.Board,
                            bot_move: chess.Move,
                            target_elo: int,
                            engine_eval: dict | None = None) -> str:
        """Explain why the bot played this move, using ELO-adaptive vocabulary.

        Beginner (ELO < 1300): plain language, no chess terminology beyond
            piece names. Short explanations.
        Intermediate (1300-1700): adds simple strategic observations
            (development, center, king safety).
        Advanced (>= 1700): uses technical chess vocabulary
            (prophylaxis, outposts, pawn breaks, initiative).
        """
        san = board_before.san(bot_move)
        piece = board_before.piece_at(bot_move.from_square)
        to_name = chess.square_name(bot_move.to_square)

        # ELO tier
        if target_elo < 1300:
            tier = "beginner"
        elif target_elo < 1700:
            tier = "intermediate"
        else:
            tier = "advanced"

        parts = [f"I played {san}"]

        # Captures / checks / castles: branch on the dominant feature first
        if board_before.is_capture(bot_move):
            captured = board_before.piece_at(bot_move.to_square)
            piece_kind = chess.piece_name(captured.piece_type) if captured else "piece"
            if tier == "beginner":
                parts.append(f"to take your {piece_kind}.")
            elif tier == "intermediate":
                parts.append(f"to win the {piece_kind} and gain material.")
            else:
                parts.append(
                    f"to capture the {piece_kind} — the trade favors me here."
                )
        elif board_before.gives_check(bot_move):
            if tier == "beginner":
                parts.append("— check!")
            elif tier == "intermediate":
                parts.append("to put pressure on your king.")
            else:
                parts.append(
                    "— a forcing check that limits your king's options and wins tempo."
                )
        elif board_before.is_castling(bot_move):
            if tier == "beginner":
                parts.append("to castle and keep my king safe.")
            elif tier == "intermediate":
                parts.append(
                    "to castle — king safety is a priority before starting operations."
                )
            else:
                parts.append(
                    "to complete castling and connect the rooks before opening the center."
                )
        elif piece and piece.piece_type == chess.PAWN:
            center = {chess.D4, chess.D5, chess.E4, chess.E5}
            if bot_move.to_square in center:
                if tier == "beginner":
                    parts.append("to fight for the center.")
                elif tier == "intermediate":
                    parts.append("to claim the central square and restrict your pieces.")
                else:
                    parts.append(
                        "to stake a classical claim in the center and open lines "
                        "for the bishops."
                    )
            else:
                if tier == "beginner":
                    parts.append(
                        random.choice(["to gain space.", "to support my pieces."])
                    )
                elif tier == "intermediate":
                    parts.append(random.choice([
                        "to advance on the wing.",
                        "to create a pawn chain.",
                        "to gain space on the queenside.",
                    ]))
                else:
                    parts.append(random.choice([
                        "as a prophylactic pawn move to prepare the break.",
                        "to fix your pawn structure and create a target.",
                        "as a flank pawn break to gain structural assets.",
                    ]))
        elif piece and piece.piece_type == chess.KNIGHT:
            if tier == "beginner":
                parts.append(random.choice([
                    f"to develop my knight.",
                    f"to bring my knight out.",
                ]))
            elif tier == "intermediate":
                parts.append(random.choice([
                    f"to develop my knight to {to_name}.",
                    f"to bring the knight closer to the center.",
                    f"to reroute my knight to a better square.",
                ]))
            else:
                parts.append(random.choice([
                    f"establishing the knight on a natural outpost at {to_name}.",
                    f"to reroute via {to_name} — long-term the knight belongs closer to your king.",
                    f"to cover the key square and deny your pieces access.",
                ]))
        elif piece and piece.piece_type == chess.BISHOP:
            if tier == "beginner":
                parts.append(random.choice([
                    "to develop my bishop.",
                    f"to get my bishop into the game.",
                ]))
            elif tier == "intermediate":
                parts.append(random.choice([
                    f"to develop the bishop to {to_name}.",
                    "to activate the bishop on a long diagonal.",
                    "to place the bishop on a useful square.",
                ]))
            else:
                parts.append(random.choice([
                    f"to put the bishop on the long diagonal — the two bishops are a structural asset.",
                    "to re-route the bishop to its best diagonal before committing elsewhere.",
                    "to pressure your weak squares along this diagonal.",
                ]))
        elif piece and piece.piece_type == chess.ROOK:
            file = chess.square_file(bot_move.to_square)
            file_letter = chr(ord('a') + file)
            if tier == "beginner":
                parts.append(f"to move my rook to a better square.")
            elif tier == "intermediate":
                parts.append(random.choice([
                    f"to activate the rook on the {file_letter}-file.",
                    "to double or connect the rooks.",
                    f"to reposition the rook to {to_name}.",
                ]))
            else:
                parts.append(random.choice([
                    f"to claim the {file_letter}-file and prepare a heavy-piece invasion.",
                    "to double rooks on the only open file — a long-term positional advantage.",
                    "to prepare the rook lift and attack along the third/fourth rank.",
                ]))
        elif piece and piece.piece_type == chess.QUEEN:
            if tier == "beginner":
                parts.append("to bring my queen out.")
            elif tier == "intermediate":
                parts.append(random.choice([
                    "to centralize my queen.",
                    f"to activate the queen on {to_name}.",
                    "to create threats with my queen.",
                ]))
            else:
                parts.append(random.choice([
                    "to reposition the queen and maintain tension across the board.",
                    f"as a prophylactic queen shift to {to_name}, dissuading your plan.",
                    "to coordinate the queen with the heavy pieces for a kingside attack.",
                ]))
        elif piece and piece.piece_type == chess.KING:
            if tier == "beginner":
                parts.append("to move the king.")
            elif tier == "intermediate":
                parts.append("as a king move — keeping the king flexible.")
            else:
                parts.append(
                    "to centralize the king, treating it as a fighting piece in the endgame."
                )
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
