"""Human-vs-bot play loop with running commentary and end-of-game summary.

Satisfies the A+ requirement:
    "allow the user to play against a bot with running commentary / evaluations."

The :class:`InteractiveGame` keeps the pieces decoupled from any specific
UI — :func:`play_cli` is one wiring, but the same object can be driven by a
Jupyter widget or a web front-end.
"""

from __future__ import annotations

import chess

from ..bot.player import ChessTutorBot
from ..bot.commentary import CommentaryGenerator
from ..feedback.generator import FeedbackGenerator
from .position_evaluator import evaluate_position


def _safe_engine_eval(stockfish_evaluator, board: chess.Board,
                      student_move: chess.Move) -> dict | None:
    """Best-effort Stockfish eval of the move the student just played.

    Returns ``None`` if no evaluator is available or the call fails, so the
    commentary gracefully degrades to engine-free mode.
    """
    if stockfish_evaluator is None:
        return None
    try:
        before = stockfish_evaluator.evaluate(board)
        after_board = board.copy()
        after_board.push(student_move)
        after = stockfish_evaluator.evaluate(after_board)
        sign = 1 if board.turn == chess.WHITE else -1
        cp_loss = max(0.0, sign * (before["score_cp"] - after["score_cp"]))
        return {
            "best_move": before.get("best_move"),
            "cp_loss": cp_loss,
            "score_cp_before": before["score_cp"],
            "score_cp_after": after["score_cp"],
        }
    except Exception:
        return None


class InteractiveGame:
    """A single human-vs-bot game with running commentary.

    Parameters
    ----------
    user_elo:
        The user's (approximate) ELO. Drives the *vocabulary* of commentary
        and feedback: beginner / intermediate / advanced tiers.
    bot_elo:
        The strength at which the bot should play. May differ from
        ``user_elo`` — e.g. a 1000-rated student can ask to play a 1300-rated
        bot to stretch themselves.
    user_color:
        ``chess.WHITE`` or ``chess.BLACK``.
    start_fen:
        Optional starting position as FEN. Defaults to the standard start.
    """

    def __init__(
        self,
        user_elo: int,
        bot_elo: int,
        user_color: chess.Color = chess.WHITE,
        start_fen: str | None = None,
        move_predictor=None,
        stockfish_evaluator=None,
        feedback_generator: FeedbackGenerator | None = None,
    ):
        self.user_elo = int(user_elo)
        self.bot_elo = int(bot_elo)
        self.user_color = user_color
        self.board = chess.Board(start_fen) if start_fen else chess.Board()

        self.bot = ChessTutorBot(
            move_predictor=move_predictor,
            stockfish_evaluator=stockfish_evaluator,
        )
        self.commentary = CommentaryGenerator(
            feedback_generator=feedback_generator,
            move_predictor=move_predictor,
        )
        self.feedback_generator = feedback_generator or FeedbackGenerator()
        self.stockfish_evaluator = stockfish_evaluator

        self.history: list[dict] = []  # list of {fen, san, side, commentary, cp_loss}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def is_user_turn(self) -> bool:
        return (not self.board.is_game_over()) and self.board.turn == self.user_color

    def user_move(self, move_str: str) -> dict:
        """Apply the user's move (UCI or SAN) and return commentary.

        Raises
        ------
        ValueError
            If the move is illegal or the format is not recognized.
        RuntimeError
            If it is not the user's turn.
        """
        if not self.is_user_turn:
            raise RuntimeError("It is not your turn.")

        move = self._parse_move(move_str)
        if move not in self.board.legal_moves:
            raise ValueError(f"Illegal move: {move_str}")

        board_before = self.board.copy()
        engine_eval = _safe_engine_eval(self.stockfish_evaluator, board_before, move)

        san = board_before.san(move)
        self.board.push(move)

        text = self.commentary.comment_on_student_move(
            board_before, move, self.user_elo, engine_eval=engine_eval,
        )

        entry = {
            "fen": board_before.fen(),
            "san": san,
            "side": "user",
            "commentary": text,
            "cp_loss": (engine_eval or {}).get("cp_loss"),
        }
        self.history.append(entry)
        return entry

    def bot_move(self) -> dict:
        """Have the bot play one move. Returns commentary dict.

        Raises
        ------
        RuntimeError
            If it is the user's turn or the game is over.
        """
        if self.board.is_game_over():
            raise RuntimeError("Game is over.")
        if self.is_user_turn:
            raise RuntimeError("It is the user's turn, not the bot's.")

        board_before = self.board.copy()
        move = self.bot.play_move(board_before, self.bot_elo)
        san = board_before.san(move)
        self.board.push(move)

        text = self.commentary.comment_on_bot_move(board_before, move, self.bot_elo)

        entry = {
            "fen": board_before.fen(),
            "san": san,
            "side": "bot",
            "commentary": text,
            "cp_loss": None,
        }
        self.history.append(entry)
        return entry

    def evaluate_current_position(self) -> dict:
        """Run :func:`evaluate_position` on the current board for the user.

        Useful for "pause and analyse this position for my level" during play.
        """
        return evaluate_position(
            self.board, self.user_elo,
            move_predictor=self.bot.move_predictor,
            feedback_generator=self.feedback_generator,
            bot=self.bot,
        )

    def summary(self) -> str:
        """End-of-game natural-language summary of the user's play."""
        user_entries = [h for h in self.history if h["side"] == "user"]
        # Reconstruct Move objects from each entry's pre-move FEN + SAN so
        # commentary.game_summary() can count blunders / mistakes / good moves.
        user_moves: list[chess.Move] = []
        evaluations: list[dict] = []
        for entry in user_entries:
            pre_board = chess.Board(entry["fen"])
            try:
                mv = pre_board.parse_san(entry["san"])
            except ValueError:
                continue
            user_moves.append(mv)
            evaluations.append({"cp_loss": entry["cp_loss"] or 0.0})
        return self.commentary.game_summary(user_moves, evaluations, self.user_elo)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _parse_move(self, move_str: str) -> chess.Move:
        move_str = move_str.strip()
        try:
            return chess.Move.from_uci(move_str)
        except ValueError:
            pass
        try:
            return self.board.parse_san(move_str)
        except ValueError as exc:
            raise ValueError(f"Cannot parse move: {move_str!r}") from exc


def play_cli(
    user_elo: int = 1200,
    bot_elo: int = 1400,
    user_color: str = "white",
    start_fen: str | None = None,
    move_predictor=None,
    stockfish_evaluator=None,
) -> None:
    """Run an interactive terminal game.

    Type UCI ("e2e4") or SAN ("Nf3"). Type ``analyze`` for an on-demand
    evaluation of the current position. ``resign`` ends the game.
    """
    color = chess.WHITE if user_color.lower().startswith("w") else chess.BLACK
    game = InteractiveGame(
        user_elo=user_elo, bot_elo=bot_elo, user_color=color,
        start_fen=start_fen, move_predictor=move_predictor,
        stockfish_evaluator=stockfish_evaluator,
    )

    print(f"You are {'White' if color == chess.WHITE else 'Black'}. "
          f"Bot plays at ~{bot_elo}. Your ELO: {user_elo}.")
    print("Commands: UCI/SAN move, 'analyze', 'board', 'resign'.")
    print(game.board)

    while not game.board.is_game_over():
        if not game.is_user_turn:
            entry = game.bot_move()
            print(f"\nBot: {entry['san']}")
            print(f"  {entry['commentary']}")
            print(game.board)
            continue

        try:
            raw = input("\nYour move: ").strip()
        except EOFError:
            print("\n(EOF — ending game.)")
            break
        if not raw:
            continue
        if raw.lower() == "resign":
            print("You resigned.")
            break
        if raw.lower() == "board":
            print(game.board)
            continue
        if raw.lower() == "analyze":
            rep = game.evaluate_current_position()
            from .position_evaluator import format_report
            print(format_report(rep))
            continue

        try:
            entry = game.user_move(raw)
        except (ValueError, RuntimeError) as exc:
            print(f"  ⚠ {exc}")
            continue
        print(f"  {entry['commentary']}")
        print(game.board)

    print("\n" + game.summary())


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Play chess against the tutor.")
    p.add_argument("--user-elo", type=int, default=1200)
    p.add_argument("--bot-elo", type=int, default=1400)
    p.add_argument("--color", choices=["white", "black"], default="white")
    p.add_argument("--fen", default=None)
    args = p.parse_args()
    play_cli(
        user_elo=args.user_elo,
        bot_elo=args.bot_elo,
        user_color=args.color,
        start_fen=args.fen,
    )
