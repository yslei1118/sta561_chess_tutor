"""Wrapper for Stockfish position evaluation."""

import chess
import chess.engine

from ..config import STOCKFISH_DEPTH, BLUNDER_THRESHOLD_CP, MISTAKE_THRESHOLD_CP, INACCURACY_THRESHOLD_CP


class StockfishEvaluator:
    """Wrapper for Stockfish position evaluation."""

    def __init__(self, stockfish_path: str = "stockfish", depth: int = STOCKFISH_DEPTH):
        self.stockfish_path = stockfish_path
        self.depth = depth
        self.engine = None

    def _ensure_engine(self):
        if self.engine is None:
            self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)

    def evaluate(self, board: chess.Board) -> dict:
        """Evaluate a position.

        Returns:
            dict with score_cp, best_move, best_line, is_mate, mate_in
        """
        self._ensure_engine()
        info = self.engine.analyse(board, chess.engine.Limit(depth=self.depth),
                                   multipv=1)
        score = info["score"].white()

        is_mate = score.is_mate()
        if is_mate:
            score_cp = 10000 * (1 if score.mate() > 0 else -1)
            mate_in = score.mate()
        else:
            score_cp = score.score()
            mate_in = None

        pv = info.get("pv", [])
        best_move = pv[0].uci() if pv else None
        best_line = [m.uci() for m in pv[:3]]

        return {
            "score_cp": float(score_cp) if score_cp is not None else 0.0,
            "best_move": best_move,
            "best_line": best_line,
            "is_mate": is_mate,
            "mate_in": mate_in,
        }

    def evaluate_move(self, board: chess.Board, move: chess.Move) -> dict:
        """Evaluate a specific move.

        Returns:
            dict with cp_loss, is_blunder, is_mistake, is_inaccuracy, eval_before, eval_after
        """
        eval_before_info = self.evaluate(board)
        eval_before = eval_before_info["score_cp"]

        board_after = board.copy()
        board_after.push(move)
        eval_after_info = self.evaluate(board_after)
        eval_after = eval_after_info["score_cp"]

        # CP loss from the perspective of the side that moved
        if board.turn == chess.WHITE:
            cp_loss = max(0.0, eval_before - eval_after)
        else:
            cp_loss = max(0.0, eval_after - eval_before)

        return {
            "cp_loss": cp_loss,
            "is_blunder": cp_loss > BLUNDER_THRESHOLD_CP,
            "is_mistake": cp_loss > MISTAKE_THRESHOLD_CP,
            "is_inaccuracy": cp_loss > INACCURACY_THRESHOLD_CP,
            "eval_before": eval_before,
            "eval_after": eval_after,
        }

    def batch_evaluate(self, positions: list[chess.Board],
                       batch_size: int = 100) -> list[dict]:
        """Evaluate multiple positions efficiently."""
        results = []
        for i in range(0, len(positions), batch_size):
            batch = positions[i:i + batch_size]
            for board in batch:
                results.append(self.evaluate(board))
        return results

    def close(self):
        """Close the engine process."""
        if self.engine is not None:
            self.engine.quit()
            self.engine = None

    def __enter__(self):
        self._ensure_engine()
        return self

    def __exit__(self, *args):
        self.close()
