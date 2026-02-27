"""Generate natural language feedback for chess positions."""

import random
import chess

from .taxonomy import FeedbackType, FEEDBACK_CONCEPT_MAP
from .templates import TEMPLATES, get_elo_tier
from ..config import BLUNDER_THRESHOLD_CP


class FeedbackGenerator:
    """Generate natural language feedback for a position + student ELO."""

    def __init__(self, blunder_detector=None, complexity_estimator=None,
                 stockfish_evaluator=None):
        self.blunder_detector = blunder_detector
        self.complexity_estimator = complexity_estimator
        self.stockfish_evaluator = stockfish_evaluator

    def generate(self, board: chess.Board, student_elo: int,
                 feedback_type: FeedbackType,
                 student_move: chess.Move | None = None,
                 engine_eval: dict | None = None) -> str:
        """Generate feedback text."""
        tier = get_elo_tier(student_elo)
        templates = TEMPLATES.get(feedback_type, {}).get(tier, [])

        if not templates:
            templates = TEMPLATES.get(feedback_type, {}).get("intermediate", [
                "Consider the position carefully."
            ])

        template = random.choice(templates)

        # Build context for template filling
        context = self._build_template_context(board, student_elo, feedback_type,
                                                student_move, engine_eval)

        # Safe format: only fill placeholders that exist
        try:
            return template.format_map(_SafeDict(context))
        except Exception:
            return template

    def _build_template_context(self, board, student_elo, feedback_type,
                                 student_move, engine_eval):
        context = {
            "piece": "piece",
            "square": "a key square",
            "tactic_type": "tactical opportunity",
            "observation": "the position has imbalances",
            "suggestion": "improving piece placement",
            "concept": "piece activity",
            "explanation": "it changes the evaluation",
            "alternative": "a developing move",
            "threat": "a tactical threat",
            "cp_loss": "significant",
            "line": "the main line",
            "pattern": "typical middlegame",
            "reference": "a classic game",
            "student_move": "",
            "engine_move": "",
            "eval_student": "",
            "eval_engine": "",
            "reason": "controlling key squares",
            "endgame_type": "simplified endgame",
        }

        if student_move is not None:
            context["student_move"] = board.san(student_move)

        if engine_eval is not None:
            if "best_move" in engine_eval and engine_eval["best_move"]:
                try:
                    best = chess.Move.from_uci(engine_eval["best_move"])
                    context["engine_move"] = board.san(best)
                except Exception:
                    context["engine_move"] = engine_eval["best_move"]
            if "score_cp" in engine_eval:
                context["eval_engine"] = f"{engine_eval['score_cp']:+.0f}cp"

        # Position analysis
        analysis = self.analyze_position_for_feedback(board, engine_eval or {})
        if analysis.get("has_tactic"):
            context["tactic_type"] = analysis.get("tactic_type", "tactic")

        # Piece counts for observation
        mobility = board.legal_moves.count()
        if mobility < 15:
            context["observation"] = "your pieces are somewhat cramped"
            context["suggestion"] = "finding a way to free your position"
        elif mobility > 30:
            context["observation"] = "you have lots of options"
            context["suggestion"] = "choosing the most forcing continuation"

        return context

    def select_best_feedback_type(self, board: chess.Board,
                                   student_elo: int,
                                   engine_eval: dict) -> FeedbackType:
        """Heuristic feedback selection (baseline for bandit comparison)."""
        analysis = self.analyze_position_for_feedback(board, engine_eval)

        if analysis["has_tactic"]:
            return FeedbackType.TACTICAL_ALERT

        if analysis["blunder_prob"] > 0.5:
            return FeedbackType.BLUNDER_WARNING

        if analysis["complexity"] > 0.7:
            return FeedbackType.SIMPLIFICATION

        if analysis.get("position_type") == "endgame":
            return FeedbackType.PATTERN_RECOGNITION

        return FeedbackType.STRATEGIC_NUDGE

    def analyze_position_for_feedback(self, board: chess.Board,
                                       engine_eval: dict) -> dict:
        """Analyze position to determine what feedback is relevant."""
        result = {
            "has_tactic": False,
            "tactic_type": None,
            "blunder_prob": 0.0,
            "complexity": 0.5,
            "critical_squares": [],
            "key_pieces": [],
            "position_type": "positional",
        }

        # Detect tactics via eval swing
        score = engine_eval.get("score_cp", 0)
        if abs(score) > 200:
            result["has_tactic"] = True
            result["tactic_type"] = "winning combination"
            result["position_type"] = "tactical"

        # Check for checks and captures available
        checks = sum(1 for m in board.legal_moves if board.gives_check(m))
        captures = sum(1 for m in board.legal_moves if board.is_capture(m))
        if checks > 0 or captures > 2:
            result["complexity"] = min(1.0, 0.5 + checks * 0.1 + captures * 0.05)
            if checks > 0:
                result["position_type"] = "tactical"

        # Use blunder detector if available
        if self.blunder_detector is not None:
            from ..data.extract_features import extract_board_features
            features = extract_board_features(board).reshape(1, -1)
            # Pad to 40 features if needed
            import numpy as np
            if features.shape[1] < 40:
                features = np.pad(features, ((0, 0), (0, 40 - features.shape[1])))
            result["blunder_prob"] = float(self.blunder_detector.predict_proba(features)[0])

        # Game phase
        from ..data.extract_features import detect_game_phase
        phase = detect_game_phase(board)
        if phase == "endgame":
            result["position_type"] = "endgame"

        return result


class _SafeDict(dict):
    """Dict that returns the key itself for missing keys in format_map."""
    def __missing__(self, key):
        return f"{{{key}}}"
