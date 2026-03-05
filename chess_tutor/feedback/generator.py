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
        from ..data.extract_features import extract_board_features, detect_game_phase

        features = extract_board_features(board)
        phase = detect_game_phase(board)
        mobility = board.legal_moves.count()
        material = features[12]  # material_balance_cp

        # Find concrete piece and square info from the position
        piece_name, square_name_str = self._find_key_piece_and_square(board)
        tactic_type = self._detect_tactic_type(board)
        observation = self._observe_position(board, features, phase, mobility, material)
        suggestion = self._suggest_plan(board, features, phase, mobility, material, student_elo)
        concept = self._identify_concept(board, features, phase)
        threat = self._find_threat(board)

        context = {
            "piece": piece_name,
            "square": square_name_str,
            "tactic_type": tactic_type,
            "observation": observation,
            "suggestion": suggestion,
            "concept": concept,
            "explanation": self._explain_position(board, features, material, phase),
            "alternative": self._suggest_alternative(board),
            "threat": threat,
            "cp_loss": f"{abs(material):.0f}" if abs(material) > 50 else "small",
            "line": self._get_key_line(board),
            "pattern": self._identify_pattern(board, phase),
            "reference": "classical play",
            "student_move": "",
            "engine_move": "",
            "eval_student": "",
            "eval_engine": "",
            "reason": concept,
            "endgame_type": self._classify_endgame(board) if phase == "endgame" else "simplified position",
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

        return context

    def _find_key_piece_and_square(self, board):
        """Find a relevant piece and square to mention in feedback."""
        # Look for undefended pieces
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece and piece.color == board.turn:
                if not board.is_attacked_by(board.turn, sq) and board.is_attacked_by(not board.turn, sq):
                    return chess.piece_name(piece.piece_type), chess.square_name(sq)

        # Look for pieces that can capture
        for move in board.legal_moves:
            if board.is_capture(move):
                piece = board.piece_at(move.from_square)
                if piece:
                    return chess.piece_name(piece.piece_type), chess.square_name(move.to_square)

        # Default: most active piece
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece and piece.color == board.turn and piece.piece_type not in (chess.PAWN, chess.KING):
                return chess.piece_name(piece.piece_type), chess.square_name(sq)

        return "piece", "a central square"

    def _detect_tactic_type(self, board):
        """Identify what type of tactic might be available."""
        checks = [m for m in board.legal_moves if board.gives_check(m)]
        captures = [m for m in board.legal_moves if board.is_capture(m)]

        if checks:
            # Check if any check also attacks another piece
            for m in checks:
                board.push(m)
                if board.is_checkmate():
                    board.pop()
                    return "checkmate"
                board.pop()
            return "discovered attack" if len(checks) > 1 else "check"

        if len(captures) > 2:
            return "combination"

        # Check for forks (piece attacking multiple targets)
        for m in board.legal_moves:
            piece = board.piece_at(m.from_square)
            if piece and piece.piece_type == chess.KNIGHT:
                board.push(m)
                attacks = sum(1 for sq in board.attacks(m.to_square)
                             if board.piece_at(sq) and board.piece_at(sq).color != piece.color
                             and board.piece_at(sq).piece_type in (chess.QUEEN, chess.ROOK))
                board.pop()
                if attacks >= 2:
                    return "knight fork"

        types = ["pin", "skewer", "tactical opportunity", "forcing sequence"]
        return random.choice(types)

    def _observe_position(self, board, features, phase, mobility, material):
        """Generate a position-specific observation."""
        observations = []

        if abs(material) > 300:
            side = "White" if material > 0 else "Black"
            observations.append(f"{side} is ahead by {abs(material):.0f} centipawns")
        if mobility < 15:
            observations.append("your pieces are somewhat cramped")
        elif mobility > 35:
            observations.append("you have excellent piece activity")
        if features[17] > 0:  # doubled pawns
            observations.append("there are doubled pawns weakening the structure")
        if features[16] > 0:  # isolated pawns
            observations.append("there's an isolated pawn that could be a target")
        if features[19] > 0:  # passed pawns
            observations.append("there's a passed pawn that could be dangerous")
        if features[29] > 0:  # hanging pieces
            observations.append("there are undefended pieces on the board")
        if features[14] < 2:  # king safety
            observations.append("the king position looks a bit exposed")

        if not observations:
            observations = [
                "the position is roughly balanced",
                "both sides have active pieces",
                "the pawn structure is symmetrical",
            ]
        return random.choice(observations)

    def _suggest_plan(self, board, features, phase, mobility, material, elo):
        """Generate a position-specific suggestion."""
        suggestions = []

        if phase == "opening":
            if features[21] < 3:  # development
                suggestions.append("developing your remaining minor pieces")
            if board.has_castling_rights(board.turn):
                suggestions.append("castling to bring your king to safety")
            suggestions.append("controlling the center with pawns")
        elif phase == "endgame":
            suggestions.append("activating your king as a fighting piece")
            if features[19] > 0:
                suggestions.append("pushing your passed pawn")
            suggestions.append("centralizing your remaining pieces")
        else:
            if mobility < 20:
                suggestions.append("finding a pawn break to open the position")
            if features[29] > 0:
                suggestions.append("protecting your hanging pieces first")
            suggestions.append("improving your least active piece")
            suggestions.append("looking for a favorable exchange")

        if material > 200:
            suggestions.append("simplifying by trading pieces to convert your advantage")

        return random.choice(suggestions) if suggestions else "improving piece placement"

    def _identify_concept(self, board, features, phase):
        """Identify the key strategic concept in the position."""
        concepts = []
        if features[13] > 30:  # mobility
            concepts.append("piece activity and mobility")
        if features[12] != 0:
            concepts.append("material imbalance")
        if features[16] > 0 or features[17] > 0:
            concepts.append("pawn structure weaknesses")
        if features[14] < 2:
            concepts.append("king safety")
        if features[15] > 2:
            concepts.append("attacking potential near the king")
        if features[19] > 0:
            concepts.append("passed pawn advancement")
        if phase == "opening":
            concepts.append("development and center control")
        elif phase == "endgame":
            concepts.append("king activity in the endgame")

        return random.choice(concepts) if concepts else "positional understanding"

    def _find_threat(self, board):
        """Find a concrete threat in the position."""
        # Check opponent's threats
        board_copy = board.copy()
        board_copy.push(chess.Move.null())
        for m in board_copy.legal_moves:
            if board_copy.is_capture(m):
                captured = board_copy.piece_at(m.to_square)
                attacker = board_copy.piece_at(m.from_square)
                if captured and attacker:
                    if captured.piece_type > attacker.piece_type:
                        return f"winning the {chess.piece_name(captured.piece_type)} on {chess.square_name(m.to_square)}"
            if board_copy.gives_check(m):
                return f"a check with {chess.piece_name(board_copy.piece_at(m.from_square).piece_type) if board_copy.piece_at(m.from_square) else 'a piece'}"

        return "a tactical threat against your position"

    def _explain_position(self, board, features, material, phase):
        """Generate position explanation."""
        if abs(material) > 300:
            return f"the material balance favors {'White' if material > 0 else 'Black'} significantly"
        if features[29] > 0:
            return "there are hanging pieces that need attention"
        if features[13] > 35:
            return "active pieces give a dynamic advantage"
        if phase == "endgame":
            return "endgame technique is critical here"
        return "subtle positional factors determine the best plan"

    def _suggest_alternative(self, board):
        """Suggest a concrete alternative move."""
        good_moves = []
        for m in board.legal_moves:
            if board.gives_check(m):
                good_moves.append(board.san(m))
            elif board.is_capture(m):
                good_moves.append(board.san(m))
        if good_moves:
            return random.choice(good_moves[:3])

        # Suggest a developing move
        for m in board.legal_moves:
            piece = board.piece_at(m.from_square)
            if piece and piece.piece_type in (chess.KNIGHT, chess.BISHOP):
                return board.san(m)
        return "a more active move"

    def _get_key_line(self, board):
        """Get a short key continuation."""
        moves = []
        b = board.copy()
        for _ in range(3):
            legal = list(b.legal_moves)
            if not legal:
                break
            # Pick captures/checks first, then random
            forcing = [m for m in legal if b.is_capture(m) or b.gives_check(m)]
            m = random.choice(forcing) if forcing else random.choice(legal)
            moves.append(b.san(m))
            b.push(m)
        return " ".join(moves) if moves else "the continuation"

    def _identify_pattern(self, board, phase):
        """Identify a pattern name for the position."""
        if phase == "opening":
            patterns = ["development", "center control", "opening principle"]
        elif phase == "endgame":
            n_pieces = len(board.piece_map())
            if n_pieces <= 6:
                patterns = ["basic endgame", "king and pawn", "minor piece endgame"]
            else:
                patterns = ["complex endgame", "rook endgame", "queen endgame"]
        else:
            checks = sum(1 for m in board.legal_moves if board.gives_check(m))
            if checks > 0:
                patterns = ["attacking", "tactical middlegame", "kingside attack"]
            else:
                patterns = ["positional middlegame", "maneuvering", "strategic play"]
        return random.choice(patterns)

    def _classify_endgame(self, board):
        """Classify the type of endgame."""
        pieces = board.piece_map()
        has_rook = any(p.piece_type == chess.ROOK for p in pieces.values())
        has_queen = any(p.piece_type == chess.QUEEN for p in pieces.values())
        has_minor = any(p.piece_type in (chess.KNIGHT, chess.BISHOP) for p in pieces.values())

        if has_queen:
            return "queen endgame"
        elif has_rook:
            return "rook endgame"
        elif has_minor:
            return "minor piece endgame"
        else:
            return "king and pawn endgame"

    def select_best_feedback_type(self, board: chess.Board,
                                   student_elo: int,
                                   engine_eval: dict) -> FeedbackType:
        """Heuristic feedback selection — varies based on position features."""
        from ..data.extract_features import extract_board_features, detect_game_phase

        features = extract_board_features(board)
        phase = detect_game_phase(board)
        mobility = board.legal_moves.count()
        material = features[12]
        checks = sum(1 for m in board.legal_moves if board.gives_check(m))
        captures = sum(1 for m in board.legal_moves if board.is_capture(m))

        # Score each feedback type based on position
        scores = {}

        # Tactical alert: when there are checks, captures, or big material swings
        scores[FeedbackType.TACTICAL_ALERT] = checks * 3 + max(0, captures - 1) * 2
        if abs(engine_eval.get("score_cp", 0)) > 200:
            scores[FeedbackType.TACTICAL_ALERT] += 5

        # Strategic nudge: quiet positions
        scores[FeedbackType.STRATEGIC_NUDGE] = max(0, 3 - checks * 2 - captures)
        if phase == "middlegame" and mobility > 20:
            scores[FeedbackType.STRATEGIC_NUDGE] += 2

        # Blunder warning: when there are hanging pieces or exposed king
        scores[FeedbackType.BLUNDER_WARNING] = 0
        if features[29] > 0:  # hanging pieces
            scores[FeedbackType.BLUNDER_WARNING] += 4
        if features[14] < 2:  # king safety
            scores[FeedbackType.BLUNDER_WARNING] += 2

        # Pattern recognition: endgames and known structures
        scores[FeedbackType.PATTERN_RECOGNITION] = 0
        if phase == "endgame":
            scores[FeedbackType.PATTERN_RECOGNITION] += 4
        if phase == "opening":
            scores[FeedbackType.PATTERN_RECOGNITION] += 2

        # Encouragement: when player has advantage
        scores[FeedbackType.ENCOURAGEMENT] = 0
        if material > 100 and board.turn == chess.WHITE:
            scores[FeedbackType.ENCOURAGEMENT] += 3
        elif material < -100 and board.turn == chess.BLACK:
            scores[FeedbackType.ENCOURAGEMENT] += 3

        # Simplification: when ahead in material
        scores[FeedbackType.SIMPLIFICATION] = 0
        if abs(material) > 200:
            scores[FeedbackType.SIMPLIFICATION] += 3

        # Move comparison: general fallback
        scores[FeedbackType.MOVE_COMPARISON] = 1

        # Add randomness to avoid always picking the same
        for ft in scores:
            scores[ft] += random.random() * 2

        return max(scores, key=scores.get)

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

        score = engine_eval.get("score_cp", 0)
        if abs(score) > 200:
            result["has_tactic"] = True
            result["tactic_type"] = "winning combination"
            result["position_type"] = "tactical"

        checks = sum(1 for m in board.legal_moves if board.gives_check(m))
        captures = sum(1 for m in board.legal_moves if board.is_capture(m))
        if checks > 0 or captures > 2:
            result["complexity"] = min(1.0, 0.5 + checks * 0.1 + captures * 0.05)
            if checks > 0:
                result["position_type"] = "tactical"

        if self.blunder_detector is not None:
            from ..data.extract_features import extract_board_features
            import numpy as np
            feat = extract_board_features(board).reshape(1, -1)
            if feat.shape[1] < 40:
                feat = np.pad(feat, ((0, 0), (0, 40 - feat.shape[1])))
            result["blunder_prob"] = float(self.blunder_detector.predict_proba(feat)[0])

        from ..data.extract_features import detect_game_phase
        phase = detect_game_phase(board)
        if phase == "endgame":
            result["position_type"] = "endgame"

        return result


class _SafeDict(dict):
    """Dict that returns the key itself for missing keys in format_map."""
    def __missing__(self, key):
        return f"{{{key}}}"
