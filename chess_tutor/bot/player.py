"""ELO-conditioned chess bot with optional KL regularization."""

import chess
import numpy as np

from ..utils.helpers import softmax, sample_from_distribution
from ..data.extract_features import extract_board_features, detect_game_phase


class ChessTutorBot:
    """Chess bot that plays at a specified ELO level with optional KL regularization."""

    def __init__(self, move_predictor=None, stockfish_evaluator=None,
                 kl_lambda: float = 0.0):
        self.move_predictor = move_predictor
        self.stockfish_evaluator = stockfish_evaluator
        self.kl_lambda = kl_lambda

    def play_move(self, board: chess.Board, target_elo: int,
                  temperature: float = 1.0) -> chess.Move:
        """Select a move at the target ELO level."""
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise ValueError("No legal moves")

        if len(legal_moves) == 1:
            return legal_moves[0]

        # Get human move probabilities
        move_probs = self._get_human_probs(board, legal_moves, target_elo)

        # KL regularization with engine
        if self.kl_lambda > 0 and self.stockfish_evaluator is not None:
            engine_scores = self._get_engine_scores(board, legal_moves)
            # pi_bot ∝ pi_human × exp(λ × Q_engine)
            log_probs = np.log(move_probs + 1e-10) + self.kl_lambda * engine_scores
            move_probs = softmax(log_probs, temperature=temperature)
        elif temperature != 1.0:
            move_probs = softmax(np.log(move_probs + 1e-10), temperature=temperature)

        idx = sample_from_distribution(move_probs)
        return legal_moves[idx]

    def _get_human_probs(self, board, legal_moves, target_elo):
        """Get P(move | position, ELO) for each legal move."""
        n = len(legal_moves)
        probs = np.ones(n) / n  # uniform default

        if self.move_predictor is not None:
            try:
                features = extract_board_features(board).reshape(1, -1)
                full_proba = self.move_predictor.predict_proba(features, target_elo)
                if full_proba.ndim == 2:
                    full_proba = full_proba[0]
                # Map to legal moves (use top-k overlap)
                top_indices = np.argsort(full_proba)[::-1]
                for i, move in enumerate(legal_moves):
                    if i < len(full_proba):
                        probs[i] = max(full_proba[i], 1e-6)
                probs /= probs.sum()
            except Exception:
                pass

        # Add heuristic bias based on ELO
        skill = (target_elo - 800) / 1200
        for i, move in enumerate(legal_moves):
            if board.is_capture(move):
                probs[i] *= 1.0 + 0.5 * (1 - skill)  # lower ELO: more capture-happy
            if board.gives_check(move):
                probs[i] *= 1.0 + 0.3 * (1 - skill)

        probs /= probs.sum()
        return probs

    def _get_engine_scores(self, board, legal_moves):
        """Get Stockfish evaluation for each legal move (normalized)."""
        scores = np.zeros(len(legal_moves))
        for i, move in enumerate(legal_moves):
            board_after = board.copy()
            board_after.push(move)
            try:
                result = self.stockfish_evaluator.evaluate(board_after)
                # Negate because eval is from opponent's perspective after push
                sign = -1 if board.turn == chess.WHITE else 1
                scores[i] = sign * result["score_cp"] / 100.0
            except Exception:
                scores[i] = 0.0
        # Normalize
        if scores.std() > 0:
            scores = (scores - scores.mean()) / scores.std()
        return scores

    def evaluate_position(self, board: chess.Board, target_elo: int) -> dict:
        """Evaluate position from the perspective of a player at target_elo."""
        features = extract_board_features(board)
        material = features[12]
        mobility = features[13]
        phase = detect_game_phase(board)

        # Assessment
        if abs(material) < 100:
            assessment = "equal"
        elif material > 300:
            assessment = "winning" if board.turn == chess.WHITE else "losing"
        elif material < -300:
            assessment = "losing" if board.turn == chess.WHITE else "winning"
        else:
            assessment = "unclear"

        # Key features
        key_features = []
        if abs(material) > 100:
            key_features.append(f"Material: {'White' if material > 0 else 'Black'} ahead by {abs(material):.0f}cp")
        if mobility > 30:
            key_features.append("High mobility — many options available")
        elif mobility < 15:
            key_features.append("Limited mobility — cramped position")
        if features[29] > 0:
            key_features.append(f"{int(features[29])} hanging piece(s)")

        # Plan suggestion
        plans = {
            "opening": "Focus on development: get your pieces out and control the center.",
            "middlegame": "Look for tactical opportunities and improve your worst-placed piece.",
            "endgame": "Activate your king and push passed pawns.",
        }
        suggested_plan = plans.get(phase, "Evaluate the position and find the best plan.")

        # Move predictions
        move_predictions = []
        legal_moves = list(board.legal_moves)
        if self.move_predictor is not None:
            probs = self._get_human_probs(board, legal_moves, target_elo)
            top_indices = np.argsort(probs)[::-1][:3]
            for idx in top_indices:
                move_predictions.append(board.san(legal_moves[idx]))
        else:
            for m in legal_moves[:3]:
                move_predictions.append(board.san(m))

        return {
            "assessment": assessment,
            "confidence": min(abs(material) / 500.0, 1.0),
            "key_features": key_features,
            "suggested_plan": suggested_plan,
            "move_predictions": move_predictions,
        }
