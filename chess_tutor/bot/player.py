"""ELO-conditioned chess bot with optional KL regularization."""

import chess
import numpy as np

from ..utils.helpers import softmax, sample_from_distribution
from ..data.extract_features import extract_board_features, detect_game_phase
from ..config import PIECE_VALUES


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
        """Estimate P(move | position, ELO) over legal moves.

        When ``self.move_predictor`` exposes a ``predict_move_probs`` method
        (i.e. it is a ``CandidateRankingPredictor``), we delegate to the
        trained RF candidate-ranking classifier.  A mild ELO-dependent
        temperature is still applied so that low-ELO distributions remain
        flatter than high-ELO ones even when the RF is used.

        Falls back to the hand-crafted heuristic when no model is available.
        """
        # --- Data-driven path (trained RF) ---
        if self.move_predictor is not None and hasattr(self.move_predictor, "predict_move_probs"):
            try:
                probs = self.move_predictor.predict_move_probs(board, legal_moves, target_elo)
                # Apply ELO-dependent temperature to the log-probs so that
                # low-ELO play stays noisier than high-ELO play even though
                # the RF already uses ELO-specific bracket models.
                skill = float(np.clip((target_elo - 1100) / 800.0, 0.0, 1.0))
                temperature = max(0.4, 2.0 - 1.6 * skill)
                return softmax(np.log(probs + 1e-10), temperature=temperature)
            except Exception:
                pass  # fall through to heuristic

        # --- Heuristic fallback ---
        # Used when no trained model is available.  10 hand-crafted rules
        # weighted by skill level replicate coarse ELO-dependent behaviour.
        skill = float(np.clip((target_elo - 1100) / 800.0, 0.0, 1.0))
        phase = detect_game_phase(board)

        scores = np.zeros(len(legal_moves))
        for i, move in enumerate(legal_moves):
            scores[i] = self._score_move(board, move, skill, phase)

        # Temperature: 2.0 at ELO 1100, 0.4 at ELO 1900
        temperature = max(0.4, 2.0 - 1.6 * skill)

        return softmax(scores, temperature=temperature)

    def _score_move(self, board, move, skill, phase):
        """Score a move from the side-to-move's perspective.

        Weights for each feature are ELO-conditioned via ``skill``
        (0 = beginner, 1 = expert). Positive score => more likely chosen.
        """
        score = 0.0
        piece = board.piece_at(move.from_square)
        to_sq = move.to_square

        # Simulate the move once to query attacker/defender bitboards
        after = board.copy()
        after.push(move)
        attackers_after = after.attackers(not board.turn, to_sq)
        defenders_after = after.attackers(board.turn, to_sq)

        our_val = PIECE_VALUES.get(piece.piece_type, 0) / 100.0 if piece else 0

        # 1. Captures -- beginners take anything, experts evaluate trades
        if board.is_capture(move):
            captured = board.piece_at(to_sq)
            # en-passant captures a pawn on a different square
            cap_val = PIECE_VALUES.get(captured.piece_type, 100) / 100.0 if captured else 1.0
            if defenders_after or not attackers_after:
                # Safe capture: everyone wants it, weak slight bonus for beginners
                score += cap_val * (1.5 + 0.3 * (1 - skill))
            else:
                # Losing/even trade (capture but piece will be recaptured)
                net = cap_val - our_val
                # Beginners still attracted even if net<0; experts refuse
                score += net * (0.3 + 2.5 * skill) + cap_val * 0.5 * (1 - skill)

        # 2. Checks -- beginners love them even when tactically pointless
        if board.gives_check(move):
            score += 1.2 + 1.0 * (1 - skill)

        # 3. Dropping a piece to an attacked undefended square (blunder)
        if piece and not board.is_capture(move):
            if attackers_after and not defenders_after:
                # About to hang the piece
                score -= our_val * (0.2 + 2.5 * skill)

        # 4. Central control (central four squares)
        if to_sq in (chess.D4, chess.E4, chess.D5, chess.E5):
            score += 0.4 + 0.5 * skill

        # 5. Develop minor pieces in the opening (knights/bishops off home rank)
        if phase == "opening" and piece is not None:
            if piece.piece_type in (chess.KNIGHT, chess.BISHOP):
                home_rank = 0 if piece.color == chess.WHITE else 7
                if chess.square_rank(move.from_square) == home_rank:
                    score += 0.5 + 0.8 * skill

        # 6. Castling -- universally good, experts prioritize it more
        if board.is_castling(move):
            score += 1.2 + 1.2 * skill

        # 7. Early queen sorties (beginners do; experts know better)
        if phase == "opening" and piece is not None and piece.piece_type == chess.QUEEN:
            if board.fullmove_number < 8:
                score -= 0.6 * skill

        # 8. Moves toward board edge (knights/bishops on edge are weak)
        if piece is not None and piece.piece_type in (chess.KNIGHT, chess.BISHOP):
            to_file = chess.square_file(to_sq)
            to_rank = chess.square_rank(to_sq)
            if to_file in (0, 7) or (to_rank in (0, 7) and piece.piece_type == chess.KNIGHT):
                score -= 0.4 * skill

        # 9. Pawn pushes toward promotion (endgame experts especially)
        if phase == "endgame" and piece is not None and piece.piece_type == chess.PAWN:
            push_distance = abs(chess.square_rank(to_sq) - chess.square_rank(move.from_square))
            direction = 1 if piece.color == chess.WHITE else -1
            toward_promotion = direction * (chess.square_rank(to_sq) - chess.square_rank(move.from_square)) > 0
            if toward_promotion:
                score += 0.3 * push_distance * (0.5 + skill)

        # 10. King activity in endgame (experts activate, beginners leave king back)
        if phase == "endgame" and piece is not None and piece.piece_type == chess.KING:
            to_rank = chess.square_rank(to_sq)
            center_rank_dist = abs(to_rank - 3.5)
            # Moving king toward center in endgame is good
            from_rank_dist = abs(chess.square_rank(move.from_square) - 3.5)
            if center_rank_dist < from_rank_dist:
                score += 0.5 * skill

        return score

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

        # Plan suggestion — ELO-adaptive and position-aware
        import random
        if target_elo < 1300:
            tier = "beginner"
        elif target_elo < 1700:
            tier = "intermediate"
        else:
            tier = "advanced"

        suggested_plan = self._build_plan(
            phase=phase, tier=tier, board=board, features=features, material=material,
        )

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

    def _build_plan(self, phase: str, tier: str, board: chess.Board,
                    features: np.ndarray, material: float) -> str:
        """Generate a position-aware plan in ELO-appropriate language.

        Branches on game phase first, then refines by tier
        (beginner / intermediate / advanced) so beginners see plain-language
        plans while advanced players see positional chess vocabulary.
        """
        import random

        mobility = features[13]
        development_score = features[21]
        hanging_pieces = features[29]
        passed_pawns = features[19]

        if phase == "opening":
            plans: list[str] = []

            castle_ok = board.has_castling_rights(board.turn)
            underdeveloped = development_score < 3
            cramped = mobility < 20

            if tier == "beginner":
                if castle_ok:
                    plans.append("Castle soon to keep your king safe.")
                if underdeveloped:
                    plans.append("Get your knights and bishops out before moving the same pawn twice.")
                plans.append("Put a pawn in the center if you can.")
                if cramped:
                    plans.append("Your pieces need more room — try to trade or free them.")
            elif tier == "intermediate":
                if castle_ok:
                    plans.append("Complete development and castle before starting operations.")
                if underdeveloped:
                    plans.append("Develop your minor pieces to their best squares first.")
                plans.append("Control the center with pawns and pieces before committing to a plan.")
                if cramped:
                    plans.append("Look for a pawn break or piece exchange to relieve the cramp.")
            else:  # advanced
                if castle_ok:
                    plans.append("Castle to finish development, then evaluate whether to play on the queenside or center.")
                if underdeveloped:
                    plans.append("Finish development — every minor piece should have a concrete role before initiating.")
                plans.append("Control the center and make prophylactic moves against your opponent's setup.")
                if cramped:
                    plans.append("Typical strategy: exchange a piece or force a pawn break to release the tension.")

        elif phase == "middlegame":
            plans = []

            hanging = hanging_pieces > 0
            high_mobility = mobility > 30
            material_ahead = material > 200 if board.turn == chess.WHITE else material < -200

            if tier == "beginner":
                plans.append("Look for captures and checks on every move.")
                if hanging:
                    plans.append("You have an unprotected piece — defend it right now.")
                if high_mobility:
                    plans.append("You have many good moves — pick the one that wins material.")
                if material_ahead:
                    plans.append("You're ahead in pieces — trade to simplify and head for the endgame.")
            elif tier == "intermediate":
                plans.append("Improve your worst-placed piece and look for tactical opportunities.")
                if hanging:
                    plans.append("Your opponent may be threatening an undefended piece — check for tactics against you.")
                if high_mobility:
                    plans.append("You have initiative — find the most forcing continuation.")
                if material_ahead:
                    plans.append("Trade pieces, not pawns, when ahead — simplification wins.")
                plans.append("Consider a pawn break to open lines for your rooks.")
            else:  # advanced
                plans.append("Think about imbalances: bishop pair, pawn structure, king safety, piece activity.")
                if hanging:
                    plans.append("There's a hanging piece on the board — calculate concretely before committing.")
                if high_mobility:
                    plans.append("You have the initiative — convert it with a concrete plan (attack, pawn break, or file seizure).")
                if material_ahead:
                    plans.append("Simplify into a technically winning endgame rather than sharpen the tactics.")
                plans.append("Prophylactic moves can be stronger than direct attacks when no break is ready.")

        elif phase == "endgame":
            plans = []

            has_passed = passed_pawns > 0

            if tier == "beginner":
                plans.append("Bring your king toward the center — it's a strong piece now.")
                if has_passed:
                    plans.append("You have a passed pawn — push it toward promotion with king support.")
                plans.append("Count how many moves each pawn needs to promote.")
            elif tier == "intermediate":
                plans.append("Activate your king in the endgame — aim for the center or the opponent's weak pawns.")
                if has_passed:
                    plans.append("Support your passed pawn with the king; promote it with tempo if possible.")
                plans.append("Trade pieces, not pawns, when you're ahead; the opposite when you're behind.")
            else:  # advanced
                plans.append("Endgame technique: centralize the king, create an outside passed pawn, and remember zugzwang as a weapon.")
                if has_passed:
                    plans.append("Convert the passed pawn — use the king as an escort and watch for stalemate tricks.")
                plans.append("Think about opposition, triangulation, and the critical squares before committing to a king move.")
        else:
            plans = ["Evaluate the position and find a concrete plan."]

        return random.choice(plans) if plans else "Find a plan."
