"""Simulated chess student for offline evaluation of teaching policies."""

import chess
import numpy as np

from ..student.model import StudentState
from ..feedback.taxonomy import FeedbackType, FEEDBACK_CONCEPT_MAP
from ..config import STUDENT_LEARNING_RATE, STUDENT_BASE_CONCEPTS, BLUNDER_THRESHOLD_CP


class StudentSimulator:
    """Simulates a chess student for offline evaluation of teaching policies."""

    def __init__(self, elo: int, weakness_profile: dict[str, float] | None = None,
                 move_predictor=None, learning_rate: float = STUDENT_LEARNING_RATE):
        self._state = StudentState(
            elo=elo,
            weakness_profile=weakness_profile or self._default_profile(elo),
        )
        self.move_predictor = move_predictor
        self.lr = learning_rate

    @staticmethod
    def _default_profile(elo: int) -> dict[str, float]:
        """Generate a random weakness profile correlated with ELO."""
        base = (elo - 800) / 1200
        return {
            "tactics": float(np.clip(base + np.random.normal(0, 0.15), 0, 1)),
            "strategy": float(np.clip(base - 0.1 + np.random.normal(0, 0.15), 0, 1)),
            "endgame": float(np.clip(base - 0.05 + np.random.normal(0, 0.15), 0, 1)),
            "opening": float(np.clip(base + np.random.normal(0, 0.1), 0, 1)),
            "calculation": float(np.clip(base + np.random.normal(0, 0.15), 0, 1)),
        }

    def respond_to_position(self, board: chess.Board,
                            feedback_type: int | None = None,
                            feedback_text: str | None = None) -> chess.Move:
        """Student plays a move, possibly influenced by feedback."""
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise ValueError("No legal moves available")

        base_move = self._sample_move(board, legal_moves)

        if feedback_type is not None:
            ft = FeedbackType(feedback_type)
            concepts = FEEDBACK_CONCEPT_MAP.get(ft, [])
            if concepts:
                mastery = np.mean([self._state.weakness_profile.get(c, 0.5) for c in concepts])
            else:
                mastery = 0.5

            relevance = self._compute_relevance(ft, board)
            p_learn = self.lr * relevance * (1 - mastery)

            if np.random.random() < p_learn:
                # Upgrade to a better move
                base_move = self._sample_better_move(board, legal_moves)

        self._state.moves_played += 1
        return base_move

    def _sample_move(self, board: chess.Board, legal_moves: list[chess.Move]) -> chess.Move:
        """Sample a move based on the student's ELO level."""
        if self.move_predictor is not None:
            try:
                from ..data.extract_features import extract_board_features
                features = extract_board_features(board).reshape(1, -1)
                proba = self.move_predictor.predict_proba(features, self._state.elo)
                if proba.ndim == 2:
                    proba = proba[0]
                # Map to legal moves (simplified)
                idx = np.random.choice(len(proba), p=proba / proba.sum())
                if idx < len(legal_moves):
                    return legal_moves[idx]
            except Exception:
                pass

        # Fallback: weighted random based on simple heuristics
        weights = np.ones(len(legal_moves))

        for i, move in enumerate(legal_moves):
            if board.is_capture(move):
                weights[i] += 1.0
            if board.gives_check(move):
                weights[i] += 0.5
            # Center moves preferred
            to_file = chess.square_file(move.to_square)
            to_rank = chess.square_rank(move.to_square)
            center_dist = abs(to_file - 3.5) + abs(to_rank - 3.5)
            weights[i] += max(0, 3 - center_dist) * 0.3

        # Higher ELO = more likely to pick good moves
        skill_factor = (self._state.elo - 800) / 1200
        weights = weights ** (1 + skill_factor)
        weights /= weights.sum()

        return legal_moves[np.random.choice(len(legal_moves), p=weights)]

    def _sample_better_move(self, board: chess.Board,
                            legal_moves: list[chess.Move]) -> chess.Move:
        """Sample a better-than-average move (learning effect)."""
        if self.move_predictor is not None:
            try:
                from ..data.extract_features import extract_board_features
                features = extract_board_features(board).reshape(1, -1)
                better_elo = min(self._state.elo + 200, 1900)
                proba = self.move_predictor.predict_proba(features, better_elo)
                if proba.ndim == 2:
                    proba = proba[0]
                idx = np.random.choice(len(proba), p=proba / proba.sum())
                if idx < len(legal_moves):
                    return legal_moves[idx]
            except Exception:
                pass

        # Fallback: pick captures and checks preferentially
        good_moves = [m for m in legal_moves if board.is_capture(m) or board.gives_check(m)]
        if good_moves:
            return np.random.choice(good_moves)
        return np.random.choice(legal_moves)

    def _compute_relevance(self, feedback_type: FeedbackType, board: chess.Board) -> float:
        """How relevant the feedback is to the current position."""
        from ..data.extract_features import detect_game_phase

        phase = detect_game_phase(board)
        concepts = FEEDBACK_CONCEPT_MAP.get(feedback_type, [])

        relevance = 0.5  # base

        # Check if feedback matches position characteristics
        has_tactics = sum(1 for m in board.legal_moves if board.is_capture(m)) > 2
        if "tactics" in concepts and has_tactics:
            relevance += 0.3
        if "endgame" in concepts and phase == "endgame":
            relevance += 0.3
        if "opening" in concepts and phase == "opening":
            relevance += 0.2
        if "strategy" in concepts:
            relevance += 0.1

        return min(relevance, 1.0)

    def update_state(self, feedback_type: int, move_quality: float,
                     position_concepts: list[str]) -> None:
        """Update student's internal state after a teaching interaction."""
        ft = FeedbackType(feedback_type)
        feedback_concepts = FEEDBACK_CONCEPT_MAP.get(ft, [])

        # Improve relevant concepts
        for concept in feedback_concepts:
            if concept in self._state.weakness_profile:
                improvement = self.lr * (1 - self._state.weakness_profile[concept])
                if move_quality < 50:  # Good move
                    improvement *= 2
                self._state.weakness_profile[concept] = min(
                    1.0, self._state.weakness_profile[concept] + improvement
                )

        # Track cp loss
        self._state.recent_cp_losses.append(move_quality)
        if len(self._state.recent_cp_losses) > 50:
            self._state.recent_cp_losses = self._state.recent_cp_losses[-50:]

        if move_quality > BLUNDER_THRESHOLD_CP:
            self._state.blunder_count += 1

        # Update ELO based on performance
        if len(self._state.recent_cp_losses) >= 10:
            avg_cp = np.mean(self._state.recent_cp_losses[-10:])
            if avg_cp < 30:
                self._state.elo = min(2200, self._state.elo + 2)
            elif avg_cp > 100:
                self._state.elo = max(800, self._state.elo - 1)

        self._state.update_trend()

    @property
    def state(self) -> StudentState:
        return self._state

    @property
    def elo(self) -> int:
        return self._state.elo

    @property
    def weakness_profile(self) -> dict[str, float]:
        return self._state.weakness_profile


class StudentPopulation:
    """Generate a population of diverse simulated students."""

    @staticmethod
    def generate(n_students: int = 50,
                 elo_range: tuple[int, int] = (1000, 2000),
                 random_state: int = 42,
                 move_predictor=None) -> list[StudentSimulator]:
        """Generate diverse students with varying ELOs and weakness profiles."""
        rng = np.random.RandomState(random_state)
        students = []

        for i in range(n_students):
            elo = int(rng.uniform(elo_range[0], elo_range[1]))
            np.random.seed(random_state + i)
            student = StudentSimulator(
                elo=elo,
                move_predictor=move_predictor,
                learning_rate=rng.uniform(0.005, 0.02),
            )
            students.append(student)

        return students
