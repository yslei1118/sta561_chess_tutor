"""Simulated chess student for offline evaluation of teaching policies."""

import chess
import numpy as np

from ..student.model import StudentState
from ..feedback.taxonomy import FeedbackType, FEEDBACK_CONCEPT_MAP
from ..config import STUDENT_LEARNING_RATE, STUDENT_BASE_CONCEPTS, BLUNDER_THRESHOLD_CP


class StudentSimulator:
    """Simulates a chess student for offline evaluation of teaching policies.

    ``concept_map_override`` lets callers inject a *different* FeedbackType →
    concepts mapping than the default in ``feedback.taxonomy``.  This exists
    specifically so that sanity-check experiments (see
    ``simulation.runner.run_sanity_check``) can randomize the mapping and
    verify that any bandit "learning" disappears when the signal is broken —
    the direct diagnostic for the self-referential-reward critique.
    """

    def __init__(self, elo: int, weakness_profile: dict[str, float] | None = None,
                 move_predictor=None, learning_rate: float = STUDENT_LEARNING_RATE,
                 concept_map_override: dict | None = None):
        self._state = StudentState(
            elo=elo,
            weakness_profile=weakness_profile or self._default_profile(elo),
        )
        self.move_predictor = move_predictor
        self.lr = learning_rate
        self._concept_map = concept_map_override or FEEDBACK_CONCEPT_MAP

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
            concepts = self._concept_map.get(ft, [])
            # Arms with no concept mapping (ENCOURAGEMENT) are pure affect:
            # they carry no pedagogical signal and must not trigger a move
            # upgrade, otherwise they get a free baseline learning effect.
            if concepts:
                mastery = np.mean([self._state.weakness_profile.get(c, 0.5) for c in concepts])
                relevance = self._compute_relevance(ft, board)
                p_learn = self.lr * relevance * (1 - mastery)

                if np.random.random() < p_learn:
                    base_move = self._sample_better_move(board, legal_moves)

        self._state.moves_played += 1
        return base_move

    def _sample_move(self, board: chess.Board, legal_moves: list[chess.Move]) -> chess.Move:
        """Sample a move based on the student's ELO level.

        When ``self.move_predictor`` is a ``CandidateRankingPredictor`` (exposes
        ``predict_move_probs``), sample from the trained RF distribution for
        the student's current ELO.  Falls back to the heuristic below when
        no model is available or the call fails.
        """
        if self.move_predictor is not None and hasattr(self.move_predictor, "predict_move_probs"):
            try:
                probs = self.move_predictor.predict_move_probs(
                    board, legal_moves, self._state.elo
                )
                return legal_moves[np.random.choice(len(legal_moves), p=probs)]
            except Exception:
                pass

        # Fallback: weighted random based on simple heuristics
        weights = np.ones(len(legal_moves))

        for i, move in enumerate(legal_moves):
            if board.is_capture(move):
                weights[i] += 1.0
            if board.gives_check(move):
                weights[i] += 0.5
            to_file = chess.square_file(move.to_square)
            to_rank = chess.square_rank(move.to_square)
            center_dist = abs(to_file - 3.5) + abs(to_rank - 3.5)
            weights[i] += max(0, 3 - center_dist) * 0.3

        skill_factor = (self._state.elo - 800) / 1200
        weights = weights ** (1 + skill_factor)
        weights /= weights.sum()

        return legal_moves[np.random.choice(len(legal_moves), p=weights)]

    def _sample_better_move(self, board: chess.Board,
                            legal_moves: list[chess.Move]) -> chess.Move:
        """Sample a better-than-average move (learning effect).

        When a ``CandidateRankingPredictor`` is available, simulates a player
        200 ELO points stronger than the current student (ZPD upper bound).
        """
        if self.move_predictor is not None and hasattr(self.move_predictor, "predict_move_probs"):
            try:
                better_elo = min(self._state.elo + 200, 1900)
                probs = self.move_predictor.predict_move_probs(
                    board, legal_moves, better_elo
                )
                return legal_moves[np.random.choice(len(legal_moves), p=probs)]
            except Exception:
                pass

        # Fallback: captures and checks only
        good_moves = [m for m in legal_moves if board.is_capture(m) or board.gives_check(m)]
        if good_moves:
            return good_moves[np.random.randint(len(good_moves))]
        return legal_moves[np.random.randint(len(legal_moves))]

    def _compute_relevance(self, feedback_type: FeedbackType, board: chess.Board) -> float:
        """How relevant the feedback is to the current position."""
        from ..data.extract_features import detect_game_phase

        phase = detect_game_phase(board)
        concepts = self._concept_map.get(feedback_type, [])

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
        feedback_concepts = self._concept_map.get(ft, [])

        # Forgetting: all concepts decay slightly each step.
        # Concepts not reinforced by the chosen feedback arm will drift back
        # toward zero over time, giving bandit policies that consistently
        # target the student's weak concepts a reward advantage.
        _DECAY = 0.998
        for c in self._state.weakness_profile:
            self._state.weakness_profile[c] = max(
                0.0, self._state.weakness_profile[c] * _DECAY
            )

        # Improve relevant concepts. ``quality_factor`` replaces the old
        # hard step ``if move_quality < 50: improvement *= 2``: a sigmoid
        # centered at 50cp gives ~2× learning on clean moves, ~1× on
        # noisy moves, with a smooth transition (no discontinuity at 50cp).
        quality_factor = 1.0 + 1.0 / (1.0 + np.exp((move_quality - 50.0) / 20.0))
        for concept in feedback_concepts:
            if concept in self._state.weakness_profile:
                improvement = self.lr * (1 - self._state.weakness_profile[concept])
                improvement *= quality_factor
                self._state.weakness_profile[concept] = min(
                    1.0, self._state.weakness_profile[concept] + improvement
                )

        # Track cp loss — keep a long enough history for the trend
        # detector (which needs ≥30 samples for its two-window test).
        self._state.recent_cp_losses.append(move_quality)
        if len(self._state.recent_cp_losses) > 60:
            self._state.recent_cp_losses = self._state.recent_cp_losses[-60:]

        if move_quality > BLUNDER_THRESHOLD_CP:
            self._state.blunder_count += 1

        # ELO update: drift current ELO toward the ELO *implied by* the
        # student's recent cp_loss.  This replaces the previous rule that
        # only granted +2 ELO when avg_cp < 30 — a threshold that was
        # almost never met by sub-1500 students, producing flat ELO
        # trajectories.  The implied-ELO mapping is calibrated so that
        # ~25cp → 1900, ~60cp → 1500, ~100cp → 1100 (roughly matching the
        # empirical Lichess Stockfish-labeled cp_loss distribution).
        if len(self._state.recent_cp_losses) >= 10:
            avg_cp = float(np.mean(self._state.recent_cp_losses[-10:]))
            implied_elo = float(np.clip(1900.0 - 20.0 * (avg_cp - 25.0), 800.0, 2200.0))
            # Exponential moving average toward implied ELO.  α=0.05 keeps
            # ELO responsive but not jittery (a 50-step window half-life).
            alpha = 0.05
            new_elo = (1 - alpha) * self._state.elo + alpha * implied_elo
            self._state.elo = int(np.clip(new_elo, 800, 2200))

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
                learning_rate=rng.uniform(0.02, 0.06),
            )
            students.append(student)

        return students
