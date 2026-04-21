"""Tests for chess_tutor.simulation.student_simulator — ZPD-based learning model."""

import chess
import numpy as np
import pytest

from chess_tutor.simulation.student_simulator import (
    StudentSimulator, StudentPopulation,
)
from chess_tutor.student.model import StudentState


class TestStudentState:
    def test_default_state_has_valid_mastery_range(self):
        state = StudentState(elo=1200)
        for concept, mastery in state.weakness_profile.items():
            assert 0.0 <= mastery <= 1.0

    def test_default_state_has_covered_concepts(self):
        state = StudentState(elo=1200)
        expected = {'tactics', 'strategy', 'endgame', 'opening', 'calculation'}
        assert set(state.weakness_profile.keys()) == expected

    def test_copy_isolates_mutations(self):
        s1 = StudentState(elo=1200)
        s2 = s1.copy()
        s2.elo = 1500
        s2.weakness_profile['tactics'] = 0.9
        assert s1.elo == 1200
        assert s1.weakness_profile['tactics'] != 0.9

    def test_trend_defaults_to_stable(self):
        s = StudentState(elo=1200)
        assert s.trend == 'stable'

    def test_trend_updates_with_recent_losses(self):
        s = StudentState(elo=1200)
        # 30 samples — older 15 low, recent 15 high → declining (t-test based).
        # Slight jitter avoids zero-variance degeneracy.
        s.recent_cp_losses = (
            [10 + i * 0.1 for i in range(15)]
            + [100 + i * 0.1 for i in range(15)]
        )
        s.update_trend()
        assert s.trend == 'declining'

    def test_trend_requires_enough_samples(self):
        """Short histories should stay 'stable' — the new t-test needs ≥30 samples."""
        s = StudentState(elo=1200)
        s.recent_cp_losses = [10] * 10 + [100] * 10  # only 20 samples
        s.update_trend()
        assert s.trend == 'stable'

    def test_trend_detects_improvement(self):
        s = StudentState(elo=1200)
        s.recent_cp_losses = (
            [100 + i * 0.1 for i in range(15)]  # older high
            + [20 + i * 0.1 for i in range(15)]  # recent low
        )
        s.update_trend()
        assert s.trend == 'improving'


class TestStudentSimulator:
    def test_simulator_elo_preserved_on_create(self):
        s = StudentSimulator(elo=1400)
        assert s.elo == 1400

    def test_simulator_has_state(self):
        s = StudentSimulator(elo=1400)
        assert hasattr(s, 'state')
        assert isinstance(s.state, StudentState)

    def test_update_state_modifies_profile(self):
        s = StudentSimulator(elo=1200, learning_rate=0.5)
        before = dict(s.weakness_profile)
        s.update_state(
            feedback_type=0,  # TACTICAL_ALERT
            move_quality=10,
            position_concepts=['tactics'],
        )
        # Tactics mastery should have changed
        assert s.weakness_profile['tactics'] != before['tactics']

    def test_respond_to_position_returns_legal_move(self, starting_board):
        s = StudentSimulator(elo=1200)
        move = s.respond_to_position(starting_board, feedback_type=0)
        assert move in starting_board.legal_moves

    def test_weakness_profile_bounded(self):
        """Mastery should never exceed [0, 1] even after many updates."""
        s = StudentSimulator(elo=1200, learning_rate=1.0)
        for _ in range(100):
            s.update_state(
                feedback_type=np.random.randint(0, 7),
                move_quality=float(np.random.rand() * 100),
                position_concepts=['tactics', 'strategy'],
            )
        for concept, mastery in s.weakness_profile.items():
            assert 0.0 <= mastery <= 1.0, (
                f"Mastery of {concept} escaped [0,1]: {mastery}"
            )


class TestStudentPopulation:
    def test_generates_correct_count(self):
        students = StudentPopulation.generate(n_students=20, random_state=42)
        assert len(students) == 20

    def test_students_are_diverse_in_elo(self):
        students = StudentPopulation.generate(n_students=50, random_state=42)
        elos = [s.elo for s in students]
        # Should span at least 400 ELO points (non-trivial diversity)
        assert max(elos) - min(elos) >= 400

    def test_deterministic_with_seed(self):
        s1 = StudentPopulation.generate(n_students=10, random_state=123)
        s2 = StudentPopulation.generate(n_students=10, random_state=123)
        elos1 = [s.elo for s in s1]
        elos2 = [s.elo for s in s2]
        assert elos1 == elos2
