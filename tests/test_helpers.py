"""Tests for chess_tutor.utils.helpers — mathematical utility functions.

These tests verify invariants that the entire teaching engine depends on:
softmax must be a valid distribution, normalize_elo must be monotone, etc.
Any of these failing would propagate into bandit decisions and feedback scores.
"""

import numpy as np
import pytest

from chess_tutor.utils.helpers import (
    sigmoid, normalize_elo, softmax, cp_to_win_prob,
    sample_from_distribution,
)


pytestmark = pytest.mark.math


class TestSigmoid:
    def test_sigmoid_at_zero_is_half(self):
        assert sigmoid(0.0) == pytest.approx(0.5)

    def test_sigmoid_bounded_in_unit_interval(self):
        for x in np.linspace(-100, 100, 50):
            y = sigmoid(float(x))
            assert 0.0 <= y <= 1.0, f"sigmoid({x}) = {y} escaped [0,1]"

    def test_sigmoid_antisymmetry(self):
        for x in np.linspace(-10, 10, 20):
            assert sigmoid(float(x)) + sigmoid(-float(x)) == pytest.approx(1.0, rel=1e-9)

    def test_sigmoid_monotone_increasing(self):
        samples = [sigmoid(float(x)) for x in np.linspace(-10, 10, 50)]
        for a, b in zip(samples, samples[1:]):
            assert a <= b + 1e-12

    def test_sigmoid_numerical_stability_large_negative(self):
        # Naive implementation would overflow; should return near-zero without error
        assert sigmoid(-1000.0) == pytest.approx(0.0, abs=1e-300)

    def test_sigmoid_numerical_stability_large_positive(self):
        assert sigmoid(1000.0) == pytest.approx(1.0)


class TestNormalizeElo:
    def test_normalize_elo_min_is_zero(self):
        assert normalize_elo(800) == pytest.approx(0.0)

    def test_normalize_elo_max_is_one(self):
        assert normalize_elo(2200) == pytest.approx(1.0)

    def test_normalize_elo_monotone(self):
        samples = [normalize_elo(e) for e in range(800, 2201, 50)]
        for a, b in zip(samples, samples[1:]):
            assert a <= b

    def test_normalize_elo_middle(self):
        # 1500 is the middle of [800, 2200]
        assert normalize_elo(1500) == pytest.approx(0.5)


class TestSoftmax:
    def test_softmax_sums_to_one(self):
        for size in [3, 7, 20]:
            x = np.random.randn(size)
            p = softmax(x)
            assert p.sum() == pytest.approx(1.0, rel=1e-9)

    def test_softmax_all_positive(self):
        x = np.random.randn(10)
        p = softmax(x)
        assert (p > 0).all()

    def test_softmax_preserves_argmax(self):
        x = np.array([0.1, 3.0, 0.5, -1.0, 2.9])
        p = softmax(x)
        assert np.argmax(p) == np.argmax(x)

    def test_softmax_low_temperature_concentrates(self):
        x = np.array([1.0, 2.0, 3.0])
        p_low_T = softmax(x, temperature=0.1)
        p_high_T = softmax(x, temperature=10.0)
        # At low T, argmax dominates; at high T, more uniform
        assert p_low_T.max() > p_high_T.max()
        assert p_high_T.max() - p_high_T.min() < p_low_T.max() - p_low_T.min()

    def test_softmax_numerical_stability_large_logits(self):
        # Without the max-subtraction trick this would overflow
        x = np.array([1000.0, 1001.0, 999.0])
        p = softmax(x)
        assert p.sum() == pytest.approx(1.0, rel=1e-9)
        assert np.argmax(p) == 1

    def test_softmax_uniform_on_equal_logits(self):
        x = np.zeros(5)
        p = softmax(x)
        assert np.allclose(p, 0.2)


class TestCpToWinProb:
    def test_even_material_half_win_prob(self):
        assert cp_to_win_prob(0.0) == pytest.approx(0.5)

    def test_winning_side_higher_prob(self):
        assert cp_to_win_prob(300.0) > 0.5
        assert cp_to_win_prob(-300.0) < 0.5

    def test_bounded_in_unit_interval(self):
        for cp in np.linspace(-2000, 2000, 50):
            assert 0.0 <= cp_to_win_prob(float(cp)) <= 1.0

    def test_monotone_increasing(self):
        samples = [cp_to_win_prob(float(cp)) for cp in np.linspace(-500, 500, 20)]
        for a, b in zip(samples, samples[1:]):
            assert a <= b + 1e-12


class TestSampleFromDistribution:
    def test_sample_returns_valid_index(self):
        probs = np.array([0.1, 0.3, 0.4, 0.2])
        for _ in range(50):
            idx = sample_from_distribution(probs)
            assert 0 <= idx < len(probs)

    def test_sample_frequencies_match_probs(self):
        probs = np.array([0.1, 0.1, 0.8])
        rng = np.random.RandomState(0)
        counts = np.zeros(3)
        for _ in range(5000):
            counts[sample_from_distribution(probs, rng=rng)] += 1
        freqs = counts / counts.sum()
        # 5000 samples → expect freqs within ~1.5% of probs
        assert np.allclose(freqs, probs, atol=0.015)
