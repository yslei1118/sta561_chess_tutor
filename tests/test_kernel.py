"""Tests for chess_tutor.models.kernel_interpolation — Nadaraya-Watson over ELO.

This is the methodological contribution of the project. We verify:
- Kernel weights sum to 1 (probability distribution)
- Small bandwidth → near-nearest-bracket (A recovers)
- Large bandwidth → near-uniform (all brackets equally weighted)
- Interpolation produces a valid probability distribution output
- Gaussian kernel: K(0)=1, K(d)=K(-d), monotone decreasing in |d|
"""

import numpy as np
import pytest

from chess_tutor.models.kernel_interpolation import NadarayaWatsonELO


pytestmark = pytest.mark.math


class TestGaussianKernel:
    def test_kernel_at_zero_distance_is_one(self):
        nw = NadarayaWatsonELO(bandwidth=100)
        assert nw.kernel(1500, 1500) == pytest.approx(1.0)

    def test_kernel_symmetric(self):
        nw = NadarayaWatsonELO(bandwidth=100)
        for d in [50, 100, 200]:
            assert nw.kernel(1500, 1500 + d) == pytest.approx(
                nw.kernel(1500, 1500 - d)
            )

    def test_kernel_monotone_decreasing(self):
        nw = NadarayaWatsonELO(bandwidth=100)
        vals = [nw.kernel(1500, 1500 + d) for d in range(0, 500, 50)]
        for a, b in zip(vals, vals[1:]):
            assert a >= b, "Kernel is not monotone in |distance|"

    def test_kernel_positive(self):
        nw = NadarayaWatsonELO(bandwidth=100)
        for d in range(-500, 501, 50):
            assert nw.kernel(1500, 1500 + d) > 0


class TestKernelWeights:
    def test_weights_sum_to_one(self, elo_brackets):
        nw = NadarayaWatsonELO(bandwidth=100)
        for target in [1100, 1400, 1640, 1900, 2100]:
            w = nw.kernel_weights(target, elo_brackets)
            assert w.sum() == pytest.approx(1.0, rel=1e-9)

    def test_weights_all_positive(self, elo_brackets):
        nw = NadarayaWatsonELO(bandwidth=50)
        w = nw.kernel_weights(1300, elo_brackets)
        assert (w > 0).all()

    def test_weights_concentrate_on_nearest_when_bandwidth_small(self, elo_brackets):
        nw = NadarayaWatsonELO(bandwidth=10)
        # Query ELO=1300, exact bracket center
        w = nw.kernel_weights(1300, elo_brackets)
        # Most weight should be on 1300 (index 1)
        assert np.argmax(w) == 1
        assert w[1] > 0.99  # nearly degenerate

    def test_weights_spread_when_bandwidth_large(self, elo_brackets):
        nw = NadarayaWatsonELO(bandwidth=10000)
        w = nw.kernel_weights(1500, elo_brackets)
        # All weights should be nearly equal
        assert np.allclose(w, 0.2, atol=0.01)

    def test_weights_argmax_at_nearest_bracket(self, elo_brackets):
        nw = NadarayaWatsonELO(bandwidth=100)
        # Query near 1640 → nearest is 1700 (index 3)
        w = nw.kernel_weights(1680, elo_brackets)
        assert np.argmax(w) == 3


class TestInterpolation:
    def test_interpolation_produces_valid_distribution(self, elo_brackets):
        """Interpolated move probabilities must be a valid distribution."""
        nw = NadarayaWatsonELO(bandwidth=100)
        n_moves = 8
        bracket_preds = {
            b: np.abs(np.random.randn(n_moves))
            for b in elo_brackets
        }
        # Normalize each bracket's predictions to a probability distribution
        for b in elo_brackets:
            bracket_preds[b] /= bracket_preds[b].sum()

        result = nw.interpolate(bracket_preds, 1500)
        assert result.shape == (n_moves,)
        assert result.sum() == pytest.approx(1.0, rel=1e-9)
        assert (result >= 0).all()

    def test_interpolation_at_bracket_center_recovers_bracket(self, elo_brackets):
        """At exact bracket center with tiny bandwidth, interpolation = that bracket."""
        nw = NadarayaWatsonELO(bandwidth=1)
        n_moves = 5
        np.random.seed(0)
        bracket_preds = {
            b: np.random.dirichlet(np.ones(n_moves))
            for b in elo_brackets
        }
        result = nw.interpolate(bracket_preds, 1300)
        # With bw=1, weight is ~entirely on 1300
        assert np.allclose(result, bracket_preds[1300], atol=1e-3)

    def test_interpolation_multi_sample_batch(self, elo_brackets):
        """The implementation should handle batch predictions (2D arrays)."""
        nw = NadarayaWatsonELO(bandwidth=100)
        batch_size = 4
        n_moves = 6
        bracket_preds = {}
        for b in elo_brackets:
            arr = np.abs(np.random.randn(batch_size, n_moves))
            arr = arr / arr.sum(axis=1, keepdims=True)
            bracket_preds[b] = arr
        result = nw.interpolate(bracket_preds, 1500)
        assert result.shape == (batch_size, n_moves)
        # Each row should sum to ~1
        assert np.allclose(result.sum(axis=1), 1.0, atol=1e-9)


class TestBandwidthSelection:
    def test_bandwidth_candidates_all_positive(self):
        from chess_tutor.config import KERNEL_BANDWIDTH_CANDIDATES
        assert all(bw > 0 for bw in KERNEL_BANDWIDTH_CANDIDATES)
        # Should span at least two orders of magnitude
        assert max(KERNEL_BANDWIDTH_CANDIDATES) / min(KERNEL_BANDWIDTH_CANDIDATES) > 10
