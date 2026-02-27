"""Nadaraya-Watson kernel regression for interpolating move predictions across ELO."""

import numpy as np
from ..config import KERNEL_BANDWIDTH_DEFAULT, KERNEL_BANDWIDTH_CANDIDATES, ELO_BRACKETS


class NadarayaWatsonELO:
    """Nadaraya-Watson kernel regression for interpolating move predictions across ELO.

    Given per-bracket classifiers P_k(m|x) at ELO centers s_k, computes:
        P(m|x, s*) = Σ_k K_h(s* - s_k) · P_k(m|x) / Σ_k K_h(s* - s_k)

    where K_h is a Gaussian kernel with bandwidth h.
    """

    def __init__(self, bandwidth: float = KERNEL_BANDWIDTH_DEFAULT):
        self.bandwidth = bandwidth

    def kernel(self, elo_query: float, elo_center: float) -> float:
        """Gaussian kernel: K_h(s* - s_k) = exp(-0.5 * ((s* - s_k)/h)^2)"""
        return np.exp(-0.5 * ((elo_query - elo_center) / self.bandwidth) ** 2)

    def kernel_weights(self, target_elo: float,
                       bracket_centers: list[int]) -> np.ndarray:
        """Return normalized kernel weights for each bracket."""
        weights = np.array([self.kernel(target_elo, c) for c in bracket_centers])
        total = weights.sum()
        if total == 0:
            return np.ones(len(bracket_centers)) / len(bracket_centers)
        return weights / total

    def interpolate(self, bracket_predictions: dict[int, np.ndarray],
                    target_elo: float) -> np.ndarray:
        """Combine per-bracket predictions via kernel weighting.

        Args:
            bracket_predictions: {elo_center: P_k(m|x)} probability arrays
            target_elo: query ELO

        Returns:
            Interpolated probability distribution over moves
        """
        centers = sorted(bracket_predictions.keys())
        weights = self.kernel_weights(target_elo, centers)

        result = None
        for w, c in zip(weights, centers):
            pred = bracket_predictions[c]
            if result is None:
                result = w * pred
            else:
                result = result + w * pred

        # Ensure valid probability distribution
        if result.ndim == 1:
            total = result.sum()
            if total > 0:
                result = result / total
        else:
            # Per-sample normalization
            totals = result.sum(axis=1, keepdims=True)
            totals = np.maximum(totals, 1e-10)
            result = result / totals

        return result

    def select_bandwidth_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        elos: np.ndarray,
        bracket_models: dict,
        bandwidths: list[float] | None = None,
    ) -> float:
        """Select optimal bandwidth via leave-one-bracket-out CV.

        For each bracket b:
            1. Hold out bracket b's test data
            2. Interpolate from remaining brackets
            3. Measure accuracy on held-out bracket

        Returns:
            Optimal bandwidth (argmax avg accuracy across held-out brackets)
        """
        if bandwidths is None:
            bandwidths = KERNEL_BANDWIDTH_CANDIDATES

        centers = sorted(bracket_models.keys())
        best_bw = bandwidths[0]
        best_acc = -1.0

        from ..config import ELO_BRACKET_WIDTH

        for bw in bandwidths:
            nw = NadarayaWatsonELO(bandwidth=bw)
            accs = []

            for held_out in centers:
                # Get held-out data
                mask = np.abs(elos - held_out) <= ELO_BRACKET_WIDTH
                if mask.sum() == 0:
                    continue
                X_ho = X[mask]
                y_ho = y[mask]

                # Get predictions from remaining brackets
                remaining = {c: m for c, m in bracket_models.items() if c != held_out}
                if not remaining:
                    continue

                bracket_preds = {}
                for c, model in remaining.items():
                    try:
                        bracket_preds[c] = model.predict_proba(X_ho)
                    except Exception:
                        continue

                if not bracket_preds:
                    continue

                interpolated = nw.interpolate(bracket_preds, float(held_out))
                if interpolated.ndim == 2:
                    pred_labels = interpolated.argmax(axis=1)
                    acc = (pred_labels == y_ho).mean()
                else:
                    acc = 0.0
                accs.append(acc)

            avg_acc = np.mean(accs) if accs else 0.0
            if avg_acc > best_acc:
                best_acc = avg_acc
                best_bw = bw

        self.bandwidth = best_bw
        return best_bw
