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
        pos_idx: np.ndarray | None = None,
    ) -> float:
        """Select optimal bandwidth via leave-one-bracket-out CV.

        Two modes:

        * **Multiclass mode** (when ``pos_idx`` is ``None`` and the
          bracket models output a full move distribution): each row's
          prediction is the argmax class, compared with ``y``. This is
          the mode intended for the original multiclass ``MovePredictor``
          architecture.

        * **Candidate-ranking mode** (when ``pos_idx`` is provided): the
          bracket models are treated as binary classifiers of
          ``P(played | board, move)`` on candidate rows. For each
          held-out position, the predicted move is the candidate with
          the highest interpolated score; accuracy is whether that
          matches the played move (``y == 1``). This is the mode that
          matches the actual candidate-ranking training pipeline in
          ``scripts/train_and_evaluate.py``.

        Args:
            X: feature matrix, shape (n, d).
            y: labels. Multiclass mode: class indices; candidate mode: {0,1}.
            elos: ELO per row, shape (n,).
            bracket_models: {elo_center: fitted_model}.
            bandwidths: candidate bandwidths to sweep.
            pos_idx: optional per-row position index. If provided, switches
                to candidate-ranking mode.

        Returns:
            Optimal bandwidth (argmax mean accuracy across held-out brackets).
        """
        if bandwidths is None:
            bandwidths = KERNEL_BANDWIDTH_CANDIDATES

        centers = sorted(bracket_models.keys())
        best_bw = bandwidths[0]
        best_acc = -1.0

        from ..config import ELO_BRACKET_WIDTH

        for bw in bandwidths:
            accs = []

            for held_out in centers:
                mask = np.abs(elos - held_out) <= ELO_BRACKET_WIDTH
                if mask.sum() == 0:
                    continue
                X_ho = X[mask]
                y_ho = y[mask]
                pidx_ho = pos_idx[mask] if pos_idx is not None else None

                remaining = {c: m for c, m in bracket_models.items() if c != held_out}
                if not remaining:
                    continue

                # Collect per-bracket predictions at the held-out rows.
                bracket_preds = {}
                for c, model in remaining.items():
                    try:
                        bracket_preds[c] = model.predict_proba(X_ho)
                    except Exception:
                        continue
                if not bracket_preds:
                    continue

                # Kernel-interpolate.
                kernel_weights = np.array(
                    [np.exp(-0.5 * ((held_out - c) / bw) ** 2)
                     for c in bracket_preds.keys()]
                )
                kernel_weights /= kernel_weights.sum()
                combined = None
                for w, p in zip(kernel_weights, bracket_preds.values()):
                    combined = w * p if combined is None else combined + w * p

                if pidx_ho is not None:
                    # Candidate-ranking mode: combined is shape (n, 2),
                    # P(played)-column is [:, 1].
                    played_score = combined[:, 1] if combined.ndim == 2 else combined
                    correct = 0
                    total = 0
                    for pos in np.unique(pidx_ho):
                        pm = pidx_ho == pos
                        if y_ho[pm].sum() == 0:
                            continue
                        total += 1
                        if y_ho[pm][np.argmax(played_score[pm])] == 1:
                            correct += 1
                    acc = correct / total if total > 0 else 0.0
                else:
                    # Multiclass mode
                    if combined.ndim == 2:
                        pred_labels = combined.argmax(axis=1)
                        acc = float((pred_labels == y_ho).mean())
                    else:
                        acc = 0.0

                accs.append(acc)

            avg_acc = float(np.mean(accs)) if accs else 0.0
            if avg_acc > best_acc:
                best_acc = avg_acc
                best_bw = bw

        self.bandwidth = best_bw
        return best_bw
