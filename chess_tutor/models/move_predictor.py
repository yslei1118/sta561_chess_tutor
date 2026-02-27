"""ELO-conditioned human move predictor with three architectures."""

import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier

from ..config import (
    ELO_BRACKETS, ELO_BRACKET_WIDTH,
    RF_N_ESTIMATORS, RF_MAX_DEPTH, RF_MIN_SAMPLES_LEAF, RF_N_JOBS, RF_RANDOM_STATE,
    SVM_KERNEL, SVM_C, SVM_GAMMA, KERNEL_BANDWIDTH_DEFAULT,
)
from .kernel_interpolation import NadarayaWatsonELO


def _make_classifier(name: str, **kwargs):
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=kwargs.get("n_estimators", RF_N_ESTIMATORS),
            max_depth=kwargs.get("max_depth", RF_MAX_DEPTH),
            min_samples_leaf=kwargs.get("min_samples_leaf", RF_MIN_SAMPLES_LEAF),
            n_jobs=kwargs.get("n_jobs", RF_N_JOBS),
            random_state=kwargs.get("random_state", RF_RANDOM_STATE),
        )
    elif name == "svm":
        return SVC(
            kernel=kwargs.get("kernel", SVM_KERNEL),
            C=kwargs.get("C", SVM_C),
            gamma=kwargs.get("gamma", SVM_GAMMA),
            probability=True,
            random_state=kwargs.get("random_state", RF_RANDOM_STATE),
        )
    elif name == "logistic":
        return LogisticRegression(
            max_iter=1000,
            random_state=kwargs.get("random_state", RF_RANDOM_STATE),
        )
    elif name == "ridge":
        return RidgeClassifier(
            alpha=kwargs.get("alpha", 1.0),
            random_state=kwargs.get("random_state", RF_RANDOM_STATE),
        )
    else:
        raise ValueError(f"Unknown classifier: {name}")


class MovePredictor:
    """ELO-conditioned human move predictor.

    Supports three architectures:
        A: Separate model per ELO bracket
        B: Single pooled model with ELO as feature
        C: Kernel-smoothed interpolation across brackets
    """

    def __init__(
        self,
        architecture: str = "C",
        base_classifier: str = "rf",
        elo_brackets: list[int] | None = None,
        kernel_bandwidth: float = KERNEL_BANDWIDTH_DEFAULT,
        **classifier_kwargs,
    ):
        self.architecture = architecture.upper()
        self.base_classifier = base_classifier
        self.elo_brackets = list(elo_brackets or ELO_BRACKETS)
        self.kernel_bandwidth = kernel_bandwidth
        self.classifier_kwargs = classifier_kwargs

        self.models: dict[int, object] = {}
        self.pooled_model = None
        self.kernel = NadarayaWatsonELO(bandwidth=kernel_bandwidth)
        self._classes = None

    def fit(self, X: np.ndarray, y: np.ndarray,
            elos: np.ndarray, **fit_kwargs) -> "MovePredictor":
        """Train the model."""
        self._classes = np.unique(y)

        if self.architecture == "A":
            self._fit_per_bracket(X, y, elos)
        elif self.architecture == "B":
            self._fit_pooled(X, y, elos)
        elif self.architecture == "C":
            self._fit_per_bracket(X, y, elos)
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")

        return self

    def _fit_per_bracket(self, X, y, elos):
        for bracket in self.elo_brackets:
            mask = np.abs(elos - bracket) <= ELO_BRACKET_WIDTH
            if mask.sum() < 10:
                continue
            X_b, y_b = X[mask], y[mask]
            clf = _make_classifier(self.base_classifier, **self.classifier_kwargs)
            clf.fit(X_b, y_b)
            self.models[bracket] = clf

    def _fit_pooled(self, X, y, elos):
        from ..utils.helpers import normalize_elo
        elo_feature = np.array([normalize_elo(e) for e in elos]).reshape(-1, 1)
        X_aug = np.hstack([X, elo_feature])
        clf = _make_classifier(self.base_classifier, **self.classifier_kwargs)
        clf.fit(X_aug, y)
        self.pooled_model = clf

    def predict(self, X: np.ndarray, target_elo: int,
                top_k: int = 5) -> np.ndarray:
        """Predict top-K move indices for a target ELO."""
        proba = self.predict_proba(X, target_elo)
        # Get top-k indices per sample
        if proba.ndim == 1:
            proba = proba.reshape(1, -1)
        top_k = min(top_k, proba.shape[1])
        return np.argsort(proba, axis=1)[:, -top_k:][:, ::-1]

    def predict_proba(self, X: np.ndarray, target_elo: int) -> np.ndarray:
        """Get full probability distribution over moves."""
        if self.architecture == "A":
            return self._predict_proba_bracket(X, target_elo)
        elif self.architecture == "B":
            return self._predict_proba_pooled(X, target_elo)
        elif self.architecture == "C":
            return self._predict_proba_kernel(X, target_elo)
        raise ValueError(f"Unknown architecture: {self.architecture}")

    def _predict_proba_bracket(self, X, target_elo):
        # Find nearest bracket
        nearest = min(self.models.keys(), key=lambda b: abs(b - target_elo))
        return self._get_proba(self.models[nearest], X)

    def _predict_proba_pooled(self, X, target_elo):
        from ..utils.helpers import normalize_elo
        elo_feat = np.full((X.shape[0], 1), normalize_elo(target_elo))
        X_aug = np.hstack([X, elo_feat])
        return self._get_proba(self.pooled_model, X_aug)

    def _predict_proba_kernel(self, X, target_elo):
        bracket_preds = {}
        for elo, model in self.models.items():
            bracket_preds[elo] = self._get_proba(model, X)
        return self.kernel.interpolate(bracket_preds, float(target_elo))

    def _get_proba(self, model, X) -> np.ndarray:
        """Get probability predictions, handling models without predict_proba."""
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            # Align to full class set
            if self._classes is not None and hasattr(model, "classes_"):
                full_proba = np.zeros((X.shape[0], len(self._classes)))
                for i, c in enumerate(model.classes_):
                    idx = np.searchsorted(self._classes, c)
                    if idx < len(self._classes) and self._classes[idx] == c:
                        full_proba[:, idx] = proba[:, i]
                return full_proba
            return proba
        else:
            # RidgeClassifier: use decision_function
            dec = model.decision_function(X)
            if dec.ndim == 1:
                dec = dec.reshape(-1, 1)
            from ..utils.helpers import softmax
            return np.apply_along_axis(softmax, 1, dec)

    def score(self, X: np.ndarray, y: np.ndarray, elos: np.ndarray) -> dict:
        """Evaluate model performance."""
        per_bracket = {}
        cross_matrix = self.cross_elo_evaluation(X, y, elos)

        for bracket in self.elo_brackets:
            mask = np.abs(elos - bracket) <= ELO_BRACKET_WIDTH
            if mask.sum() == 0:
                continue
            proba = self.predict_proba(X[mask], bracket)
            preds = proba.argmax(axis=1)
            per_bracket[bracket] = float((preds == y[mask]).mean())

        # Overall accuracy
        all_preds = []
        for i, (x_i, elo_i) in enumerate(zip(X, elos)):
            proba = self.predict_proba(x_i.reshape(1, -1), int(elo_i))
            all_preds.append(proba.argmax(axis=1)[0])
        all_preds = np.array(all_preds)

        # Top-k
        top1 = float((all_preds == y).mean())

        return {
            "accuracy_top1": top1,
            "per_bracket_accuracy": per_bracket,
            "cross_elo_matrix": cross_matrix,
        }

    def cross_elo_evaluation(self, X_test: np.ndarray, y_test: np.ndarray,
                             test_elos: np.ndarray) -> np.ndarray:
        """Build cross-ELO evaluation matrix."""
        brackets = [b for b in self.elo_brackets if b in self.models]
        n = len(brackets)
        matrix = np.zeros((n, n))

        for j, test_elo in enumerate(brackets):
            mask = np.abs(test_elos - test_elo) <= ELO_BRACKET_WIDTH
            if mask.sum() == 0:
                continue
            X_b = X_test[mask]
            y_b = y_test[mask]

            for i, train_elo in enumerate(brackets):
                if train_elo not in self.models:
                    continue
                proba = self._get_proba(self.models[train_elo], X_b)
                preds = proba.argmax(axis=1)
                matrix[i, j] = float((preds == y_b).mean())

        return matrix

    def feature_importance(self, bracket: int | None = None) -> dict[str, float]:
        """Get feature importance scores."""
        from ..config import BOARD_FEATURE_NAMES, MOVE_FEATURE_NAMES
        names = BOARD_FEATURE_NAMES + MOVE_FEATURE_NAMES

        if self.architecture == "B" and self.pooled_model is not None:
            model = self.pooled_model
        elif bracket is not None and bracket in self.models:
            model = self.models[bracket]
        elif self.models:
            model = next(iter(self.models.values()))
        else:
            return {}

        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
        elif hasattr(model, "coef_"):
            imp = np.abs(model.coef_).mean(axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_)
        else:
            return {}

        n = min(len(names), len(imp))
        return {names[i]: float(imp[i]) for i in range(n)}

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({
                "architecture": self.architecture,
                "base_classifier": self.base_classifier,
                "elo_brackets": self.elo_brackets,
                "kernel_bandwidth": self.kernel_bandwidth,
                "models": self.models,
                "pooled_model": self.pooled_model,
                "_classes": self._classes,
            }, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.architecture = data["architecture"]
        self.base_classifier = data["base_classifier"]
        self.elo_brackets = data["elo_brackets"]
        self.kernel_bandwidth = data["kernel_bandwidth"]
        self.models = data["models"]
        self.pooled_model = data["pooled_model"]
        self._classes = data["_classes"]
        self.kernel = NadarayaWatsonELO(bandwidth=self.kernel_bandwidth)
