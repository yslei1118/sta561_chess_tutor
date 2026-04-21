"""Position evaluation models: blunder detection and position complexity."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

from ..config import RF_N_ESTIMATORS, RF_MAX_DEPTH, RF_RANDOM_STATE


class BlunderDetector:
    """Binary classifier: predicts whether a move is a blunder (>100cp loss)."""

    def __init__(self, classifier: str = "rf", **kwargs):
        self.classifier_name = classifier
        if classifier == "rf":
            self.model = RandomForestClassifier(
                n_estimators=kwargs.get("n_estimators", RF_N_ESTIMATORS),
                max_depth=kwargs.get("max_depth", RF_MAX_DEPTH),
                random_state=kwargs.get("random_state", RF_RANDOM_STATE),
                n_jobs=-1,
            )
        elif classifier == "gbt":
            self.model = GradientBoostingClassifier(
                n_estimators=kwargs.get("n_estimators", 200),
                max_depth=kwargs.get("max_depth", 6),
                random_state=kwargs.get("random_state", RF_RANDOM_STATE),
            )
        elif classifier == "logistic":
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=kwargs.get("random_state", RF_RANDOM_STATE),
            )
        else:
            raise ValueError(f"Unknown classifier: {classifier}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BlunderDetector":
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(blunder) for each sample. Shape (n,)."""
        proba = self.model.predict_proba(X)
        # Find the index of class 1
        classes = self.model.classes_
        idx = np.where(classes == 1)[0]
        if len(idx) == 0:
            return np.zeros(X.shape[0])
        return proba[:, idx[0]]

    def score(self, X: np.ndarray, y: np.ndarray) -> dict:
        proba = self.predict_proba(X)
        preds = (proba > 0.5).astype(int)
        return {
            "auc": float(roc_auc_score(y, proba)) if len(np.unique(y)) > 1 else 0.0,
            "precision": float(precision_score(y, preds, zero_division=0)),
            "recall": float(recall_score(y, preds, zero_division=0)),
            "f1": float(f1_score(y, preds, zero_division=0)),
        }


class PositionComplexity:
    """Regression model: predicts how difficult a position is for a given ELO."""

    def __init__(self, model: str = "rf"):
        if model == "rf":
            self.model = RandomForestRegressor(
                n_estimators=RF_N_ESTIMATORS,
                max_depth=RF_MAX_DEPTH,
                random_state=RF_RANDOM_STATE,
                n_jobs=-1,
            )
        elif model == "ridge":
            self.model = Ridge(alpha=1.0)
        elif model == "gradient_boosting":
            self.model = GradientBoostingRegressor(
                n_estimators=200, max_depth=6,
                random_state=RF_RANDOM_STATE,
            )
        else:
            raise ValueError(f"Unknown model: {model}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PositionComplexity":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return complexity scores. Shape (n,)."""
        return self.model.predict(X)
