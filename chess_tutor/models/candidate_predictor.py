"""Wrapper around the trained candidate-ranking RF classifiers.

The trained models (``models/saved/models_a_candidate.pkl`` and
``models/saved/model_b_candidate.pkl``) are binary classifiers that predict
P(human_played=1 | board_features, move_features) for each candidate move.
This module wraps them behind a single ``predict_move_probs`` interface
used by both ``ChessTutorBot`` (demo) and ``StudentSimulator`` (offline eval).
"""

import os
import pickle
import numpy as np


_DEFAULT_ARCH_A_PATH = "models/saved/models_a_candidate.pkl"
_DEFAULT_ARCH_B_PATH = "models/saved/model_b_candidate.pkl"


class CandidateRankingPredictor:
    """Wraps the trained Architecture-A or Architecture-B RF classifiers.

    Architecture A: dict ``{elo_bracket: RandomForestClassifier}``.
        Each RF expects 40-dim input (30 board + 10 move features).
        At inference, the nearest bracket model is used.

    Architecture B: single ``RandomForestClassifier`` with 41-dim input
        (40 features + normalised ELO appended as the last column).

    Usage::

        predictor = CandidateRankingPredictor.load_arch_a()
        probs = predictor.predict_move_probs(board, legal_moves, target_elo=1500)
        # probs: np.ndarray of shape (len(legal_moves),), sums to 1
    """

    def __init__(self, models, architecture: str = "A"):
        """
        Args:
            models: For Arch A, a dict ``{int: RF}``; for Arch B, a single RF.
            architecture: ``"A"`` or ``"B"``.
        """
        self._models = models
        self.architecture = architecture.upper()

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def load_arch_a(cls, path: str = _DEFAULT_ARCH_A_PATH) -> "CandidateRankingPredictor":
        """Load Architecture-A per-bracket models."""
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Arch-A model not found at '{path}'. "
                "Run scripts/train_and_evaluate.py first."
            )
        with open(path, "rb") as f:
            models = pickle.load(f)
        return cls(models, architecture="A")

    @classmethod
    def load_arch_b(cls, path: str = _DEFAULT_ARCH_B_PATH) -> "CandidateRankingPredictor":
        """Load Architecture-B pooled model."""
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Arch-B model not found at '{path}'. "
                "Run scripts/train_and_evaluate.py first."
            )
        with open(path, "rb") as f:
            model = pickle.load(f)
        return cls(model, architecture="B")

    @classmethod
    def try_load(cls, prefer: str = "A") -> "CandidateRankingPredictor | None":
        """Try loading models, return None if files are not present."""
        try:
            if prefer.upper() == "B":
                return cls.load_arch_b()
            return cls.load_arch_a()
        except FileNotFoundError:
            return None

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def predict_move_probs(
        self,
        board,
        legal_moves: list,
        target_elo: int,
    ) -> np.ndarray:
        """Return a probability distribution over ``legal_moves``.

        Builds one 40-dim feature row per legal move, feeds the batch to the
        RF, and normalises P(human_played=1) across moves.

        Args:
            board: ``chess.Board`` — the position to evaluate.
            legal_moves: list of ``chess.Move``.
            target_elo: ELO level to predict for.

        Returns:
            ``np.ndarray`` of shape ``(len(legal_moves),)`` summing to 1.
        """
        from ..data.extract_features import extract_board_features, extract_move_features

        board_feats = extract_board_features(board)
        rows = []
        for move in legal_moves:
            move_feats = extract_move_features(board, move)
            rows.append(np.concatenate([board_feats, move_feats]))
        X = np.array(rows, dtype=np.float64)  # (n_legal, 40)

        if self.architecture == "A":
            nearest = min(self._models.keys(), key=lambda b: abs(b - target_elo))
            raw = self._models[nearest].predict_proba(X)[:, 1]
        elif self.architecture == "B":
            elo_col = np.full((len(legal_moves), 1), (target_elo - 1100) / 800.0)
            X_aug = np.hstack([X, elo_col])
            raw = self._models.predict_proba(X_aug)[:, 1]
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")

        raw = np.clip(raw, 1e-8, None)
        return raw / raw.sum()
