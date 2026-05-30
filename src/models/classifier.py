"""
src/models/classifier.py — TierClassifier
==========================================
Wraps sklearn RandomForestClassifier with the BaseModel interface.
Single responsibility: reach tier classification (low / medium / high).

Register via: ModelFactory.register("random_forest", TierClassifier)
Create via:   ModelFactory.create("random_forest", n_estimators=200)
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import numpy as np
import logging

from sklearn.ensemble import RandomForestClassifier

from src.patterns.factory import BaseModel, ModelFactory

logger = logging.getLogger(__name__)


class TierClassifier(BaseModel):
    """
    Random Forest classifier for predicting Instagram reach tier (0/1/2).

    Why RandomForest?
        - Handles non-linear feature interactions (e.g. reel + peak hour).
        - Provides feature_importances_ for dashboard visualization.
        - Robust to outliers without requiring extensive preprocessing.
        - predict_proba() enables confidence scoring in the API response.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs,
    ) -> None:
        self._model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )
        self._is_fitted: bool = False

    # ── BaseModel interface ───────────────────────────────────────────────────

    def fit(self, X, y) -> "TierClassifier":
        n = len(y) if hasattr(y, "__len__") else "?"
        logger.info(
            "[TierClassifier] Fitting RandomForest(%d trees) on %s samples…",
            self._model.n_estimators, n,
        )
        self._model.fit(X, y)
        self._is_fitted = True
        logger.info("[TierClassifier] Fit complete.")
        return self

    def predict(self, X) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("TierClassifier must be fitted before calling predict().")
        return self._model.predict(X)

    def get_params(self) -> Dict[str, Any]:
        return self._model.get_params()

    @property
    def model_type(self) -> str:
        return "RandomForestClassifier"

    # ── Extra capabilities ────────────────────────────────────────────────────

    def predict_proba(self, X) -> np.ndarray:
        """Return class probability matrix (n_samples × 3) for confidence scoring."""
        if not self._is_fitted:
            raise RuntimeError("TierClassifier must be fitted before predict_proba().")
        return self._model.predict_proba(X)

    @property
    def feature_importances_(self) -> np.ndarray:
        """Gini-based feature importances (available after fit)."""
        return self._model.feature_importances_

    @property
    def classes_(self) -> np.ndarray:
        return self._model.classes_

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted


# ── Self-register with ModelFactory on import ─────────────────────────────────
ModelFactory.register("random_forest", TierClassifier)
