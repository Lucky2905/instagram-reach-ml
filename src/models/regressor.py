"""
src/models/regressor.py — ReachRegressor
=========================================
Wraps sklearn LinearRegression with the BaseModel interface.
Single responsibility: reach value prediction (regression task only).

Register via: ModelFactory.register("linear_regression", ReachRegressor)
Create via:   ModelFactory.create("linear_regression")
"""

from __future__ import annotations

from typing import Any, Dict
import numpy as np
import logging

from sklearn.linear_model import LinearRegression

from src.patterns.factory import BaseModel, ModelFactory

logger = logging.getLogger(__name__)


class ReachRegressor(BaseModel):
    """
    Linear Regression model for predicting continuous Instagram reach values.

    Why LinearRegression?
        - Interpretable coefficients expose feature impact.
        - Fast to train — suitable as a baseline model.
        - Serves as the inner estimator inside CrossValidationDecorator.
    """

    def __init__(self, **kwargs) -> None:
        self._model = LinearRegression(**kwargs)
        self._is_fitted: bool = False

    # ── BaseModel interface ───────────────────────────────────────────────────

    def fit(self, X, y) -> "ReachRegressor":
        n = len(y) if hasattr(y, "__len__") else "?"
        logger.info("[ReachRegressor] Fitting LinearRegression on %s samples…", n)
        self._model.fit(X, y)
        self._is_fitted = True
        logger.info("[ReachRegressor] Fit complete.")
        return self

    def predict(self, X) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("ReachRegressor must be fitted before calling predict().")
        return self._model.predict(X)

    def get_params(self) -> Dict[str, Any]:
        return self._model.get_params()

    @property
    def model_type(self) -> str:
        return "LinearRegression"

    # ── Extra properties ──────────────────────────────────────────────────────

    @property
    def coef_(self) -> np.ndarray:
        """Feature coefficients (available after fit)."""
        return self._model.coef_

    @property
    def intercept_(self) -> float:
        return float(self._model.intercept_)

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted


# ── Self-register with ModelFactory on import ─────────────────────────────────
ModelFactory.register("linear_regression", ReachRegressor)
