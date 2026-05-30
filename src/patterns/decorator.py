"""
src/patterns/decorator.py — Decorator Pattern
=============================================
Wraps any BaseModel with additional capabilities (cross-validation,
feature selection) without modifying the wrapped model's class.

Pattern verification:
    CrossValidationDecorator wraps ANY model and adds .cv_score() method —
    a method that does not exist on the base model.

Usage:
    base  = ModelFactory.create("linear_regression")
    model = CrossValidationDecorator(base, cv=5, scoring="r2")
    model.fit(X_train, y_train)
    print(model.cv_score())           # ← capability added by decorator

    # Stack decorators
    model = FeatureSelectionDecorator(
                CrossValidationDecorator(base, cv=5), k=8
            )
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


# ── Base Decorator ────────────────────────────────────────────────────────────

class ModelDecorator:
    """
    Transparent proxy that delegates all calls to the wrapped model.
    Subclasses override only the methods they augment.
    Does NOT inherit BaseModel to avoid metaclass conflicts with ABC;
    duck-typing satisfies the BaseModel contract.
    """

    def __init__(self, model) -> None:
        self._model = model

    # ── Delegate BaseModel interface ──────────────────────────────────────────

    def fit(self, X, y) -> "ModelDecorator":
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)

    def get_params(self) -> Dict[str, Any]:
        return self._model.get_params()

    @property
    def model_type(self) -> str:
        return f"Decorated({self._model.model_type})"

    @property
    def wrapped(self):
        """Direct access to the inner model."""
        return self._model


# ── CrossValidation Decorator ─────────────────────────────────────────────────

class CrossValidationDecorator(ModelDecorator):
    """
    Augments fit() with k-fold cross-validation before final training.

    ADDS: .cv_score() — mean CV score (absent on the base model).
    ADDS: .cv_score_details() — full fold statistics.

    The observer pattern FOLD_COMPLETE event is fired here so that
    observers log fold metrics without the model knowing about them.

    Args:
        model:   Any BaseModel-compatible instance.
        cv:      Number of folds (default 5).
        scoring: sklearn scorer string (default 'r2').
        subject: Optional TrainingSubject to fire FOLD_COMPLETE events.
    """

    def __init__(
        self,
        model,
        cv: int = 5,
        scoring: str = "r2",
        subject=None,
    ) -> None:
        super().__init__(model)
        self._cv = cv
        self._scoring = scoring
        self._subject = subject          # optional observer hook
        self._cv_scores: Optional[np.ndarray] = None

    def fit(self, X, y) -> "CrossValidationDecorator":
        from sklearn.model_selection import cross_val_score

        # Unwrap to the underlying sklearn estimator for CV
        estimator = self._model._model

        logger.info(
            "[CrossValidationDecorator] Starting %d-fold CV (scoring='%s')…",
            self._cv, self._scoring,
        )
        self._cv_scores = cross_val_score(
            estimator, X, y, cv=self._cv, scoring=self._scoring
        )

        # Fire FOLD_COMPLETE event for each fold via observer
        if self._subject is not None:
            for fold_idx, score in enumerate(self._cv_scores, start=1):
                self._subject.notify("FOLD_COMPLETE", {
                    "fold": fold_idx,
                    "score": round(float(score), 4),
                    "scoring": self._scoring,
                })

        logger.info(
            "[CrossValidationDecorator] CV scores=%s  mean=%.4f ± %.4f",
            np.round(self._cv_scores, 4),
            self._cv_scores.mean(),
            self._cv_scores.std(),
        )

        # Train on the full dataset after CV validation
        super().fit(X, y)
        return self

    # ── New capability: cv_score ──────────────────────────────────────────────

    def cv_score(self) -> float:
        """
        Return the mean cross-validation score.
        This method is ADDED by the decorator — it does not exist on BaseModel.

        Raises:
            RuntimeError: If called before fit().
        """
        if self._cv_scores is None:
            raise RuntimeError(
                "cv_score() called before fit(). Run fit(X, y) first."
            )
        return float(self._cv_scores.mean())

    def cv_score_details(self) -> Dict[str, Any]:
        """Return full fold statistics including per-fold scores."""
        if self._cv_scores is None:
            return {}
        return {
            "mean": float(self._cv_scores.mean()),
            "std": float(self._cv_scores.std()),
            "min": float(self._cv_scores.min()),
            "max": float(self._cv_scores.max()),
            "scores": self._cv_scores.tolist(),
            "cv_folds": self._cv,
            "scoring": self._scoring,
        }

    @property
    def model_type(self) -> str:
        return f"CV-{self._cv}Fold({self._model.model_type})"


# ── Feature Selection Decorator ───────────────────────────────────────────────

class FeatureSelectionDecorator(ModelDecorator):
    """
    Applies SelectKBest feature selection before fit() and predict().
    Transparently reduces dimensionality for the wrapped model.

    ADDS: .selected_feature_indices — indices of chosen features.

    Args:
        model:      Any BaseModel-compatible instance.
        k:          Number of top features to retain (default 7).
        score_func: 'f_regression' | 'f_classif' | 'mutual_info' (default 'f_regression').
    """

    _SCORE_FUNC_MAP: Dict[str, str] = {
        "f_regression": "sklearn.feature_selection.f_regression",
        "f_classif":    "sklearn.feature_selection.f_classif",
        "mutual_info":  "sklearn.feature_selection.mutual_info_regression",
    }

    def __init__(self, model, k: int = 7, score_func: str = "f_regression") -> None:
        super().__init__(model)
        self._k = k
        self._score_func_name = score_func
        self._selector = None
        self._selected_indices: Optional[List[int]] = None

    def _resolve_score_func(self):
        import importlib
        if self._score_func_name not in self._SCORE_FUNC_MAP:
            raise ValueError(
                f"Unknown score_func '{self._score_func_name}'. "
                f"Choose from: {list(self._SCORE_FUNC_MAP)}"
            )
        module_path, fn_name = self._SCORE_FUNC_MAP[self._score_func_name].rsplit(".", 1)
        return getattr(importlib.import_module(module_path), fn_name)

    def fit(self, X, y) -> "FeatureSelectionDecorator":
        from sklearn.feature_selection import SelectKBest

        self._selector = SelectKBest(
            score_func=self._resolve_score_func(), k=self._k
        )
        X_sel = self._selector.fit_transform(X, y)
        self._selected_indices = list(self._selector.get_support(indices=True))

        logger.info(
            "[FeatureSelectionDecorator] Selected %d/%d features → indices %s",
            self._k, X.shape[1] if hasattr(X, "shape") else "?",
            self._selected_indices,
        )
        super().fit(X_sel, y)
        return self

    def predict(self, X):
        if self._selector is None:
            raise RuntimeError("Call fit() before predict().")
        return super().predict(self._selector.transform(X))

    @property
    def selected_feature_indices(self) -> Optional[List[int]]:
        """Indices of selected features (available after fit())."""
        return self._selected_indices

    @property
    def model_type(self) -> str:
        return f"FeatureSelected-K{self._k}({self._model.model_type})"
