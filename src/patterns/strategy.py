"""
src/patterns/strategy.py — Strategy Pattern
============================================
Interchangeable preprocessing algorithms behind a unified interface.

Key guarantee (pattern verification):
    Swapping NormalizeStrategy ↔ StandardizeStrategy requires ZERO changes
    in train.py. Change config.PREPROCESSING_STRATEGY only.

Available strategies:
    'normalize'   → MinMaxScaler  → output ∈ [0, 1]
    'standardize' → StandardScaler → μ=0, σ=1
    'robust'      → RobustScaler  → IQR-based, outlier-resistant
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Type
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


# ── Abstract Strategy ─────────────────────────────────────────────────────────

class PreprocessingStrategy(ABC):
    """Abstract base — concrete strategies implement fit() and transform()."""

    @abstractmethod
    def fit(self, X: pd.DataFrame) -> "PreprocessingStrategy":
        """Fit parameters to training data (in-place, returns self)."""

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Apply fitted transformation; return numpy array."""

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.fit(X).transform(X)

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy display name."""

    @property
    @abstractmethod
    def scaler(self):
        """Expose underlying sklearn scaler for joblib serialization."""


# ── Concrete Strategies ───────────────────────────────────────────────────────

class NormalizeStrategy(PreprocessingStrategy):
    """Min-Max normalization → scales features to [0, 1]."""

    def __init__(self):
        from sklearn.preprocessing import MinMaxScaler
        self._scaler = MinMaxScaler()

    def fit(self, X: pd.DataFrame) -> "NormalizeStrategy":
        self._scaler.fit(X)
        logger.info("[%s] Fitted on %d features.", self.name, X.shape[1])
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return self._scaler.transform(X)

    @property
    def name(self) -> str:
        return "NormalizeStrategy"

    @property
    def scaler(self):
        return self._scaler


class StandardizeStrategy(PreprocessingStrategy):
    """Z-score standardization → zero mean, unit variance."""

    def __init__(self):
        from sklearn.preprocessing import StandardScaler
        self._scaler = StandardScaler()

    def fit(self, X: pd.DataFrame) -> "StandardizeStrategy":
        self._scaler.fit(X)
        logger.info("[%s] Fitted on %d features.", self.name, X.shape[1])
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return self._scaler.transform(X)

    @property
    def name(self) -> str:
        return "StandardizeStrategy"

    @property
    def scaler(self):
        return self._scaler


class RobustScaleStrategy(PreprocessingStrategy):
    """IQR-based robust scaling — resistant to outliers."""

    def __init__(self):
        from sklearn.preprocessing import RobustScaler
        self._scaler = RobustScaler()

    def fit(self, X: pd.DataFrame) -> "RobustScaleStrategy":
        self._scaler.fit(X)
        logger.info("[%s] Fitted on %d features.", self.name, X.shape[1])
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return self._scaler.transform(X)

    @property
    def name(self) -> str:
        return "RobustScaleStrategy"

    @property
    def scaler(self):
        return self._scaler


# ── Context (Strategy Executor) ───────────────────────────────────────────────

class PreprocessingContext:
    """
    Executes the active strategy. Supports hot-swapping at runtime.
    train.py only ever references PreprocessingContext — never a concrete class.
    This satisfies the zero-change swap requirement.
    """

    _REGISTRY: Dict[str, Type[PreprocessingStrategy]] = {
        "normalize": NormalizeStrategy,
        "standardize": StandardizeStrategy,
        "robust": RobustScaleStrategy,
    }

    def __init__(self, strategy: PreprocessingStrategy):
        self._strategy = strategy

    @classmethod
    def from_name(cls, name: str) -> "PreprocessingContext":
        """
        Instantiate context from config string.
        This is the ONLY entry point used by train.py, so swapping
        strategies only requires changing config.PREPROCESSING_STRATEGY.
        """
        key = name.lower()
        if key not in cls._REGISTRY:
            raise ValueError(
                f"Unknown strategy '{name}'. Choose from: {list(cls._REGISTRY)}"
            )
        strategy_instance = cls._REGISTRY[key]()
        logger.info("[PreprocessingContext] Active strategy → %s", strategy_instance.name)
        return cls(strategy_instance)

    def set_strategy(self, strategy: PreprocessingStrategy) -> None:
        """Hot-swap the strategy without recreating the context."""
        logger.info("[PreprocessingContext] Swapping → %s", strategy.name)
        self._strategy = strategy

    def fit(self, X: pd.DataFrame) -> "PreprocessingContext":
        self._strategy.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return self._strategy.transform(X)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self._strategy.fit_transform(X)

    @property
    def active_strategy(self) -> str:
        return self._strategy.name

    @property
    def scaler(self):
        """Expose sklearn scaler for joblib persistence."""
        return self._strategy.scaler
