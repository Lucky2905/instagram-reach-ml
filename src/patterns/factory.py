"""
src/patterns/factory.py — Factory Pattern
==========================================
ModelFactory creates model instances by name from a central registry.
All model instantiation MUST go through this factory — never directly.

Why Factory here?
    Decouples train.py / API from concrete sklearn classes.
    Adding a new model = one .register() call, zero changes elsewhere.

Usage:
    import src.models                              # triggers self-registration
    model = ModelFactory.create("linear_regression")
    model = ModelFactory.create("random_forest", n_estimators=300)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Type
import logging

logger = logging.getLogger(__name__)


# ── Abstract Model Interface ──────────────────────────────────────────────────

class BaseModel(ABC):
    """
    Contract all models must satisfy.
    Single responsibility: define the predict/fit interface.
    """

    @abstractmethod
    def fit(self, X, y) -> "BaseModel":
        """Train the model on (X, y)."""

    @abstractmethod
    def predict(self, X):
        """Return predictions for X."""

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Return model hyperparameters as a dict."""

    @property
    @abstractmethod
    def model_type(self) -> str:
        """Human-readable model name (e.g. 'LinearRegression')."""


# ── Factory ───────────────────────────────────────────────────────────────────

class ModelFactory:
    """
    Central registry + factory for all model classes.
    Supports runtime model selection from config string or API param.
    """

    _registry: Dict[str, Type[BaseModel]] = {}

    @classmethod
    def register(cls, name: str, model_class: Type[BaseModel]) -> None:
        """Register a model class under a lookup key."""
        cls._registry[name.lower()] = model_class
        logger.debug("[ModelFactory] Registered → '%s'", name)

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseModel:
        """
        Instantiate a model by registered name.

        Args:
            name: Key used at registration time (case-insensitive).
            **kwargs: Forwarded to the model constructor as hyperparameters.

        Returns:
            A fresh, unfitted BaseModel instance.

        Raises:
            ValueError: If name is not in the registry.
        """
        key = name.lower()
        if key not in cls._registry:
            raise ValueError(
                f"[ModelFactory] Unknown model '{name}'. "
                f"Available: {list(cls._registry.keys())}"
            )
        logger.info("[ModelFactory] Creating '%s' | kwargs=%s", name, kwargs)
        return cls._registry[key](**kwargs)

    @classmethod
    def available(cls) -> list:
        """Return all registered model name keys."""
        return list(cls._registry.keys())
