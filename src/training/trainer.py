"""
src/training/trainer.py — Trainer
===================================
Orchestrates the full ML training pipeline end-to-end.
Composes: DataLoader → FeatureEngineer → PreprocessingContext → ModelFactory → Observers.
Single responsibility: training coordination only.

Observer pattern in action here:
    Trainer inherits/composes TrainingSubject and fires events.
    Models are never aware of observers — full decoupling.

Strategy pattern in action here:
    Trainer reads config.PREPROCESSING_STRATEGY and passes it to
    PreprocessingContext.from_name(). Swapping strategy = one config change.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    accuracy_score,
    classification_report,
)

logger = logging.getLogger(__name__)


class Trainer:
    """
    End-to-end training orchestrator.

    Args:
        strategy_name: Preprocessing strategy key ('standardize', 'normalize', 'robust').
        test_size:     Fraction of data used for evaluation (default 0.20).
        random_state:  Reproducibility seed (default 42).
    """

    def __init__(
        self,
        strategy_name: str = "standardize",
        test_size: float = 0.20,
        random_state: int = 42,
    ) -> None:
        # Add project root to path
        self._root = Path(__file__).parents[2]
        sys.path.insert(0, str(self._root))

        self._strategy_name = strategy_name
        self._test_size = test_size
        self._random_state = random_state

        # Set up observer infrastructure
        from src.patterns.observer import (
            TrainingSubject, ConsoleObserver, FileMetricsObserver,
        )
        from config import METRICS_FILE

        self._subject = TrainingSubject()
        self._subject.attach(ConsoleObserver())
        self._subject.attach(FileMetricsObserver(METRICS_FILE))

        # Populated after train()
        self.regressor = None
        self.classifier = None
        self.preprocessor = None
        self._feature_columns = None
        self.metrics: Dict[str, Any] = {}

    # ── Pipeline stages ───────────────────────────────────────────────────────

    def _load_and_engineer(self):
        from src.data.loader import DataLoader
        from src.data.feature_engineer import FeatureEngineer

        loader = DataLoader()
        df = loader.load()
        if not loader.validate(df):
            raise ValueError("Dataset validation failed. Check data integrity.")

        fe = FeatureEngineer()
        X, y_reg, y_cls = fe.transform(df)
        self._feature_columns = fe.feature_columns
        return X, y_reg, y_cls

    def _split(self, X, y_reg, y_cls):
        return train_test_split(
            X, y_reg, y_cls,
            test_size=self._test_size,
            random_state=self._random_state,
        )

    def _preprocess(self, X_train, X_test):
        from src.patterns.strategy import PreprocessingContext
        ctx = PreprocessingContext.from_name(self._strategy_name)
        X_train_s = ctx.fit_transform(X_train)
        X_test_s = ctx.transform(X_test)
        self.preprocessor = ctx
        return X_train_s, X_test_s

    def _train_regressor(self, X_train, y_train):
        """
        Factory Pattern in action:
        ModelFactory.create() is the ONLY place where a model is instantiated.
        """
        import src.models  # noqa: F401 — triggers self-registration
        from src.patterns.factory import ModelFactory

        model = ModelFactory.create("linear_regression")
        model.fit(X_train, y_train)
        return model

    def _train_classifier(self, X_train, y_train):
        import src.models  # noqa: F401 — triggers self-registration
        from src.patterns.factory import ModelFactory
        from config import CLASSIFIER_PARAMS

        model = ModelFactory.create("random_forest", **CLASSIFIER_PARAMS)
        model.fit(X_train, y_train)
        return model

    def _eval_regressor(self, model, X_test, y_test) -> Dict[str, float]:
        preds = model.predict(X_test)
        mae = float(mean_absolute_error(y_test, preds))
        r2 = float(r2_score(y_test, preds))
        logger.info("[Trainer] Regressor → MAE=%.0f | R²=%.4f", mae, r2)
        return {"mae": round(mae, 2), "r2": round(r2, 4)}

    def _eval_classifier(self, model, X_test, y_test) -> Dict[str, Any]:
        preds = model.predict(X_test)
        acc = float(accuracy_score(y_test, preds))
        report = classification_report(
            y_test, preds, target_names=["low", "medium", "high"]
        )
        logger.info("[Trainer] Classifier → Accuracy=%.4f\n%s", acc, report)
        return {"accuracy": round(acc, 4), "classification_report": report}

    def _save_models(self) -> None:
        import config as cfg
        joblib.dump(self.regressor, cfg.REGRESSOR_PKL)
        joblib.dump(self.classifier, cfg.CLASSIFIER_PKL)
        joblib.dump(self.preprocessor, cfg.PREPROCESSOR_PKL)
        logger.info("[Trainer] Models saved → %s", cfg.SAVED_MODELS_DIR)

    # ── Public entry point ────────────────────────────────────────────────────

    def train(self) -> Dict[str, Any]:
        """
        Execute the full training pipeline.

        Observer events fired (models are NEVER aware of these):
            TRAINING_START → TRAINING_COMPLETE (or ERROR on failure)

        Returns:
            dict with 'regression' and 'classification' metric sub-dicts.
        """
        self._subject.notify("TRAINING_START", {
            "strategy": self._strategy_name,
            "test_size": self._test_size,
            "random_state": self._random_state,
        })

        try:
            logger.info("=" * 60)
            logger.info("STAGE 1 / 4  Loading data & engineering features...")
            X, y_reg, y_cls = self._load_and_engineer()

            logger.info("STAGE 2 / 4  Splitting & preprocessing (strategy=%s)...",
                        self._strategy_name)
            X_tr, X_te, yr_tr, yr_te, yc_tr, yc_te = self._split(X, y_reg, y_cls)
            X_tr_s, X_te_s = self._preprocess(X_tr, X_te)

            logger.info("STAGE 3 / 4  Training models via ModelFactory...")
            self.regressor = self._train_regressor(X_tr_s, yr_tr)
            self.classifier = self._train_classifier(X_tr_s, yc_tr)

            logger.info("STAGE 4 / 4  Evaluating & persisting models...")
            reg_m = self._eval_regressor(self.regressor, X_te_s, yr_te)
            cls_m = self._eval_classifier(self.classifier, X_te_s, yc_te)

            self.metrics = {"regression": reg_m, "classification": cls_m}
            self._save_models()

            self._subject.notify("TRAINING_COMPLETE", {
                "r2": reg_m["r2"],
                "mae": reg_m["mae"],
                "accuracy": cls_m["accuracy"],
                "strategy": self._strategy_name,
            })

            return self.metrics

        except Exception as exc:
            self._subject.notify("ERROR", {"message": str(exc)})
            logger.exception("[Trainer] Pipeline failed: %s", exc)
            raise
