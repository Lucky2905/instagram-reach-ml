"""
tests/test_models.py — Pattern and model tests.
Verifies: ModelFactory, all 4 design patterns, model contracts.
"""

import sys
from pathlib import Path
import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

import src.models  # noqa — triggers self-registration


def _make_data(n=150, n_features=11, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.random((n, n_features)) * 1000
    y_reg = rng.random(n) * 50_000
    y_cls = rng.integers(0, 3, n)
    return X, y_reg, y_cls


# ── Factory Pattern ───────────────────────────────────────────────────────────

class TestModelFactory:

    def test_create_linear_regression(self):
        from src.patterns.factory import ModelFactory
        model = ModelFactory.create("linear_regression")
        assert model.model_type == "LinearRegression"

    def test_create_random_forest(self):
        from src.patterns.factory import ModelFactory
        model = ModelFactory.create("random_forest", n_estimators=10, random_state=42)
        assert model.model_type == "RandomForestClassifier"

    def test_create_with_kwargs(self):
        from src.patterns.factory import ModelFactory
        model = ModelFactory.create("random_forest", n_estimators=50)
        assert model._model.n_estimators == 50

    def test_raises_on_unknown_model(self):
        from src.patterns.factory import ModelFactory
        with pytest.raises(ValueError, match="Unknown model"):
            ModelFactory.create("gradient_boosting_xyz")

    def test_available_lists_registered_models(self):
        from src.patterns.factory import ModelFactory
        available = ModelFactory.available()
        assert "linear_regression" in available
        assert "random_forest" in available


# ── Strategy Pattern ─────────────────────────────────────────────────────────

class TestPreprocessingStrategy:
    import pandas as pd

    def _df(self, n=60, cols=5):
        import pandas as pd
        rng = np.random.default_rng(0)
        return pd.DataFrame(
            rng.random((n, cols)) * 1000,
            columns=[f"f{i}" for i in range(cols)],
        )

    def test_normalize_output_range(self):
        from src.patterns.strategy import PreprocessingContext
        ctx = PreprocessingContext.from_name("normalize")
        X = self._df()
        scaled = ctx.fit_transform(X)
        assert scaled.min() >= 0.0
        assert scaled.max() <= 1.0 + 1e-9

    def test_standardize_zero_mean(self):
        from src.patterns.strategy import PreprocessingContext
        ctx = PreprocessingContext.from_name("standardize")
        X = self._df()
        scaled = ctx.fit_transform(X)
        assert abs(scaled.mean()) < 0.1

    def test_robust_strategy_works(self):
        from src.patterns.strategy import PreprocessingContext
        ctx = PreprocessingContext.from_name("robust")
        X = self._df()
        scaled = ctx.fit_transform(X)
        assert scaled.shape == X.shape

    def test_hot_swap_strategy(self):
        """Verify strategy swap without changing calling code."""
        from src.patterns.strategy import (
            PreprocessingContext, NormalizeStrategy, StandardizeStrategy,
        )
        ctx = PreprocessingContext.from_name("normalize")
        assert ctx.active_strategy == "NormalizeStrategy"
        ctx.set_strategy(StandardizeStrategy())
        assert ctx.active_strategy == "StandardizeStrategy"

    def test_unknown_strategy_raises(self):
        from src.patterns.strategy import PreprocessingContext
        with pytest.raises(ValueError):
            PreprocessingContext.from_name("magic_strategy")


# ── Observer Pattern ─────────────────────────────────────────────────────────

class TestObserverPattern:

    def test_observer_receives_events(self):
        from src.patterns.observer import (
            TrainingSubject, TrainingObserver, TrainingEvent,
        )

        received = []

        class RecordObserver(TrainingObserver):
            def update(self, event: TrainingEvent):
                received.append((event.event_type, event.data))
            @property
            def observer_name(self): return "RecordObserver"

        subject = TrainingSubject()
        subject.attach(RecordObserver())
        subject.notify("TRAINING_START", {"strategy": "standardize"})
        subject.notify("TRAINING_COMPLETE", {"r2": 0.88, "accuracy": 0.91})

        assert len(received) == 2
        assert received[0][0] == "TRAINING_START"
        assert received[1][0] == "TRAINING_COMPLETE"
        assert received[1][1]["r2"] == 0.88

    def test_multiple_observers_all_notified(self):
        from src.patterns.observer import TrainingSubject, TrainingObserver, TrainingEvent

        counts = [0, 0]

        class Obs1(TrainingObserver):
            def update(self, e): counts[0] += 1
            @property
            def observer_name(self): return "Obs1"

        class Obs2(TrainingObserver):
            def update(self, e): counts[1] += 1
            @property
            def observer_name(self): return "Obs2"

        s = TrainingSubject()
        s.attach(Obs1()); s.attach(Obs2())
        s.notify("ANY_EVENT", {})
        assert counts == [1, 1]

    def test_faulty_observer_does_not_crash_pipeline(self):
        from src.patterns.observer import TrainingSubject, TrainingObserver

        class BrokenObserver(TrainingObserver):
            def update(self, e): raise RuntimeError("Simulated observer crash")
            @property
            def observer_name(self): return "BrokenObserver"

        s = TrainingSubject()
        s.attach(BrokenObserver())
        # Should NOT raise
        s.notify("TEST", {"key": "value"})

    def test_file_observer_writes_json(self, tmp_path):
        from src.patterns.observer import FileMetricsObserver, TrainingEvent

        path = tmp_path / "metrics.json"
        obs = FileMetricsObserver(path)
        obs.update(TrainingEvent("TRAINING_COMPLETE", {"r2": 0.9, "accuracy": 0.85}))

        import json
        with open(path) as f:
            records = json.load(f)
        assert len(records) == 1
        assert records[0]["event_type"] == "TRAINING_COMPLETE"


# ── Decorator Pattern ────────────────────────────────────────────────────────

class TestDecoratorPattern:

    def test_cv_decorator_adds_cv_score_method(self):
        """Core pattern verification: cv_score() must NOT exist on base model."""
        from src.patterns.factory import ModelFactory
        from src.patterns.decorator import CrossValidationDecorator

        base = ModelFactory.create("linear_regression")
        assert not hasattr(base, "cv_score"), "Base model should NOT have cv_score()"

        decorated = CrossValidationDecorator(base, cv=3, scoring="r2")
        assert hasattr(decorated, "cv_score"), "Decorated model MUST have cv_score()"

    def test_cv_score_returns_float_after_fit(self):
        from src.patterns.factory import ModelFactory
        from src.patterns.decorator import CrossValidationDecorator

        X, y_reg, _ = _make_data()
        base = ModelFactory.create("linear_regression")
        dec = CrossValidationDecorator(base, cv=3, scoring="r2")
        dec.fit(X, y_reg)

        score = dec.cv_score()
        assert isinstance(score, float)
        assert -1.0 <= score <= 1.0

    def test_cv_score_raises_before_fit(self):
        from src.patterns.factory import ModelFactory
        from src.patterns.decorator import CrossValidationDecorator

        base = ModelFactory.create("linear_regression")
        dec = CrossValidationDecorator(base, cv=3)
        with pytest.raises(RuntimeError):
            dec.cv_score()

    def test_cv_decorator_wraps_classifier(self):
        from src.patterns.factory import ModelFactory
        from src.patterns.decorator import CrossValidationDecorator

        X, _, y_cls = _make_data()
        base = ModelFactory.create("random_forest", n_estimators=10, random_state=42)
        dec = CrossValidationDecorator(base, cv=3, scoring="accuracy")
        dec.fit(X, y_cls)
        assert isinstance(dec.cv_score(), float)

    def test_feature_selection_decorator_reduces_features(self):
        from src.patterns.factory import ModelFactory
        from src.patterns.decorator import FeatureSelectionDecorator

        X, y_reg, _ = _make_data()
        base = ModelFactory.create("linear_regression")
        dec = FeatureSelectionDecorator(base, k=5, score_func="f_regression")
        dec.fit(X, y_reg)

        assert dec.selected_feature_indices is not None
        assert len(dec.selected_feature_indices) == 5

    def test_decorator_predict_works_after_fit(self):
        from src.patterns.factory import ModelFactory
        from src.patterns.decorator import CrossValidationDecorator

        X, y_reg, _ = _make_data()
        base = ModelFactory.create("linear_regression")
        dec = CrossValidationDecorator(base, cv=3)
        dec.fit(X, y_reg)
        preds = dec.predict(X[:5])
        assert len(preds) == 5


# ── Model contracts ───────────────────────────────────────────────────────────

class TestModelContracts:

    def test_regressor_fit_predict_cycle(self):
        from src.models.regressor import ReachRegressor
        X, y_reg, _ = _make_data()
        m = ReachRegressor()
        m.fit(X, y_reg)
        preds = m.predict(X[:10])
        assert len(preds) == 10
        assert m.is_fitted

    def test_classifier_predict_proba(self):
        from src.models.classifier import TierClassifier
        X, _, y_cls = _make_data()
        m = TierClassifier(n_estimators=10, random_state=42)
        m.fit(X, y_cls)
        proba = m.predict_proba(X[:5])
        assert proba.shape == (5, 3)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_classifier_feature_importances_shape(self):
        from src.models.classifier import TierClassifier
        X, _, y_cls = _make_data()
        m = TierClassifier(n_estimators=10, random_state=42)
        m.fit(X, y_cls)
        assert len(m.feature_importances_) == 11  # 11 features

    def test_regressor_raises_if_not_fitted(self):
        from src.models.regressor import ReachRegressor
        m = ReachRegressor()
        with pytest.raises(RuntimeError):
            m.predict(np.zeros((1, 11)))
