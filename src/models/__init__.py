"""
src/models/__init__.py
Importing this package triggers ModelFactory self-registration for all models.
Always import src.models before using ModelFactory.create().
"""
from src.models.regressor import ReachRegressor    # noqa: F401 — registers "linear_regression"
from src.models.classifier import TierClassifier   # noqa: F401 — registers "random_forest"
