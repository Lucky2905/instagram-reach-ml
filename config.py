"""
config.py — Central configuration for the Instagram Reach ML system.
All paths, hyperparameters, and constants live here.
Import this module anywhere to access project-wide settings.
"""

from pathlib import Path

# ── Directory layout ──────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
SAVED_MODELS_DIR = BASE_DIR / "saved_models"
LOGS_DIR = BASE_DIR / "logs"
METRICS_DIR = BASE_DIR / "metrics"
DASHBOARD_DIR = BASE_DIR / "dashboard"

# Create critical dirs on import
for _dir in (DATA_DIR, SAVED_MODELS_DIR, LOGS_DIR, METRICS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ── Dataset configuration ────────────────────────────────────────────────────
DATASET_FILENAME = "instagram_posts.csv"
DATASET_PATH = DATA_DIR / DATASET_FILENAME
N_SAMPLES = 5000
RANDOM_STATE = 42
TEST_SIZE = 0.20

# ── Feature schema ───────────────────────────────────────────────────────────
NUMERIC_FEATURES = [
    "likes",
    "comments",
    "shares",
    "saves",
    "hashtag_count",
    "hour_of_day",
    "day_of_week",
    "follower_count",
    "account_age_days",
]
CATEGORICAL_FEATURES = ["post_type"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET_REGRESSION = "reach"
TARGET_CLASSIFICATION = "reach_tier"

POST_TYPES = ["image", "video", "reel"]

# ── Reach tier thresholds (percentile-based) ─────────────────────────────────
TIER_LOW_PERCENTILE = 33
TIER_HIGH_PERCENTILE = 66
TIER_LABELS = {0: "low", 1: "medium", 2: "high"}

# ── ML hyperparameters ───────────────────────────────────────────────────────
REGRESSOR_PARAMS: dict = {}  # LinearRegression — no hyperparameters required

CLASSIFIER_PARAMS: dict = {
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_split": 2,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# ── Saved model filenames ────────────────────────────────────────────────────
REGRESSOR_PKL = SAVED_MODELS_DIR / "reach_regressor.pkl"
CLASSIFIER_PKL = SAVED_MODELS_DIR / "tier_classifier.pkl"
PREPROCESSOR_PKL = SAVED_MODELS_DIR / "preprocessor.pkl"

# ── Flask API ─────────────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 5000
API_DEBUG = True

# ── Observer / Metrics ───────────────────────────────────────────────────────
METRICS_FILE = METRICS_DIR / "training_metrics.json"
