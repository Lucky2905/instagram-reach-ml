"""
src/api/routes.py — Flask Blueprints
======================================
REST API endpoints. All business logic delegated to Trainer / model layer.

Endpoints:
    GET  /           → Serve dashboard HTML
    GET  /health     → Model load status
    POST /predict    → Single-row prediction (validated JSON input)
    POST /train      → Trigger training from API
    GET  /metrics    → Return training history
    POST /predict-batch → Batch predictions from JSON array (for dashboard CSV upload)
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import joblib
from flask import Blueprint, jsonify, render_template, request

logger = logging.getLogger(__name__)
api_bp = Blueprint("api", __name__)

# ── Constants ─────────────────────────────────────────────────────────────────
TIER_LABELS: Dict[int, str] = {0: "low", 1: "medium", 2: "high"}

REQUIRED_FIELDS: Dict[str, type] = {
    "likes":           (int, float),
    "comments":        (int, float),
    "shares":          (int, float),
    "saves":           (int, float),
    "hashtag_count":   (int, float),
    "post_type":       str,
    "hour_of_day":     (int, float),
    "day_of_week":     (int, float),
    "follower_count":  (int, float),
    "account_age_days":(int, float),
}
VALID_POST_TYPES = {"image", "video", "reel"}

FEATURE_ORDER = [
    "likes", "comments", "shares", "saves", "hashtag_count",
    "hour_of_day", "day_of_week", "follower_count", "account_age_days",
    "post_type_reel", "post_type_video",
]

# ── Lazy model cache ──────────────────────────────────────────────────────────
_cache: Dict[str, Any] = {}


def _load_models() -> bool:
    """Load all .pkl models into memory. Cached after first successful load."""
    if _cache:
        return True

    _root = Path(__file__).parents[2]
    sys.path.insert(0, str(_root))
    from config import REGRESSOR_PKL, CLASSIFIER_PKL, PREPROCESSOR_PKL

    try:
        _cache["regressor"]   = joblib.load(REGRESSOR_PKL)
        _cache["classifier"]  = joblib.load(CLASSIFIER_PKL)
        _cache["preprocessor"]= joblib.load(PREPROCESSOR_PKL)
        logger.info("[API] Models loaded from disk.")
        return True
    except FileNotFoundError as e:
        logger.warning("[API] Model file missing: %s", e)
        return False


# ── Input validation ──────────────────────────────────────────────────────────

def _validate(data: dict) -> Tuple[bool, str]:
    """Validate a single prediction request dict. Returns (ok, error_msg)."""
    for field, expected in REQUIRED_FIELDS.items():
        if field not in data:
            return False, f"Missing required field: '{field}'"
        if not isinstance(data[field], expected):
            return False, f"Field '{field}' must be {expected}, got {type(data[field]).__name__}"

    if data["post_type"] not in VALID_POST_TYPES:
        return False, f"'post_type' must be one of {sorted(VALID_POST_TYPES)}"
    if not (0 <= int(data["hour_of_day"]) <= 23):
        return False, "'hour_of_day' must be 0–23"
    if not (0 <= int(data["day_of_week"]) <= 6):
        return False, "'day_of_week' must be 0–6"
    if data["follower_count"] < 0:
        return False, "'follower_count' must be non-negative"
    return True, ""


def _to_feature_df(data: dict) -> pd.DataFrame:
    """Convert a validated JSON dict to a 1-row feature DataFrame."""
    return pd.DataFrame([{
        "likes":            float(data["likes"]),
        "comments":         float(data["comments"]),
        "shares":           float(data["shares"]),
        "saves":            float(data["saves"]),
        "hashtag_count":    float(data["hashtag_count"]),
        "hour_of_day":      float(data["hour_of_day"]),
        "day_of_week":      float(data["day_of_week"]),
        "follower_count":   float(data["follower_count"]),
        "account_age_days": float(data["account_age_days"]),
        "post_type_reel":   1.0 if data["post_type"] == "reel"  else 0.0,
        "post_type_video":  1.0 if data["post_type"] == "video" else 0.0,
    }], columns=FEATURE_ORDER)


def _run_prediction(data: dict) -> dict:
    """Core prediction logic shared by /predict and /predict-batch."""
    X_df = _to_feature_df(data)
    X_scaled = _cache["preprocessor"].transform(X_df)

    reach = int(max(0, _cache["regressor"].predict(X_scaled)[0]))
    tier_int = int(_cache["classifier"].predict(X_scaled)[0])
    proba = _cache["classifier"].predict_proba(X_scaled)[0]
    confidence = round(float(proba[tier_int]), 4)

    return {
        "predicted_reach": reach,
        "reach_tier": TIER_LABELS[tier_int],
        "confidence": confidence,
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@api_bp.route("/", methods=["GET"])
def dashboard():
    """Serve the interactive web dashboard."""
    return render_template("index.html")


@api_bp.route("/health", methods=["GET"])
def health():
    loaded = _load_models()
    return jsonify({
        "status": "ok",
        "models_loaded": loaded,
        "message": "Models ready." if loaded else "Run `python src/train.py` first.",
    })


@api_bp.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict

    Body (JSON):
        { "likes": 500, "comments": 50, "shares": 10, "saves": 30,
          "hashtag_count": 10, "post_type": "reel",
          "hour_of_day": 19, "day_of_week": 5,
          "follower_count": 50000, "account_age_days": 730 }

    Response:
        { "predicted_reach": 34200, "reach_tier": "high", "confidence": 0.91 }

    Errors:
        400 — bad input (missing fields, wrong types, invalid values)
        503 — models not trained yet
    """
    if not _load_models():
        return jsonify({"error": "Models not trained. Run `python src/train.py` first."}), 503

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    ok, err = _validate(data)
    if not ok:
        return jsonify({"error": err}), 400

    try:
        return jsonify(_run_prediction(data))
    except Exception as exc:
        logger.exception("[/predict] Error: %s", exc)
        return jsonify({"error": "Prediction failed.", "detail": str(exc)}), 500


@api_bp.route("/predict-batch", methods=["POST"])
def predict_batch():
    """
    POST /predict-batch

    Body (JSON array): list of prediction request objects (same schema as /predict).

    Response:
        {
          "count": 100,
          "predictions": [ { "predicted_reach": ..., "reach_tier": ..., "confidence": ... }, ... ],
          "feature_importances": { "feature_name": importance_value, ... }
        }
    """
    if not _load_models():
        return jsonify({"error": "Models not trained. Run `python src/train.py` first."}), 503

    rows = request.get_json(silent=True)
    if not rows or not isinstance(rows, list):
        return jsonify({"error": "Body must be a JSON array of prediction objects."}), 400

    results = []
    errors = []
    for i, row in enumerate(rows):
        ok, err = _validate(row)
        if not ok:
            errors.append({"row": i, "error": err})
            continue
        try:
            results.append(_run_prediction(row))
        except Exception as exc:
            errors.append({"row": i, "error": str(exc)})

    # Feature importances from the classifier
    fi_raw = _cache["classifier"].feature_importances_
    feature_importances = dict(zip(FEATURE_ORDER, [round(float(v), 6) for v in fi_raw]))

    return jsonify({
        "count": len(results),
        "predictions": results,
        "feature_importances": feature_importances,
        "errors": errors,
    })


@api_bp.route("/metrics", methods=["GET"])
def metrics():
    """Return full training history from the JSON metrics log."""
    _root = Path(__file__).parents[2]
    sys.path.insert(0, str(_root))
    from config import METRICS_FILE

    if not METRICS_FILE.exists():
        return jsonify({"error": "No metrics yet. Run training first."}), 404

    with open(METRICS_FILE, encoding="utf-8") as f:
        records = json.load(f)

    completed = [r for r in records if r.get("event_type") == "TRAINING_COMPLETE"]
    return jsonify({
        "history": records,
        "latest": completed[-1] if completed else None,
        "run_count": len(completed),
    })


@api_bp.route("/train", methods=["POST"])
def train_api():
    """
    POST /train
    Optional body: { "strategy": "normalize" }
    Triggers a synchronous training run and returns metrics.
    """
    body = request.get_json(silent=True) or {}
    strategy = body.get("strategy", "standardize")

    try:
        from src.training.trainer import Trainer
        trainer = Trainer(strategy_name=strategy)
        result = trainer.train()
        _cache.clear()   # force model reload on next predict
        return jsonify({"status": "success", "metrics": result})
    except Exception as exc:
        logger.exception("[/train] Training error: %s", exc)
        return jsonify({"error": str(exc)}), 500
