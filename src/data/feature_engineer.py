"""
src/data/feature_engineer.py — FeatureEngineer
================================================
Transforms the raw DataFrame into a model-ready numeric feature matrix.
Single responsibility: encoding and feature construction only.
Scaling is NOT done here — that is the Strategy layer's job.

Output feature columns (11 total):
    likes, comments, shares, saves, hashtag_count,
    hour_of_day, day_of_week, follower_count, account_age_days,
    post_type_reel, post_type_video
    (post_type_image is the reference category, dropped to avoid multicollinearity)
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Canonical feature order — must match API input order
FEATURE_COLUMNS: List[str] = [
    "likes",
    "comments",
    "shares",
    "saves",
    "hashtag_count",
    "hour_of_day",
    "day_of_week",
    "follower_count",
    "account_age_days",
    "post_type_reel",
    "post_type_video",
]


class FeatureEngineer:
    """
    Encodes categorical features and returns (X, y_regression, y_classification).
    Does NOT mutate the input DataFrame.
    """

    def __init__(self) -> None:
        self._feature_columns: List[str] = FEATURE_COLUMNS

    def transform(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Encode and split a raw DataFrame into features + targets.

        Args:
            df: Raw DataFrame from DataLoader (must include 'reach' and 'reach_tier').

        Returns:
            X      — numeric feature DataFrame, shape (n, 11)
            y_reg  — regression target Series  (reach, float)
            y_cls  — classification target Series (reach_tier, int 0/1/2)
        """
        df = df.copy()

        # ── One-hot encode post_type ──────────────────────────────────────────
        # Produces: post_type_image, post_type_reel, post_type_video
        dummies = pd.get_dummies(df["post_type"], prefix="post_type", drop_first=False)
        for col in ("post_type_reel", "post_type_video"):
            df[col] = dummies.get(col, pd.Series(0, index=df.index)).astype(int)

        # ── Extract targets ───────────────────────────────────────────────────
        y_reg = df["reach"].astype(float)
        y_cls = df["reach_tier"].astype(int)

        # ── Build feature matrix ──────────────────────────────────────────────
        X = df[self._feature_columns].copy().astype(float)

        logger.info(
            "[FeatureEngineer] X=%s | reach=[%d, %d] | tier_dist=%s",
            X.shape,
            int(y_reg.min()),
            int(y_reg.max()),
            dict(y_cls.value_counts().sort_index()),
        )
        return X, y_reg, y_cls

    @property
    def feature_columns(self) -> List[str]:
        """Return the ordered list of feature column names."""
        return list(self._feature_columns)
