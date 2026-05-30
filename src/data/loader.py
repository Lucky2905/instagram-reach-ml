"""
src/data/loader.py — DataLoader
================================
Loads raw Instagram post data from CSV, or auto-generates synthetic data
if the file doesn't exist. Single responsibility: data acquisition only.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Required schema for validation
_REQUIRED_COLUMNS = [
    "likes", "comments", "shares", "saves", "hashtag_count",
    "post_type", "hour_of_day", "day_of_week",
    "follower_count", "account_age_days",
    "reach", "reach_tier",
]


class DataLoader:
    """
    Loads Instagram post data from disk.
    Falls back to synthetic generation if the file is absent.

    Args:
        path: Path to the CSV file. Defaults to config.DATASET_PATH.
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        if path is None:
            sys.path.insert(0, str(Path(__file__).parents[2]))
            from config import DATASET_PATH
            path = DATASET_PATH
        self._path = Path(path)

    def load(self) -> pd.DataFrame:
        """
        Return the dataset as a DataFrame.
        Generates synthetic data if CSV is not found.
        """
        if self._path.exists():
            logger.info("[DataLoader] Loading %s", self._path)
            df = pd.read_csv(self._path)
            logger.info("[DataLoader] Loaded %d rows, %d cols.", *df.shape)
            return df

        logger.warning(
            "[DataLoader] File not found: %s. Generating synthetic data…",
            self._path,
        )
        return self._generate_and_save()

    def _generate_and_save(self) -> pd.DataFrame:
        sys.path.insert(0, str(Path(__file__).parents[2]))
        from config import N_SAMPLES, RANDOM_STATE
        from src.data.generator import InstagramDataGenerator

        gen = InstagramDataGenerator(n_samples=N_SAMPLES, random_state=RANDOM_STATE)
        df = gen.generate()
        gen.save(df, self._path)
        return df

    def validate(self, df: pd.DataFrame) -> bool:
        """
        Check that the DataFrame has the required columns and no null values.

        Returns:
            True if valid, False otherwise.
        """
        missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            logger.error("[DataLoader] Missing columns: %s", missing)
            return False
        if df.isnull().any().any():
            null_counts = df.isnull().sum()
            logger.error("[DataLoader] Null values found:\n%s", null_counts[null_counts > 0])
            return False
        logger.info("[DataLoader] Validation passed ✓ (%d rows, no nulls)", len(df))
        return True
