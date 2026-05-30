"""
src/data/generator.py — Synthetic Instagram dataset generator.

Generates 5,000 rows of realistic Instagram post data using numpy distributions.
No external API calls required — fully reproducible via random_state.

Features generated:
    likes, comments, shares, saves, hashtag_count, post_type,
    hour_of_day, day_of_week, follower_count, account_age_days

Targets computed:
    reach (continuous) — derived from engagement + post metadata
    reach_tier (0/1/2) — low / medium / high based on 33rd/66th percentile
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
POST_TYPES = ["image", "video", "reel"]
POST_TYPE_PROBS = [0.50, 0.20, 0.30]   # images dominate; reels growing
PEAK_HOURS = range(18, 22)              # 6 PM – 10 PM engagement peak


class InstagramDataGenerator:
    """
    Generates a synthetic Instagram post dataset with realistic statistical
    distributions and controlled correlations between features and reach.

    Responsibility: Data generation ONLY.
    Does not perform preprocessing or feature encoding.

    Args:
        n_samples (int): Number of rows to generate. Default 5000.
        random_state (int): Seed for full reproducibility. Default 42.
    """

    def __init__(self, n_samples: int = 5000, random_state: int = 42) -> None:
        self.n_samples = n_samples
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)

    # ── Private feature builders ──────────────────────────────────────────────

    def _follower_count(self) -> np.ndarray:
        """Log-normal follower distribution: 1K – 10M."""
        raw = self._rng.lognormal(mean=9.0, sigma=1.8, size=self.n_samples)
        return raw.clip(1_000, 10_000_000).astype(int)

    def _engagement(self, follower_count: np.ndarray) -> dict:
        """
        Likes, comments, shares, saves — correlated with followers via
        a beta-distributed engagement rate (typical ~2–5%).
        """
        rate = self._rng.beta(2.5, 60, size=self.n_samples)   # mean ≈ 4%
        noise = self._rng.uniform(0.85, 1.15, size=self.n_samples)

        likes = (follower_count * rate * noise).astype(int).clip(0)
        comments = (likes * self._rng.uniform(0.02, 0.08, self.n_samples)).astype(int)
        shares = (likes * self._rng.uniform(0.01, 0.05, self.n_samples)).astype(int)
        saves = (likes * self._rng.uniform(0.03, 0.12, self.n_samples)).astype(int)

        return {"likes": likes, "comments": comments, "shares": shares, "saves": saves}

    def _post_metadata(self) -> dict:
        """Hashtags, post type, timing, and account age."""
        return {
            "hashtag_count": self._rng.integers(0, 31, size=self.n_samples),
            "post_type": self._rng.choice(POST_TYPES, size=self.n_samples, p=POST_TYPE_PROBS),
            "hour_of_day": self._rng.integers(0, 24, size=self.n_samples),
            "day_of_week": self._rng.integers(0, 7, size=self.n_samples),
            "account_age_days": self._rng.integers(30, 3651, size=self.n_samples),
        }

    def _compute_reach(self, df: pd.DataFrame) -> np.ndarray:
        """
        Reach is a function of engagement signals, post type, and posting time.

        Multipliers applied:
            Reels  → ×1.6  (algorithmic boost)
            Videos → ×1.25
            Peak hours (18–21) → ×1.2
            Weekend (Sat/Sun)  → ×1.1
        """
        base = (
            df["follower_count"] * self._rng.uniform(0.08, 0.65, self.n_samples)
            + df["likes"] * 3.5
            + df["shares"] * 12
            + df["comments"] * 2.5
            + df["saves"] * 6
            + df["hashtag_count"] * 80
        )

        type_mult = np.where(
            df["post_type"] == "reel", 1.60,
            np.where(df["post_type"] == "video", 1.25, 1.0)
        )
        hour_mult = np.where(df["hour_of_day"].between(18, 21), 1.20, 1.0)
        day_mult = np.where(df["day_of_week"].isin([5, 6]), 1.10, 1.0)
        noise = self._rng.normal(1.0, 0.08, self.n_samples)

        return (base * type_mult * hour_mult * day_mult * noise).clip(0).astype(int)

    @staticmethod
    def _assign_tier(reach: np.ndarray) -> np.ndarray:
        """
        Assign reach tier using 33rd and 66th percentile thresholds.
            0 = low    (≤ p33)
            1 = medium (p33 < x ≤ p66)
            2 = high   (> p66)
        """
        p33 = np.percentile(reach, 33)
        p66 = np.percentile(reach, 66)
        return np.where(reach <= p33, 0, np.where(reach <= p66, 1, 2)).astype(int)

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(self) -> pd.DataFrame:
        """
        Generate and return the full synthetic dataset.

        Returns:
            pd.DataFrame with shape (n_samples, 12):
                features + 'reach' + 'reach_tier'
        """
        logger.info("Generating %d synthetic Instagram records...", self.n_samples)

        followers = self._follower_count()
        engagement = self._engagement(followers)
        metadata = self._post_metadata()

        df = pd.DataFrame({"follower_count": followers, **engagement, **metadata})
        df["reach"] = self._compute_reach(df)
        df["reach_tier"] = self._assign_tier(df["reach"].values)

        logger.info(
            "Dataset ready. Shape=%s | Tier distribution:\n%s",
            df.shape,
            df["reach_tier"].value_counts().sort_index().to_string(),
        )
        return df

    def save(self, df: pd.DataFrame, output_path: Path) -> Path:
        """Persist the dataset to CSV."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info("Dataset saved → %s", path)
        return path


# ── CLI entry-point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    sys.path.insert(0, str(Path(__file__).parents[2]))
    from config import DATASET_PATH, N_SAMPLES, RANDOM_STATE

    gen = InstagramDataGenerator(n_samples=N_SAMPLES, random_state=RANDOM_STATE)
    dataset = gen.generate()
    gen.save(dataset, DATASET_PATH)

    print("\nSample rows:")
    print(dataset.head(3).to_string())
    print("\nStatistics:")
    print(dataset.describe().round(1).to_string())
