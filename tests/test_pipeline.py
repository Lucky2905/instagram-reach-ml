"""
tests/test_pipeline.py — Data pipeline tests.
Verifies: generator output shape, null-free data, correct schema, tier ranges.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))


class TestInstagramDataGenerator:

    def test_generates_correct_number_of_rows(self):
        from src.data.generator import InstagramDataGenerator
        gen = InstagramDataGenerator(n_samples=200, random_state=42)
        df = gen.generate()
        assert len(df) == 200

    def test_no_null_values(self):
        from src.data.generator import InstagramDataGenerator
        gen = InstagramDataGenerator(n_samples=200, random_state=42)
        df = gen.generate()
        assert df.isnull().sum().sum() == 0, "Dataset contains null values"

    def test_required_columns_present(self):
        from src.data.generator import InstagramDataGenerator
        required = [
            "likes", "comments", "shares", "saves", "hashtag_count",
            "post_type", "hour_of_day", "day_of_week",
            "follower_count", "account_age_days", "reach", "reach_tier",
        ]
        gen = InstagramDataGenerator(n_samples=100, random_state=42)
        df = gen.generate()
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_reach_tier_values_are_valid(self):
        from src.data.generator import InstagramDataGenerator
        gen = InstagramDataGenerator(n_samples=300, random_state=1)
        df = gen.generate()
        assert set(df["reach_tier"].unique()).issubset({0, 1, 2})

    def test_post_type_values_are_valid(self):
        from src.data.generator import InstagramDataGenerator
        gen = InstagramDataGenerator(n_samples=100, random_state=42)
        df = gen.generate()
        assert set(df["post_type"].unique()).issubset({"image", "video", "reel"})

    def test_hour_of_day_range(self):
        from src.data.generator import InstagramDataGenerator
        gen = InstagramDataGenerator(n_samples=100, random_state=42)
        df = gen.generate()
        assert df["hour_of_day"].between(0, 23).all()

    def test_reach_is_non_negative(self):
        from src.data.generator import InstagramDataGenerator
        gen = InstagramDataGenerator(n_samples=200, random_state=42)
        df = gen.generate()
        assert (df["reach"] >= 0).all()

    def test_reproducibility(self):
        from src.data.generator import InstagramDataGenerator
        df1 = InstagramDataGenerator(100, 42).generate()
        df2 = InstagramDataGenerator(100, 42).generate()
        pd.testing.assert_frame_equal(df1, df2)

    def test_full_5000_rows_no_nulls(self):
        """Functional check: generate(5000) → no nulls."""
        from src.data.generator import InstagramDataGenerator
        gen = InstagramDataGenerator(n_samples=5000, random_state=42)
        df = gen.generate()
        assert len(df) == 5000
        assert df.isnull().sum().sum() == 0


class TestDataLoader:

    def test_validate_passes_on_good_df(self, tmp_path):
        from src.data.generator import InstagramDataGenerator
        from src.data.loader import DataLoader

        gen = InstagramDataGenerator(n_samples=50, random_state=42)
        df = gen.generate()
        path = tmp_path / "test.csv"
        df.to_csv(path, index=False)

        loader = DataLoader(path=path)
        loaded = loader.load()
        assert loader.validate(loaded)

    def test_validate_fails_on_missing_column(self):
        from src.data.loader import DataLoader
        import pandas as pd

        bad_df = pd.DataFrame({"likes": [1, 2, 3]})  # missing most columns
        loader = DataLoader.__new__(DataLoader)
        loader._path = Path("/nonexistent")
        assert not loader.validate(bad_df)


class TestFeatureEngineer:

    def test_output_shapes(self):
        from src.data.generator import InstagramDataGenerator
        from src.data.feature_engineer import FeatureEngineer

        gen = InstagramDataGenerator(n_samples=100, random_state=42)
        df = gen.generate()
        fe = FeatureEngineer()
        X, y_reg, y_cls = fe.transform(df)

        assert X.shape == (100, 11), f"Expected (100, 11), got {X.shape}"
        assert len(y_reg) == 100
        assert len(y_cls) == 100

    def test_no_nulls_in_features(self):
        from src.data.generator import InstagramDataGenerator
        from src.data.feature_engineer import FeatureEngineer

        gen = InstagramDataGenerator(n_samples=100, random_state=42)
        df = gen.generate()
        X, _, _ = FeatureEngineer().transform(df)
        assert not X.isnull().any().any()

    def test_post_type_encoded_correctly(self):
        from src.data.feature_engineer import FeatureEngineer
        import pandas as pd

        df = pd.DataFrame([
            {"likes":1,"comments":1,"shares":1,"saves":1,"hashtag_count":1,
             "post_type":"reel","hour_of_day":10,"day_of_week":1,
             "follower_count":1000,"account_age_days":100,"reach":100,"reach_tier":0},
            {"likes":1,"comments":1,"shares":1,"saves":1,"hashtag_count":1,
             "post_type":"video","hour_of_day":10,"day_of_week":1,
             "follower_count":1000,"account_age_days":100,"reach":100,"reach_tier":0},
            {"likes":1,"comments":1,"shares":1,"saves":1,"hashtag_count":1,
             "post_type":"image","hour_of_day":10,"day_of_week":1,
             "follower_count":1000,"account_age_days":100,"reach":100,"reach_tier":0},
        ])
        X, _, _ = FeatureEngineer().transform(df)
        assert X.iloc[0]["post_type_reel"]  == 1
        assert X.iloc[0]["post_type_video"] == 0
        assert X.iloc[1]["post_type_video"] == 1
        assert X.iloc[1]["post_type_reel"]  == 0
        assert X.iloc[2]["post_type_reel"]  == 0
        assert X.iloc[2]["post_type_video"] == 0
