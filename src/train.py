"""
src/train.py — Training CLI entry point.

Usage:
    python src/train.py
    python src/train.py --strategy normalize
    python src/train.py --strategy robust --test-size 0.15

Success criteria:
    R²       ≥ 0.75  (regression)
    Accuracy ≥ 0.80  (classification)
    .pkl files saved to ./saved_models/

Strategy swap (pattern verification):
    Change --strategy flag only — zero changes inside Trainer or models.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ── Ensure project root is importable ────────────────────────────────────────
ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Instagram Reach ML models (regression + classification).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--strategy",
        default="standardize",
        choices=["normalize", "standardize", "robust"],
        help="Preprocessing strategy.",
    )
    p.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        dest="test_size",
        help="Fraction of data for test split.",
    )
    p.add_argument(
        "--random-state",
        type=int,
        default=42,
        dest="random_state",
        help="Random seed for reproducibility.",
    )
    return p.parse_args()


def print_banner(strategy: str) -> None:
    print("\n" + "=" * 62)
    print("  Instagram Reach ML -- Training Pipeline")
    print(f"  Strategy : {strategy}")
    print("=" * 62)


def print_results(metrics: dict) -> None:
    reg = metrics["regression"]
    cls = metrics["classification"]

    print("\n" + "=" * 62)
    print("  TRAINING RESULTS")
    print("-" * 62)
    print(f"  Regression     | MAE : {reg['mae']:>12,.0f}  |  R2: {reg['r2']:.4f}")
    print(f"  Classification | Accuracy : {cls['accuracy']:.4f} ({cls['accuracy']*100:.2f}%)")
    print("-" * 62)
    print(cls.get("classification_report", ""))
    print("=" * 62)


def check_criteria(metrics: dict) -> bool:
    reg = metrics["regression"]
    cls = metrics["classification"]

    r2_pass  = reg["r2"]       >= 0.75
    acc_pass = cls["accuracy"] >= 0.80

    if r2_pass and acc_pass:
        print("\n[SUCCESS] R2 >= 0.75 and Accuracy >= 0.80 both met.")
    else:
        if not r2_pass:
            print(f"\n[WARN] R2 ({reg['r2']:.4f}) is below the 0.75 threshold.")
        if not acc_pass:
            print(f"\n[WARN] Accuracy ({cls['accuracy']:.4f}) is below the 0.80 threshold.")
    return r2_pass and acc_pass


def main() -> int:
    args = parse_args()
    print_banner(args.strategy)

    from src.training.trainer import Trainer

    trainer = Trainer(
        strategy_name=args.strategy,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    metrics = trainer.train()

    print_results(metrics)
    passed = check_criteria(metrics)

    print(f"\n  Models saved -> {ROOT / 'saved_models'}\n")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
