#!/usr/bin/env python3
"""
split.py — Create reproducible, stratified train / validation / test splits.

Split ratios:   70% train  /  15% validation  /  15% test
Method:         stratified by label  →  class proportions identical in every split
Random seed:    42  (fixed; change only if you want a fresh split)

Label encoding (consistent across all project scripts):
    0  =  FoxNews
    1  =  NBC

Outputs (data/processed/splits/):
    train.csv              full clean dataset for the training split
    val.csv                validation split
    test.csv               held-out test split  (DO NOT peek during development)
    split_metadata.json    sizes, ratios, class distribution, seed

Usage:
    python src/split.py
    python src/split.py --input data/processed/clean_headlines.csv --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SPLITS_DIR = PROCESSED_DIR / "splits"
LOGS_DIR = PROJECT_ROOT / "logs"

DEFAULT_INPUT = PROCESSED_DIR / "clean_headlines.csv"

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-9, "Ratios must sum to 1."


# ---------------------------------------------------------------------------
# Core split logic
# ---------------------------------------------------------------------------

def create_splits(
    df: pd.DataFrame,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform two successive stratified splits to produce train / val / test sets.

    Step 1: split df  →  train (train_ratio)  +  temp (1 - train_ratio)
    Step 2: split temp  →  val (val_ratio / (1-train_ratio))
                        +  test (test_ratio / (1-train_ratio))

    Stratification is on the 'label' column, preserving class balance in each split.
    """
    label_col = "label"
    if label_col not in df.columns:
        raise ValueError(f"Column '{label_col}' not found in dataframe.")

    # Step 1 — carve off the training portion
    temp_ratio = 1.0 - train_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_ratio,
        random_state=seed,
        stratify=df[label_col],
    )

    # Step 2 — split the remainder into val and test in the correct proportions.
    # test_ratio / (val_ratio + test_ratio) gives the fraction of temp that becomes test,
    # so that the final test set is exactly test_ratio of the total.
    temp_total = val_ratio + (1.0 - train_ratio - val_ratio)  # = 1 - train_ratio
    relative_test_size = (1.0 - train_ratio - val_ratio) / temp_total
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        random_state=seed,
        stratify=temp_df[label_col],
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# Metadata & saving
# ---------------------------------------------------------------------------

def _class_distribution(df: pd.DataFrame) -> dict:
    # Prefer human-readable source names; fall back to numeric labels if absent.
    if "source" in df.columns:
        counts = df["source"].value_counts().to_dict()
    else:
        counts = {str(k): v for k, v in df["label"].value_counts().to_dict().items()}
    total = len(df)
    return {
        src: {"count": int(cnt), "pct": round(100 * cnt / total, 2)}
        for src, cnt in counts.items()
    }


def build_metadata(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict:
    total = len(df_train) + len(df_val) + len(df_test)
    return {
        "label_encoding": {"FoxNews": 0, "NBC": 1},
        "seed": seed,
        "ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": test_ratio,
        },
        "sizes": {
            "total": total,
            "train": len(df_train),
            "val": len(df_val),
            "test": len(df_test),
        },
        "actual_ratios": {
            "train": round(len(df_train) / total, 4),
            "val": round(len(df_val) / total, 4),
            "test": round(len(df_test) / total, 4),
        },
        "class_distribution": {
            "train": _class_distribution(df_train),
            "val": _class_distribution(df_val),
            "test": _class_distribution(df_test),
        },
    }


def save_splits(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    output_dir: Path,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    df_train.to_csv(output_dir / "train.csv", index=False)
    df_val.to_csv(output_dir / "val.csv", index=False)
    df_test.to_csv(output_dir / "test.csv", index=False)

    metadata = build_metadata(
        df_train, df_val, df_test,
        seed, train_ratio, val_ratio, test_ratio,
    )
    (output_dir / "split_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def split(input_path: Path, output_dir: Path, seed: int) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[
            logging.FileHandler(LOGS_DIR / "split.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    log = logging.getLogger(__name__)

    if not input_path.exists():
        log.error("Input file not found: %s", input_path)
        sys.exit(1)

    log.info("Loading clean dataset from: %s", input_path)
    df = pd.read_csv(input_path, dtype=str)

    # Coerce label to integer for stratification
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    log.info("Total rows: %d  |  Classes: %s", len(df), df["label"].value_counts().to_dict())

    train_df, val_df, test_df = create_splits(df, seed, TRAIN_RATIO, VAL_RATIO)

    # Verify stratification held (class proportions should be within 1-2% of overall)
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        dist = split_df["label"].value_counts(normalize=True).to_dict()
        log.info(
            "%-6s  n=%4d  |  label-0=%.1f%%  label-1=%.1f%%",
            name,
            len(split_df),
            100 * dist.get(0, 0),
            100 * dist.get(1, 0),
        )

    save_splits(train_df, val_df, test_df, output_dir, seed, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)
    log.info("Splits saved to: %s", output_dir)
    log.info(
        "  train=%d  val=%d  test=%d",
        len(train_df), len(val_df), len(test_df),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Create stratified train/val/test splits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                   help="Cleaned headlines CSV.")
    p.add_argument("--output-dir", type=Path, default=SPLITS_DIR,
                   help="Directory for split CSV files.")
    p.add_argument("--seed", type=int, default=SEED,
                   help="Random seed for reproducibility.")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    split(input_path=args.input, output_dir=args.output_dir, seed=args.seed)
