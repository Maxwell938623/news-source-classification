#!/usr/bin/env python3
"""
predict.py — Run inference with the saved best model for leaderboard submission.

Applies the same preprocessing pipeline used during training, then runs the
classifier.  The text column is resolved from model metadata automatically
(or overridden via --text-col).

Label encoding:
    0  =  FoxNews
    1  =  NBC

Input CSV must contain at least one of:
  - A preprocessed headline column (headline_minimal, headline_lowercase, etc.)
  - A raw headline column (headline, raw_headline, or similar)
    → this script will apply minimal cleaning on the fly

Output CSV columns:
    url            (if present in input)
    raw_headline   (if present in input)
    predicted_label      0 or 1
    predicted_source     FoxNews or NBC

If the model supports predict_proba, two extra columns are added:
    prob_foxnews   probability of class 0
    prob_nbc       probability of class 1

Usage:
    # Predict on a CSV with preprocessed headline columns (e.g., your test split)
    python src/predict.py --input data/processed/splits/test.csv

    # Predict on a raw CSV (apply minimal cleaning on the fly)
    python src/predict.py --input path/to/new_headlines.csv --raw-col raw_headline

    # Custom model / output path
    python src/predict.py --input data/processed/splits/test.csv \\
                           --model models/best_model.joblib \\
                           --output reports/predictions.csv
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR  = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
LOGS_DIR    = PROJECT_ROOT / "logs"
SPLITS_DIR  = PROJECT_ROOT / "data" / "processed" / "splits"

DEFAULT_MODEL    = MODELS_DIR / "best_model.joblib"
DEFAULT_METADATA = MODELS_DIR / "best_model_metadata.json"
DEFAULT_INPUT    = SPLITS_DIR / "test.csv"
DEFAULT_OUTPUT   = REPORTS_DIR / "predictions.csv"

LABEL_NAMES = {0: "FoxNews", 1: "NBC"}

# ---------------------------------------------------------------------------
# On-the-fly minimal cleaning (mirrors preprocess.py::clean_minimal)
# ---------------------------------------------------------------------------
_HTML_TAG = re.compile(r"<[^>]+>")
_MULTI_WS = re.compile(r"\s+")


def _minimal_clean(text: str) -> str:
    text = html.unescape(text)
    text = _HTML_TAG.sub(" ", text)
    return _MULTI_WS.sub(" ", text).strip()


# ---------------------------------------------------------------------------
# Text column resolution
# ---------------------------------------------------------------------------

def _resolve_text_col(
    df: pd.DataFrame,
    text_col: str | None,
    metadata: dict,
    raw_col: str | None,
) -> tuple[list[str], str]:
    """
    Determine what text to feed into the pipeline.

    Resolution order:
      1. --text-col if provided and present in df
      2. prep_col from model metadata if present in df
      3. A known preprocessed column (headline_minimal, headline_lowercase, …)
      4. --raw-col if provided (apply minimal cleaning on the fly)
      5. A column whose name contains "headline"
      6. The first non-URL / non-label column

    Returns (list_of_texts, source_column_name).
    """
    # Check explicit/metadata candidates first (these are preprocessed columns).
    candidates = [
        text_col,
        metadata.get("prep_col"),
        "headline_minimal",
        "headline_lowercase",
        "headline_nopunct",
    ]
    for col in candidates:
        if col and col in df.columns:
            return df[col].fillna("").tolist(), col

    # --raw-col is checked BEFORE the generic "headline" fallback so that the
    # user's explicit request for on-the-fly cleaning is always honoured.
    if raw_col and raw_col in df.columns:
        texts = [_minimal_clean(t) for t in df[raw_col].fillna("").tolist()]
        return texts, raw_col

    # Generic fallback: any column whose name contains "headline".
    # This may be raw text — apply minimal cleaning defensively.
    for col in df.columns:
        if "headline" in col.lower():
            texts = [_minimal_clean(t) for t in df[col].fillna("").tolist()]
            return texts, col

    # Absolute fallback: second column (first is usually URL or ID)
    col = df.columns[min(1, len(df.columns) - 1)]
    logging.getLogger(__name__).warning(
        "Could not identify a headline column — falling back to '%s'.", col
    )
    return df[col].fillna("").tolist(), col


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict(
    model_path: Path,
    input_path: Path,
    output_path: Path,
    text_col: str | None,
    raw_col: str | None,
) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[
            logging.FileHandler(LOGS_DIR / "predict.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    log = logging.getLogger(__name__)

    # ---- Load model + metadata ----------------------------------------
    if not model_path.exists():
        log.error("Model not found: %s  — run train_best_model.py first.", model_path)
        sys.exit(1)

    pipeline = joblib.load(model_path)
    log.info("Loaded model: %s", model_path)

    metadata: dict = {}
    meta_path = model_path.parent / "best_model_metadata.json"
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        log.info(
            "Model metadata — experiment: %s  |  prep: %s",
            metadata.get("experiment_name", "?"),
            metadata.get("prep_col", "?"),
        )

    # ---- Load input --------------------------------------------------
    if not input_path.exists():
        log.error("Input file not found: %s", input_path)
        sys.exit(1)

    df = pd.read_csv(input_path, dtype=str)
    log.info("Input: %s  (%d rows)", input_path, len(df))

    # ---- Resolve text ------------------------------------------------
    texts, used_col = _resolve_text_col(df, text_col, metadata, raw_col)
    log.info("Text source column: '%s'", used_col)

    # ---- Predict (single forward pass) ---------------------------------
    # Attempt predict_proba first; if unsupported fall back to predict.
    # This avoids running the TF-IDF transform twice.
    proba: np.ndarray | None = None
    try:
        proba  = pipeline.predict_proba(texts)
        y_pred = np.argmax(proba, axis=1)
        log.info("Predictions via predict_proba (single forward pass).")
    except AttributeError:
        y_pred = pipeline.predict(texts)
        log.info("Predictions via predict (classifier has no predict_proba).")

    # ---- Build output DataFrame --------------------------------------
    out_df = pd.DataFrame()

    # Preserve useful columns from input if present
    for pass_col in ("url", "raw_headline"):
        if pass_col in df.columns:
            out_df[pass_col] = df[pass_col].values

    out_df["predicted_label"]  = y_pred
    out_df["predicted_source"] = pd.Series(y_pred).map(LABEL_NAMES)

    if proba is not None:
        out_df["prob_foxnews"] = np.round(proba[:, 0], 4)
        out_df["prob_nbc"]     = np.round(proba[:, 1], 4)

    # ---- Save predictions --------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    log.info("Predictions saved: %s  (%d rows)", output_path, len(out_df))

    # ---- Distribution summary ----------------------------------------
    dist = pd.Series(y_pred).map(LABEL_NAMES).value_counts()
    log.info("Prediction distribution:")
    for src, cnt in dist.items():
        log.info("  %-10s  %d  (%.1f%%)", src, cnt, 100 * cnt / len(y_pred))

    # ---- Accuracy (if true labels available) -------------------------
    if "label" in df.columns:
        try:
            labels_series = pd.to_numeric(df["label"], errors="coerce")
            valid_mask = labels_series.isin([0, 1])
            if valid_mask.all() and len(labels_series) == len(y_pred):
                from sklearn.metrics import accuracy_score, f1_score
                y_true = labels_series.astype(int).to_numpy()
                acc = accuracy_score(y_true, y_pred)
                f1  = f1_score(y_true, y_pred, average="macro")
                log.info("Accuracy vs ground truth: %.4f  |  F1 macro: %.4f", acc, f1)
            elif not valid_mask.all():
                log.warning(
                    "Skipping accuracy: %d rows have labels outside {0, 1}.",
                    (~valid_mask).sum(),
                )
        except Exception as exc:
            log.warning("Could not compute accuracy: %s", exc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate leaderboard predictions from a saved model pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",    type=Path, default=DEFAULT_MODEL,
                   help="Path to saved joblib pipeline.")
    p.add_argument("--input",    type=Path, default=DEFAULT_INPUT,
                   help="Input CSV of headlines to classify.")
    p.add_argument("--output",   type=Path, default=DEFAULT_OUTPUT,
                   help="Output path for predictions CSV.")
    p.add_argument("--text-col", type=str,  default=None,
                   help="Column containing preprocessed headline text (overrides metadata).")
    p.add_argument("--raw-col",  type=str,  default=None,
                   help="Column containing raw headline text (minimal cleaning applied on the fly).")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    predict(
        model_path=args.model,
        input_path=args.input,
        output_path=args.output,
        text_col=args.text_col,
        raw_col=args.raw_col,
    )
