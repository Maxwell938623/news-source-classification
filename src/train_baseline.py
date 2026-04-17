#!/usr/bin/env python3
"""
train_baseline.py — Reproduce the exact TF-IDF + Logistic Regression baseline.

Baseline specification (from CIS 4190/5190 handout):
    Vectorizer:  TfidfVectorizer(stop_words='english', max_features=100)
    Classifier:  LogisticRegression(max_iter=100)
    Reported accuracy: ~0.6649

The fitted vectorizer and classifier are wrapped in a single sklearn Pipeline
so the same object handles both feature extraction and prediction.

Label encoding:
    0  =  FoxNews
    1  =  NBC

Inputs:
    data/processed/splits/train.csv
    data/processed/splits/test.csv

Outputs:
    models/baseline_pipeline.joblib        sklearn Pipeline (tfidf + lr)
    reports/metrics_baseline.json          full metric set
    reports/figures/baseline_confusion_matrix.png
    logs/train_baseline.log

Usage:
    python src/train_baseline.py
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # headless rendering — no display required
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
LOGS_DIR = PROJECT_ROOT / "logs"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Label encoding — must match split.py, preprocess.py, and train_best_model.py
LABEL_NAMES = {0: "FoxNews", 1: "NBC"}

#: Column in the split CSV that holds the text for the baseline.
#: We use headline_minimal because TfidfVectorizer lowercases internally (lowercase=True default).
TEXT_COLUMN = "headline_minimal"


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------

def build_baseline_pipeline() -> Pipeline:
    """
    Exact baseline as specified in the handout.

    TfidfVectorizer defaults used here:
        lowercase=True   (applied internally before tokenisation)
        norm='l2'
        use_idf=True
        smooth_idf=True
        sublinear_tf=False

    LogisticRegression defaults:
        solver='lbfgs'   (sklearn >= 0.22 default)
        penalty='l2'
        C=1.0
        multi_class='auto' → binary → one coef vector
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=100)),
        ("lr",    LogisticRegression(max_iter=100, random_state=42)),
    ])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return a comprehensive metric dictionary for binary classification."""
    class_names = [LABEL_NAMES[0], LABEL_NAMES[1]]  # ["FoxNews", "NBC"]
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "f1_foxnews": float(f1_score(y_true, y_pred, pos_label=0, average="binary")),
        "f1_nbc": float(f1_score(y_true, y_pred, pos_label=1, average="binary")),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro")),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro")),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    title: str = "Baseline Confusion Matrix",
) -> None:
    class_names = [LABEL_NAMES[0], LABEL_NAMES[1]]
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y_true, y_pred),
        display_labels=class_names,
    )
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title, fontsize=13, pad=12)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s  %(levelname)-8s  %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(LOGS_DIR / "train_baseline.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train_baseline() -> None:
    log = _setup_logging()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Load splits -------------------------------------------------
    for split_name in ("train.csv", "test.csv"):
        path = SPLITS_DIR / split_name
        if not path.exists():
            log.error("Split file not found: %s  — run split.py first.", path)
            sys.exit(1)

    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    test_df = pd.read_csv(SPLITS_DIR / "test.csv")
    log.info("Train size: %d  |  Test size: %d", len(train_df), len(test_df))

    if TEXT_COLUMN not in train_df.columns:
        log.error("Column '%s' not found in split CSV. Run preprocess.py first.", TEXT_COLUMN)
        sys.exit(1)

    X_train = train_df[TEXT_COLUMN].fillna("").tolist()
    y_train = train_df["label"].astype(int).to_numpy()
    X_test = test_df[TEXT_COLUMN].fillna("").tolist()
    y_test = test_df["label"].astype(int).to_numpy()

    # ---- Build and fit baseline pipeline ----------------------------
    log.info("Building baseline pipeline: TF-IDF(max_features=100) + LR(max_iter=100)")
    pipeline = build_baseline_pipeline()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pipeline.fit(X_train, y_train)
        if caught:
            for w in caught:
                log.warning("Training warning: %s", w.message)

    # ---- Evaluate on test set ---------------------------------------
    y_pred = pipeline.predict(X_test)
    metrics = compute_all_metrics(y_test, y_pred)

    log.info("=" * 56)
    log.info("BASELINE RESULTS (test split)")
    log.info("  Accuracy:          %.4f   (handout reports ~0.6649)", metrics["accuracy"])
    log.info("  F1 macro:          %.4f", metrics["f1_macro"])
    log.info("  F1 FoxNews:        %.4f", metrics["f1_foxnews"])
    log.info("  F1 NBC:            %.4f", metrics["f1_nbc"])
    log.info("  Precision macro:   %.4f", metrics["precision_macro"])
    log.info("  Recall macro:      %.4f", metrics["recall_macro"])
    log.info("=" * 56)

    # Full classification report
    cr = classification_report(
        y_test, y_pred,
        target_names=[LABEL_NAMES[0], LABEL_NAMES[1]],
    )
    log.info("Classification report:\n%s", cr)

    # ---- Top features -----------------------------------------------
    feature_names = pipeline.named_steps["tfidf"].get_feature_names_out()
    coef = pipeline.named_steps["lr"].coef_.ravel()
    top_n = 15
    top_nbc_idx = np.argsort(coef)[-top_n:][::-1]
    top_fox_idx = np.argsort(coef)[:top_n]
    log.info("Top %d features -> NBC  (positive coef):", top_n)
    for i in top_nbc_idx:
        log.info("  %+.3f  %s", coef[i], feature_names[i])
    log.info("Top %d features -> FoxNews  (negative coef):", top_n)
    for i in top_fox_idx:
        log.info("  %+.3f  %s", coef[i], feature_names[i])

    # ---- Save pipeline ----------------------------------------------
    pipeline_path = MODELS_DIR / "baseline_pipeline.joblib"
    joblib.dump(pipeline, pipeline_path)
    log.info("Saved pipeline: %s", pipeline_path)

    # ---- Save model-specific metadata (used by evaluate.py) ---------
    # Named baseline_pipeline_metadata.json so evaluate.py can auto-detect
    # the correct text column without conflicting with best_model_metadata.json.
    baseline_meta = {
        "model": "baseline",
        "prep_col": TEXT_COLUMN,
        "vectorizer": "TfidfVectorizer(stop_words='english', max_features=100)",
        "classifier": "LogisticRegression(max_iter=100)",
        "label_encoding": {"FoxNews": 0, "NBC": 1},
        "train_size": len(X_train),
        "test_size": len(X_test),
        "test_metrics": metrics,
    }
    meta_path = MODELS_DIR / "baseline_pipeline_metadata.json"
    meta_path.write_text(json.dumps(baseline_meta, indent=2), encoding="utf-8")
    log.info("Saved baseline metadata: %s", meta_path)

    # ---- Save metrics -----------------------------------------------
    metrics_out = {
        "model": "baseline",
        "text_column": TEXT_COLUMN,
        "vectorizer": "TfidfVectorizer(stop_words='english', max_features=100)",
        "classifier": "LogisticRegression(max_iter=100)",
        "label_encoding": {"FoxNews": 0, "NBC": 1},
        "train_size": len(X_train),
        "test_size": len(X_test),
        "metrics": metrics,
    }
    metrics_path = REPORTS_DIR / "metrics_baseline.json"
    metrics_path.write_text(json.dumps(metrics_out, indent=2), encoding="utf-8")
    log.info("Saved metrics: %s", metrics_path)

    # ---- Confusion matrix plot --------------------------------------
    plot_confusion_matrix(
        y_test, y_pred,
        output_path=FIGURES_DIR / "baseline_confusion_matrix.png",
        title=f"Baseline Confusion Matrix  (acc={metrics['accuracy']:.4f})",
    )
    log.info("Saved confusion matrix plot.")


if __name__ == "__main__":
    train_baseline()
