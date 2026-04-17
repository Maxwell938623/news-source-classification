#!/usr/bin/env python3
"""
evaluate.py — Comprehensive evaluation and error analysis for any saved model.

Computes and saves:
  - Accuracy, Precision, Recall, F1 (macro, weighted, per-class)
  - Confusion matrix
  - Classification report
  - Top discriminative features (linear models only)
  - Error analysis: misclassified examples, headline length vs error rate,
    most confidently wrong predictions (models with predict_proba)
  - Comparison plot: baseline vs best model

Label encoding:
    0  =  FoxNews
    1  =  NBC

Usage:
    # Evaluate best model on test split (default)
    python src/evaluate.py

    # Evaluate baseline on test split
    python src/evaluate.py --model models/baseline_pipeline.joblib

    # Evaluate a model on a custom CSV
    python src/evaluate.py --model models/best_model.joblib \\
                           --data data/processed/splits/test.csv \\
                           --text-col headline_lowercase
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR  = PROJECT_ROOT / "data" / "processed" / "splits"
MODELS_DIR  = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
LOGS_DIR    = PROJECT_ROOT / "logs"

LABEL_NAMES = {0: "FoxNews", 1: "NBC"}

DEFAULT_MODEL    = MODELS_DIR / "best_model.joblib"
DEFAULT_METADATA = MODELS_DIR / "best_model_metadata.json"
DEFAULT_DATA     = SPLITS_DIR / "test.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_metadata(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def resolve_text_column(df: pd.DataFrame, text_col: str | None, metadata: dict) -> str:
    """
    Determine which column to use for text, in priority order:
      1. Explicit --text-col argument
      2. prep_col from model metadata JSON
      3. First available headline_* column
      4. First column that contains "headline"
    """
    if text_col and text_col in df.columns:
        return text_col
    if metadata.get("prep_col") and metadata["prep_col"] in df.columns:
        return metadata["prep_col"]
    for col in ("headline_minimal", "headline_lowercase", "headline_nopunct"):
        if col in df.columns:
            return col
    for col in df.columns:
        if "headline" in col.lower():
            return col
    raise ValueError(
        f"Cannot determine text column. Columns available: {list(df.columns)}"
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def full_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    names = [LABEL_NAMES[0], LABEL_NAMES[1]]
    return {
        "accuracy":            float(accuracy_score(y_true, y_pred)),
        "f1_macro":            float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted":         float(f1_score(y_true, y_pred, average="weighted")),
        "f1_foxnews":          float(f1_score(y_true, y_pred, pos_label=0, average="binary")),
        "f1_nbc":              float(f1_score(y_true, y_pred, pos_label=1, average="binary")),
        "precision_macro":     float(precision_score(y_true, y_pred, average="macro")),
        "precision_foxnews":   float(precision_score(y_true, y_pred, pos_label=0, average="binary")),
        "precision_nbc":       float(precision_score(y_true, y_pred, pos_label=1, average="binary")),
        "recall_macro":        float(recall_score(y_true, y_pred, average="macro")),
        "recall_foxnews":      float(recall_score(y_true, y_pred, pos_label=0, average="binary")),
        "recall_nbc":          float(recall_score(y_true, y_pred, pos_label=1, average="binary")),
        "confusion_matrix":    confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, target_names=names, output_dict=True
        ),
    }


# ---------------------------------------------------------------------------
# Feature importance extraction
# ---------------------------------------------------------------------------

def get_feature_importances(
    pipeline,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Extract feature names and a 1D importance/discriminability score from the pipeline.

    For LR / LinearSVC:  coef_.ravel()  (positive → NBC, negative → FoxNews)
    For MultinomialNB:   log P(feature | NBC) − log P(feature | FoxNews)
                         (positive → more probable in NBC)
    Returns (feature_names, scores) or (None, None) if unsupported.
    """
    # The baseline pipeline names its steps "tfidf"/"lr"; the best-model
    # pipeline names them "tfidf"/"clf".  Accept either convention.
    tfidf = pipeline.named_steps.get("tfidf")
    clf   = pipeline.named_steps.get("clf")
    if clf is None:
        clf = pipeline.named_steps.get("lr")
    if tfidf is None or clf is None:
        return None, None
    feature_names = tfidf.get_feature_names_out()

    # Unwrap CalibratedClassifierCV if present
    inner = clf
    if hasattr(clf, "calibrated_classifiers_"):
        inner = clf.calibrated_classifiers_[0].estimator

    if hasattr(inner, "coef_"):
        return feature_names, inner.coef_.ravel()

    if hasattr(inner, "feature_log_prob_"):
        # MNB: shape (n_classes, n_features); row 0 = FoxNews, row 1 = NBC
        scores = inner.feature_log_prob_[1] - inner.feature_log_prob_[0]
        return feature_names, scores

    return None, None


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _save_plot(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    title: str = "Confusion Matrix",
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y_true, y_pred),
        display_labels=[LABEL_NAMES[0], LABEL_NAMES[1]],
    )
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title, fontsize=13, pad=12)
    plt.tight_layout()
    _save_plot(fig, output_path)


def plot_feature_importances(
    feature_names: np.ndarray,
    scores: np.ndarray,
    output_path: Path,
    top_n: int = 20,
    title: str = "Top Discriminative Features",
) -> None:
    top_pos = np.argsort(scores)[-top_n:][::-1]
    top_neg = np.argsort(scores)[:top_n]

    fig, (ax_nbc, ax_fox) = plt.subplots(1, 2, figsize=(14, 6))

    ax_nbc.barh(feature_names[top_pos], scores[top_pos], color="steelblue")
    ax_nbc.set_title(f"Top {top_n} → NBC (label=1)", fontsize=11)
    ax_nbc.set_xlabel("Score (positive = NBC)")
    ax_nbc.invert_yaxis()

    ax_fox.barh(feature_names[top_neg], np.abs(scores[top_neg]), color="tomato")
    ax_fox.set_title(f"Top {top_n} → FoxNews (label=0)", fontsize=11)
    ax_fox.set_xlabel("|Score| (negative = FoxNews)")
    ax_fox.invert_yaxis()

    plt.suptitle(title, fontsize=13, y=1.01)
    plt.tight_layout()
    _save_plot(fig, output_path)


def plot_length_vs_error(
    texts: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    bins: int = 8,
) -> None:
    """
    Bin headlines by word count and plot error rate per bin.
    Reveals whether short/long headlines are systematically harder.
    """
    lengths = np.array([len(t.split()) for t in texts])
    errors  = (y_true != y_pred).astype(int)

    bin_edges = np.percentile(lengths, np.linspace(0, 100, bins + 1))
    bin_edges = np.unique(bin_edges)  # deduplicate in case of quantile ties
    if len(bin_edges) < 2:
        return  # not enough range to bin

    bin_idx   = np.digitize(lengths, bin_edges[:-1]) - 1
    bin_idx   = np.clip(bin_idx, 0, len(bin_edges) - 2)

    bin_errors = []
    bin_counts = []
    bin_labels = []
    for i in range(len(bin_edges) - 1):
        mask = bin_idx == i
        if mask.sum() == 0:
            continue
        bin_errors.append(errors[mask].mean())
        bin_counts.append(mask.sum())
        bin_labels.append(f"{int(bin_edges[i])}-{int(bin_edges[i+1])}")

    fig, ax1 = plt.subplots(figsize=(9, 4))
    x = np.arange(len(bin_labels))
    ax1.bar(x, bin_errors, color="steelblue", alpha=0.7, label="Error rate")
    ax1.set_ylabel("Error Rate", color="steelblue")
    ax1.set_xticks(x)
    ax1.set_xticklabels(bin_labels, rotation=30, ha="right")
    ax1.set_xlabel("Headline Word Count Range")
    ax1.set_title("Error Rate by Headline Length", fontsize=12)

    ax2 = ax1.twinx()
    ax2.plot(x, bin_counts, "o-", color="orange", label="# headlines")
    ax2.set_ylabel("# Headlines", color="orange")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

    plt.tight_layout()
    _save_plot(fig, output_path)


def plot_model_comparison(baseline_path: Path, best_path: Path, output_path: Path) -> None:
    """Bar chart comparing baseline vs best model on key metrics."""
    if not baseline_path.exists() or not best_path.exists():
        return

    base = json.loads(baseline_path.read_text(encoding="utf-8"))["metrics"]
    best = json.loads(best_path.read_text(encoding="utf-8"))["metrics"]

    metric_keys   = ["accuracy", "f1_macro", "f1_foxnews", "f1_nbc",
                     "precision_macro", "recall_macro"]
    metric_labels = ["Accuracy", "F1 Macro", "F1 FoxNews", "F1 NBC",
                     "Precision Macro", "Recall Macro"]

    base_vals = [base.get(k, 0) for k in metric_keys]
    best_vals = [best.get(k, 0) for k in metric_keys]

    x   = np.arange(len(metric_labels))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w / 2, base_vals, width=w, label="Baseline (TF-IDF 100 + LR)", color="lightcoral")
    ax.bar(x + w / 2, best_vals, width=w, label="Best Model",                  color="steelblue")

    ax.set_ylabel("Score")
    ax.set_title("Baseline vs Best Model — Test Set Metrics", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, rotation=20, ha="right")
    ax.set_ylim(0, 1.0)
    ax.legend()

    for xi, (bv, bstv) in enumerate(zip(base_vals, best_vals)):
        ax.text(xi - w / 2, bv + 0.01, f"{bv:.3f}", ha="center", va="bottom", fontsize=7.5)
        ax.text(xi + w / 2, bstv + 0.01, f"{bstv:.3f}", ha="center", va="bottom", fontsize=7.5)

    plt.tight_layout()
    _save_plot(fig, output_path)


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------

def error_analysis(
    df: pd.DataFrame,
    text_col: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pipeline,
    log: logging.Logger,
    output_dir: Path,
) -> None:
    """
    Detailed error analysis:
    1. Summary of misclassified headlines
    2. Headline length distribution for correct vs incorrect
    3. Top confidently wrong predictions (if predict_proba available)
    """
    mask_wrong = y_true != y_pred
    n_wrong    = mask_wrong.sum()
    n_total    = len(y_true)

    log.info("-" * 56)
    log.info("ERROR ANALYSIS")
    log.info("  Total test examples:    %d", n_total)
    log.info("  Misclassified:          %d  (%.1f%%)", n_wrong, 100 * n_wrong / n_total)

    wrong_df = df[mask_wrong].copy()
    wrong_df["true_label"]  = y_true[mask_wrong]
    wrong_df["pred_label"]  = y_pred[mask_wrong]
    wrong_df["true_source"] = wrong_df["true_label"].map(LABEL_NAMES)
    wrong_df["pred_source"] = wrong_df["pred_label"].map(LABEL_NAMES)

    # Breakdown by direction of error
    fox_pred_nbc = ((y_true == 0) & (y_pred == 1)).sum()
    nbc_pred_fox = ((y_true == 1) & (y_pred == 0)).sum()
    log.info("  FoxNews predicted as NBC: %d", fox_pred_nbc)
    log.info("  NBC predicted as FoxNews: %d", nbc_pred_fox)

    # Headline length stats: correct vs wrong
    texts  = df[text_col].fillna("").tolist()
    lengths = np.array([len(t.split()) for t in texts])
    log.info(
        "  Avg words — correct: %.1f  |  wrong: %.1f",
        lengths[~mask_wrong].mean() if (~mask_wrong).any() else 0,
        lengths[mask_wrong].mean()  if mask_wrong.any()    else 0,
    )

    # Sample misclassified
    sample_n = min(20, len(wrong_df))
    sample_path = output_dir / "misclassified_sample.csv"
    save_cols = [c for c in ["url", text_col, "true_source", "pred_source"] if c in wrong_df.columns]
    wrong_df.head(sample_n)[save_cols].to_csv(sample_path, index=False)
    log.info("  Saved %d misclassified examples to: %s", sample_n, sample_path)

    # Confidently wrong predictions (requires predict_proba)
    try:
        # Pipeline's predict_proba: only works if clf has predict_proba
        # LinearSVC wrapped in CalibratedClassifierCV supports it;
        # plain LinearSVC does not.
        proba = pipeline.predict_proba(df[text_col].fillna("").tolist())
        confidence = proba.max(axis=1)
        conf_wrong_idx = np.where(mask_wrong)[0]
        if len(conf_wrong_idx):
            conf_wrong_conf = confidence[conf_wrong_idx]
            top_conf_idx = conf_wrong_idx[np.argsort(conf_wrong_conf)[::-1][:10]]
            log.info("  Most confident errors (top 10):")
            for idx in top_conf_idx:
                headline = texts[idx][:80]
                log.info(
                    "    [true=%s / pred=%s / conf=%.2f]  %s",
                    LABEL_NAMES[y_true[idx]],
                    LABEL_NAMES[y_pred[idx]],
                    confidence[idx],
                    headline,
                )
    except AttributeError:
        log.info("  (Classifier does not support predict_proba — skipping confidence analysis.)")

    # Plot: length vs error rate
    plot_length_vs_error(
        texts=texts,
        y_true=y_true,
        y_pred=y_pred,
        output_path=FIGURES_DIR / "length_vs_error_rate.png",
    )
    log.info("-" * 56)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _setup_logging(log_name: str) -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s  %(levelname)-8s  %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(LOGS_DIR / "evaluate.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(log_name)


def evaluate(
    model_path: Path,
    data_path: Path,
    text_col: str | None,
    output_suffix: str = "",
) -> None:
    log = _setup_logging(__name__)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Load model --------------------------------------------------
    if not model_path.exists():
        log.error("Model not found: %s", model_path)
        sys.exit(1)
    pipeline = joblib.load(model_path)
    log.info("Loaded pipeline: %s", model_path)

    # ---- Load data ---------------------------------------------------
    if not data_path.exists():
        log.error("Data file not found: %s", data_path)
        sys.exit(1)
    df = pd.read_csv(data_path)
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(-1).astype(int)
    df = df[df["label"].isin([0, 1])]

    # ---- Resolve text column -----------------------------------------
    # Try model-specific metadata first (e.g. baseline_pipeline_metadata.json),
    # then fall back to best_model_metadata.json, then to static defaults.
    meta_stem = model_path.stem + "_metadata.json"
    meta = load_metadata(model_path.parent / meta_stem)
    if not meta:
        meta = load_metadata(model_path.parent / "best_model_metadata.json")
    resolved_col = resolve_text_column(df, text_col, meta)
    log.info("Using text column: '%s'", resolved_col)

    X = df[resolved_col].fillna("").tolist()
    y_true = df["label"].to_numpy()

    # ---- Predict -----------------------------------------------------
    y_pred = pipeline.predict(X)
    metrics = full_metrics(y_true, y_pred)

    # ---- Log metrics -------------------------------------------------
    log.info("=" * 56)
    log.info("EVALUATION RESULTS  —  %s", model_path.name)
    log.info("  Data:              %s  (n=%d)", data_path.name, len(y_true))
    log.info("  Text column:       %s", resolved_col)
    log.info("  Accuracy:          %.4f", metrics["accuracy"])
    log.info("  F1 macro:          %.4f", metrics["f1_macro"])
    log.info("  F1 weighted:       %.4f", metrics["f1_weighted"])
    log.info("  F1 FoxNews:        %.4f", metrics["f1_foxnews"])
    log.info("  F1 NBC:            %.4f", metrics["f1_nbc"])
    log.info("  Precision macro:   %.4f", metrics["precision_macro"])
    log.info("  Recall macro:      %.4f", metrics["recall_macro"])
    log.info("=" * 56)

    log.info(
        "Classification report:\n%s",
        classification_report(y_true, y_pred, target_names=[LABEL_NAMES[0], LABEL_NAMES[1]]),
    )

    # ---- Save metrics ------------------------------------------------
    sfx = f"_{output_suffix}" if output_suffix else ""
    metrics_path = REPORTS_DIR / f"metrics_eval{sfx}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log.info("Saved metrics: %s", metrics_path)

    # ---- Confusion matrix plot --------------------------------------
    plot_confusion_matrix(
        y_true, y_pred,
        output_path=FIGURES_DIR / f"confusion_matrix{sfx}.png",
        title=f"{model_path.stem} — Confusion Matrix  (acc={metrics['accuracy']:.4f})",
    )

    # ---- Feature importance plot ------------------------------------
    feat_names, feat_scores = get_feature_importances(pipeline)
    if feat_names is not None:
        plot_feature_importances(
            feat_names, feat_scores,
            output_path=FIGURES_DIR / f"top_features{sfx}.png",
            title=f"Top Features — {model_path.stem}",
        )
        log.info("Saved feature importance plot.")
    else:
        log.info("Feature importance: not available for this model type.")

    # ---- Error analysis ---------------------------------------------
    error_analysis(df, resolved_col, y_true, y_pred, pipeline, log, REPORTS_DIR)

    # ---- Baseline vs best comparison (if both exist) ----------------
    plot_model_comparison(
        baseline_path=REPORTS_DIR / "metrics_baseline.json",
        best_path=REPORTS_DIR / "metrics_best.json",
        output_path=FIGURES_DIR / "baseline_vs_best.png",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate a saved model pipeline with full metrics and error analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",    type=Path, default=DEFAULT_MODEL,
                   help="Path to saved joblib pipeline.")
    p.add_argument("--data",     type=Path, default=DEFAULT_DATA,
                   help="CSV file to evaluate on (must have 'label' column).")
    p.add_argument("--text-col", type=str,  default=None,
                   help="Column to use as text (auto-resolved from metadata if omitted).")
    p.add_argument("--suffix",   type=str,  default="",
                   help="Suffix appended to output file names (useful for multiple runs).")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    evaluate(
        model_path=args.model,
        data_path=args.data,
        text_col=args.text_col,
        output_suffix=args.suffix,
    )
