#!/usr/bin/env python3
"""
train_best_model.py — Run all model experiments, select the best,
retrain on train+val, and evaluate once on the held-out test split.

Model architectures live in src/models/ (one file per family):
    tfidf_logreg.py      TF-IDF word n-grams + Logistic Regression
    tfidf_svm.py         TF-IDF word n-grams + LinearSVC (+ calibrated)
    tfidf_nb.py          TF-IDF word n-grams + MultinomialNB
    char_ngram.py        Character n-gram TF-IDF (LR, SVM, combined)
    stylometric.py       15 handcrafted stylometric features (LR, SVM, RF, HGB)
    hybrid.py            TF-IDF + stylometric FeatureUnion
    voting_ensemble.py   Soft-voting over diverse base learners
    stacking_ensemble.py Stacking with LR / HistGBM meta-learner

Selection criterion: validation F1-macro  (robust to any class imbalance)

Final model is retrained on train+val combined so the vectoriser's IDF
weights use all available labelled data before final test evaluation.

Outputs:
    models/best_model.joblib
    models/best_model_metadata.json
    reports/experiment_results.csv
    reports/metrics_best.json
    reports/figures/experiment_comparison.png
    reports/figures/best_confusion_matrix.png
    reports/figures/top_features.png          (linear models only)
    logs/train_best_model.log

Usage:
    python src/train_best_model.py
    python src/train_best_model.py --fast   # skip slow model families
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from datetime import date
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.base import clone

# ---------------------------------------------------------------------------
# Paths — resolved relative to this file so the script can be invoked from
# any working directory.
# ---------------------------------------------------------------------------
_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))   # ensures "from models.X import …" resolves

PROJECT_ROOT = _SRC.parent
MODELS_DIR   = PROJECT_ROOT / "models"
REPORTS_DIR  = PROJECT_ROOT / "reports"
FIGURES_DIR  = REPORTS_DIR / "figures"
LOGS_DIR     = PROJECT_ROOT / "logs"

# ---------------------------------------------------------------------------
# Import shared base utilities
# ---------------------------------------------------------------------------
from models._base import (          # noqa: E402  (after sys.path insert)
    LABEL_NAMES,
    ModelConfig,
    compute_metrics,
    load_splits,
    run_experiments,
    setup_logging,
)

# ---------------------------------------------------------------------------
# Import per-family config builders
# ---------------------------------------------------------------------------
from models.tfidf_logreg      import get_configs as _lr_configs
from models.tfidf_svm         import get_configs as _svm_configs
from models.tfidf_nb          import get_configs as _nb_configs
from models.char_ngram        import get_configs as _char_configs
from models.stylometric       import get_configs as _stylo_configs
from models.hybrid            import get_configs as _hybrid_configs
from models.voting_ensemble   import get_configs as _voting_configs
from models.stacking_ensemble import get_configs as _stacking_configs
try:
    from models.sentence_embedding import get_configs as _sentence_embedding_configs
    _SENTENCE_EMBEDDING_AVAILABLE = True
except Exception:  # noqa: BLE001 — torch/torchvision DLL issues on some Windows envs
    _sentence_embedding_configs = lambda: []  # type: ignore[assignment]
    _SENTENCE_EMBEDDING_AVAILABLE = False


# ---------------------------------------------------------------------------
# Assemble experiment list
# ---------------------------------------------------------------------------

def build_all_configs(fast: bool = False) -> list[ModelConfig]:
    """
    Collect configs from every model family.

    --fast skips the slow families (char n-grams, stylometric, hybrid,
    ensembles) so a quick sanity-check run finishes in under a minute.
    """
    configs: list[ModelConfig] = []
    configs += _lr_configs()
    configs += _svm_configs()
    configs += _nb_configs()

    if not fast:
        configs += _char_configs()
        configs += _stylo_configs()
        configs += _hybrid_configs()
        configs += _voting_configs()
        configs += _stacking_configs()
        configs += _sentence_embedding_configs()

    return configs


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_experiment_comparison(results_df: pd.DataFrame,
                                 output_path: Path) -> None:
    TOP_N = 10
    top = (
        results_df.nlargest(TOP_N, "val_f1_macro")
        .sort_values("val_f1_macro", ascending=True)
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.barh(
        top["name"], top["val_f1_macro"],
        color="steelblue", edgecolor="white", height=0.72,
    )
    ax.set_xlabel("Validation macro-F1", fontsize=11)
    ax.set_title(
        f"Top {TOP_N} classical experiment configurations (by validation macro-F1)",
        fontsize=13, pad=10,
    )
    ax.axvline(x=0.6649, color="red", linestyle="--", linewidth=1.2,
               label="Handout reference baseline (~0.6649)")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim(0.6, 0.85)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{w:.4f}", va="center", ha="left", fontsize=9)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                            output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y_true, y_pred),
        display_labels=[LABEL_NAMES[0], LABEL_NAMES[1]],
    )
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title, fontsize=13, pad=12)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_top_features(estimator, output_path: Path,
                       log: logging.Logger, top_n: int = 20) -> None:
    """
    Plot the most discriminative features for linear Pipeline models.
    Gracefully skips ensembles and non-linear classifiers.
    """
    if not hasattr(estimator, "named_steps"):
        log.info("Top features: skipped (not a flat Pipeline).")
        return

    tfidf = estimator.named_steps.get("tfidf")
    clf   = estimator.named_steps.get("clf")
    if clf is None:
        clf = estimator.named_steps.get("lr")
    if tfidf is None or clf is None:
        log.info("Top features: skipped (tfidf or clf step not found).")
        return

    inner = clf
    if hasattr(clf, "calibrated_classifiers_"):
        inner = clf.calibrated_classifiers_[0].estimator

    if not hasattr(inner, "coef_"):
        log.info("Top features: skipped (no coef_ on %s).", type(inner).__name__)
        return

    feature_names = tfidf.get_feature_names_out()
    coef = inner.coef_.ravel()

    top_pos = np.argsort(coef)[-top_n:][::-1]
    top_neg = np.argsort(coef)[:top_n]

    fig, (ax_nbc, ax_fox) = plt.subplots(1, 2, figsize=(14, 6))
    ax_nbc.barh(feature_names[top_pos], coef[top_pos], color="steelblue")
    ax_nbc.set_title(f"Top {top_n} → NBC (label=1)", fontsize=11)
    ax_nbc.set_xlabel("Coefficient")
    ax_nbc.invert_yaxis()
    ax_fox.barh(feature_names[top_neg], np.abs(coef[top_neg]), color="tomato")
    ax_fox.set_title(f"Top {top_n} → FoxNews (label=0)", fontsize=11)
    ax_fox.set_xlabel("|Coefficient|")
    ax_fox.invert_yaxis()
    plt.suptitle("Most Discriminative Features (Best Model)", fontsize=13,
                 y=1.02)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved top features plot: %s", output_path)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s  %(levelname)-8s  %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(LOGS_DIR / "train_best_model.log",
                                encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train_best_model(fast: bool = False) -> None:
    log = _setup_logging()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if not _SENTENCE_EMBEDDING_AVAILABLE:
        log.warning(
            "sentence_embedding configs unavailable (torch/torchvision DLL issue); "
            "sentence-transformer experiments will be skipped."
        )

    # ---- Load splits -------------------------------------------------------
    train_df, val_df, test_df = load_splits()
    y_test = test_df["label"].astype(int).to_numpy()
    log.info(
        "Data — train:%d  val:%d  test:%d",
        len(train_df), len(val_df), len(test_df),
    )

    # ---- Build experiment list --------------------------------------------
    configs = build_all_configs(fast=fast)
    log.info("Total experiments: %d  (fast=%s)", len(configs), fast)

    # ---- Run all experiments on train → evaluate on val ------------------
    records = run_experiments(
        configs, train_df, val_df, log=log, tqdm_desc="Experiments"
    )

    if not records:
        log.error(
            "No experiments completed.  Ensure split CSVs contain the "
            "expected headline_* columns (run preprocess.py then split.py)."
        )
        sys.exit(1)

    results_df = pd.DataFrame(records).sort_values("val_f1_macro",
                                                    ascending=False)
    results_path = REPORTS_DIR / "experiment_results.csv"
    results_df.to_csv(results_path, index=False)
    log.info("Saved results table: %s", results_path)

    log.info("\nTop 10 experiments by val_f1_macro:")
    log.info(
        results_df[["name", "group", "val_f1_macro", "val_accuracy"]]
        .head(10)
        .to_string(index=False)
    )

    # ---- Select best config by val_f1_macro ------------------------------
    best_row  = results_df.iloc[0]
    best_name = best_row["name"]
    best_cfg  = next(c for c in configs if c.name == best_name)
    log.info(
        "Best: '%s'  (val_f1_macro=%.4f, group=%s)",
        best_name, best_row["val_f1_macro"], best_cfg.group,
    )

    # ---- Retrain best config on train + val combined ---------------------
    # The vectoriser is refitted on all available labelled data so IDF
    # weights are computed from the full train+val vocabulary.
    trainval_df = pd.concat([train_df, val_df], ignore_index=True)
    X_tv = trainval_df[best_cfg.prep_col].fillna("").tolist()
    y_tv = trainval_df["label"].astype(int).to_numpy()

    best_est = clone(best_cfg.estimator)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        best_est.fit(X_tv, y_tv)
    log.info("Retrained best model on train+val (%d samples).", len(y_tv))

    # ---- Final test evaluation  (ONE time, never used for selection) -----
    X_test     = test_df[best_cfg.prep_col].fillna("").tolist()
    y_test_pred = best_est.predict(X_test)
    test_m      = compute_metrics(y_test, y_test_pred)

    log.info("=" * 62)
    log.info("BEST MODEL — FINAL TEST RESULTS")
    log.info("  Experiment   : %s", best_name)
    log.info("  Group        : %s", best_cfg.group)
    log.info("  Preprocessing: %s", best_cfg.prep_col)
    log.info("  Accuracy     : %.4f", test_m["accuracy"])
    log.info("  F1 macro     : %.4f", test_m["f1_macro"])
    log.info("  F1 FoxNews   : %.4f", test_m["f1_foxnews"])
    log.info("  F1 NBC       : %.4f", test_m["f1_nbc"])
    log.info("=" * 62)

    # ---- Save best model pipeline ----------------------------------------
    model_path = MODELS_DIR / "best_model.joblib"
    joblib.dump(best_est, model_path)
    log.info("Saved model: %s", model_path)

    # ---- Save metadata (read by evaluate.py and predict.py) --------------
    meta_path = MODELS_DIR / "best_model_metadata.json"
    meta_path.write_text(
        json.dumps({
            "experiment_name":  best_name,
            "group":            best_cfg.group,
            "prep_col":         best_cfg.prep_col,
            "description":      best_cfg.description,
            "estimator_type":   type(best_est).__name__,
            "estimator_repr":   repr(best_est)[:600],
            "label_encoding":   {"FoxNews": 0, "NBC": 1},
            "train_date":       str(date.today()),
            "train_size":       len(X_tv),
            "test_size":        len(X_test),
            "val_metrics":      {
                k: float(best_row[k])
                for k in ("val_f1_macro", "val_accuracy",
                          "val_f1_fox", "val_f1_nbc")
            },
            "test_metrics": {
                k: v for k, v in test_m.items()
                if k not in ("confusion_matrix", "classification_report")
            },
        }, indent=2),
        encoding="utf-8",
    )
    log.info("Saved metadata: %s", meta_path)

    # ---- Save test metrics -----------------------------------------------
    metrics_path = REPORTS_DIR / "metrics_best.json"
    metrics_path.write_text(
        json.dumps({
            "model":      best_name,
            "text_column": best_cfg.prep_col,
            "train_size":  len(X_tv),
            "test_size":   len(X_test),
            "metrics": {
                k: v for k, v in test_m.items()
                if k != "classification_report"
            },
        }, indent=2),
        encoding="utf-8",
    )

    # ---- Plots -----------------------------------------------------------
    _plot_experiment_comparison(
        results_df,
        FIGURES_DIR / "experiment_comparison.png",
    )
    log.info("Saved experiment comparison plot.")

    _plot_confusion_matrix(
        y_test, y_test_pred,
        output_path=FIGURES_DIR / "best_confusion_matrix.png",
        title=(
            f"Best Model — Confusion Matrix  "
            f"(acc={test_m['accuracy']:.4f})"
        ),
    )
    log.info("Saved confusion matrix plot.")

    _plot_top_features(
        best_est,
        output_path=FIGURES_DIR / "top_features.png",
        log=log,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run all model experiments and train the best model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--fast",
        action="store_true",
        help=(
            "Skip slow model families (char n-grams, stylometric, hybrid, "
            "ensembles).  Runs only TF-IDF + LR/SVM/NB — useful for a "
            "quick sanity check."
        ),
    )
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    train_best_model(fast=args.fast)
