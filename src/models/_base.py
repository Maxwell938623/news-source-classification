"""
_base.py — Shared utilities imported by every model file in src/models/.

Provides:
  - ModelConfig  dataclass  (name, group, prep_col, estimator, …)
  - compute_metrics()
  - load_splits()
  - setup_logging()
  - run_experiments()    trains a list of ModelConfigs, returns result dicts
  - standalone_train()   reusable __main__ logic for each model file
"""
from __future__ import annotations

import json
import logging
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.base import clone

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SPLITS_DIR   = PROJECT_ROOT / "data" / "processed" / "splits"
MODELS_DIR   = PROJECT_ROOT / "models"
REPORTS_DIR  = PROJECT_ROOT / "reports"
FIGURES_DIR  = REPORTS_DIR / "figures"
LOGS_DIR     = PROJECT_ROOT / "logs"

LABEL_NAMES = {0: "FoxNews", 1: "NBC"}


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """
    One row in the experiment table.

    `estimator` must be an unfitted sklearn-compatible estimator that
    accepts a list of strings (headlines) as X.  Use sklearn.utils.clone()
    before fitting so configs can be safely reused.
    """
    name: str
    group: str
    prep_col: str       # column in the split CSV to feed as text input
    estimator: Any      # unfitted sklearn estimator (Pipeline, Voting, Stacking …)
    description: str = ""
    needs_nonneg: bool = False  # informational: True for MNB-based models


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    names = [LABEL_NAMES[0], LABEL_NAMES[1]]
    return {
        "accuracy":         float(accuracy_score(y_true, y_pred)),
        "f1_macro":         float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted":      float(f1_score(y_true, y_pred, average="weighted")),
        "f1_foxnews":       float(f1_score(y_true, y_pred, pos_label=0, average="binary")),
        "f1_nbc":           float(f1_score(y_true, y_pred, pos_label=1, average="binary")),
        "precision_macro":  float(precision_score(y_true, y_pred, average="macro")),
        "recall_macro":     float(recall_score(y_true, y_pred, average="macro")),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, target_names=names, output_dict=True
        ),
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for fname in ("train.csv", "val.csv", "test.csv"):
        p = SPLITS_DIR / fname
        if not p.exists():
            raise FileNotFoundError(
                f"Split not found: {p} — run split.py first."
            )
    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    val_df   = pd.read_csv(SPLITS_DIR / "val.csv")
    test_df  = pd.read_csv(SPLITS_DIR / "test.csv")
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(name: str, log_filename: str) -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s  %(levelname)-8s  %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(LOGS_DIR / log_filename, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiments(
    configs: list[ModelConfig],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    log: logging.Logger | None = None,
    tqdm_desc: str = "Training",
) -> list[dict]:
    """
    Train each config on train_df and evaluate on val_df.
    Returns a list of result dicts (one per config that ran successfully).

    Uses sklearn.utils.clone() so each run gets a fresh unfitted estimator,
    making it safe to share ModelConfig objects across callers.
    """
    try:
        from tqdm import tqdm
        wrap = lambda x, **kw: tqdm(x, **kw, file=sys.stdout)
    except ImportError:
        wrap = lambda x, **kw: x

    y_train = train_df["label"].astype(int).to_numpy()
    y_val   = val_df["label"].astype(int).to_numpy()
    records: list[dict] = []

    for cfg in wrap(configs, desc=tqdm_desc, unit="model"):
        if cfg.prep_col not in train_df.columns:
            if log:
                log.warning(
                    "Column '%s' missing — skipping '%s'.", cfg.prep_col, cfg.name
                )
            continue

        X_tr = train_df[cfg.prep_col].fillna("").tolist()
        X_va = val_df[cfg.prep_col].fillna("").tolist()

        estimator = clone(cfg.estimator)

        t0 = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught_w:
            warnings.simplefilter("always")
            estimator.fit(X_tr, y_train)
        for w in caught_w:
            if "ConvergenceWarning" in str(w.category) and log:
                log.debug("ConvergenceWarning in '%s': %s", cfg.name, str(w.message))

        y_val_pred = estimator.predict(X_va)
        elapsed = time.perf_counter() - t0

        m = compute_metrics(y_val, y_val_pred)
        records.append({
            "name":         cfg.name,
            "group":        cfg.group,
            "prep_col":     cfg.prep_col,
            "description":  cfg.description,
            "val_accuracy": m["accuracy"],
            "val_f1_macro": m["f1_macro"],
            "val_f1_fox":   m["f1_foxnews"],
            "val_f1_nbc":   m["f1_nbc"],
            "val_prec_mac": m["precision_macro"],
            "val_rec_mac":  m["recall_macro"],
            "elapsed_s":    round(elapsed, 3),
        })

    return records


# ---------------------------------------------------------------------------
# Reusable standalone __main__ helper
# ---------------------------------------------------------------------------

def standalone_train(
    group_name: str,
    get_configs_fn: Callable[[], list[ModelConfig]],
    log_filename: str,
) -> None:
    """
    Reusable __main__ logic for each model file.

    1. Trains all configs from get_configs_fn() on the train split.
    2. Selects the best config by val_f1_macro.
    3. Retrains on train+val combined.
    4. Evaluates once on the held-out test split.
    5. Saves the best model as models/<group_name>_best.joblib.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    log = setup_logging(__name__, log_filename)

    train_df, val_df, test_df = load_splits()
    configs = get_configs_fn()
    log.info("Running %d configs for '%s'.", len(configs), group_name)

    records = run_experiments(configs, train_df, val_df, log=log, tqdm_desc=group_name)
    if not records:
        log.error("No experiments completed — check preprocessing columns.")
        sys.exit(1)

    results_df = pd.DataFrame(records).sort_values("val_f1_macro", ascending=False)
    log.info(
        "\nTop 5 by val_f1_macro:\n%s",
        results_df[["name", "val_f1_macro", "val_accuracy"]].head(5).to_string(index=False),
    )

    best_row  = results_df.iloc[0]
    best_name = best_row["name"]
    best_cfg  = next(c for c in configs if c.name == best_name)
    log.info("Best: %s  (val_f1_macro=%.4f)", best_name, best_row["val_f1_macro"])

    # Retrain on train+val
    trainval_df = pd.concat([train_df, val_df], ignore_index=True)
    X_tv = trainval_df[best_cfg.prep_col].fillna("").tolist()
    y_tv = trainval_df["label"].astype(int).to_numpy()
    best_est = clone(best_cfg.estimator)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        best_est.fit(X_tv, y_tv)

    # Test evaluation (once)
    X_test = test_df[best_cfg.prep_col].fillna("").tolist()
    y_test = test_df["label"].astype(int).to_numpy()
    y_pred = best_est.predict(X_test)
    test_m = compute_metrics(y_test, y_pred)

    log.info("=" * 56)
    log.info("BEST (%s): %s", group_name, best_name)
    log.info("  val F1-macro  = %.4f", best_row["val_f1_macro"])
    log.info("  test accuracy = %.4f", test_m["accuracy"])
    log.info("  test F1-macro = %.4f", test_m["f1_macro"])
    log.info("=" * 56)
    log.info(
        "Classification report:\n%s",
        classification_report(
            y_test, y_pred, target_names=[LABEL_NAMES[0], LABEL_NAMES[1]]
        ),
    )

    # Persist
    model_out = MODELS_DIR / f"{group_name}_best.joblib"
    joblib.dump(best_est, model_out)
    meta_out = MODELS_DIR / f"{group_name}_best_metadata.json"
    meta_out.write_text(
        json.dumps({
            "group":            group_name,
            "experiment_name":  best_name,
            "prep_col":         best_cfg.prep_col,
            "description":      best_cfg.description,
            "label_encoding":   {"FoxNews": 0, "NBC": 1},
            "val_f1_macro":     float(best_row["val_f1_macro"]),
            "test_metrics":     {k: v for k, v in test_m.items()
                                 if k != "classification_report"},
        }, indent=2),
        encoding="utf-8",
    )
    log.info("Saved model : %s", model_out)
    log.info("Saved meta  : %s", meta_out)
