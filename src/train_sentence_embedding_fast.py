#!/usr/bin/env python3
"""
Fast sentence-embedding benchmark on GPU.

Uses one embedding model (all-MiniLM-L6-v2) with a small classifier set for
quick turnaround:
  - LogisticRegression
  - LinearSVC
  - XGBoost
  - LightGBM
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

# Preload torch early to avoid Windows DLL init issues.
import torch
import joblib
import pandas as pd
from sklearn.base import clone

_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))

from models._base import compute_metrics, load_splits, run_experiments, setup_logging
from models.sentence_embedding import get_configs


PROJECT_ROOT = _SRC.parent
MODELS_DIR = PROJECT_ROOT / "models"


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Refusing to run non-GPU fast training.")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    log = setup_logging(__name__, "train_sentence_embedding_fast.log")
    log.info("CUDA device: %s", torch.cuda.get_device_name(0))

    train_df, val_df, test_df = load_splits()
    all_configs = get_configs()

    allowed_suffixes = ("_logreg", "_svm", "_xgboost", "_lightgbm")
    configs = [
        c
        for c in all_configs
        if c.name.startswith("SE_all-MiniLM-L6-v2_") and c.name.endswith(allowed_suffixes)
    ]
    if not configs:
        raise RuntimeError("No fast sentence-embedding configs were found.")

    log.info("Running %d fast configs.", len(configs))
    records = run_experiments(
        configs=configs,
        train_df=train_df,
        val_df=val_df,
        log=log,
        tqdm_desc="sentence_embedding_fast",
    )
    if not records:
        raise RuntimeError("No fast sentence-embedding experiments completed.")

    results_df = pd.DataFrame(records).sort_values("val_f1_macro", ascending=False)
    log.info(
        "\nFast results:\n%s",
        results_df[["name", "val_f1_macro", "val_accuracy"]].to_string(index=False),
    )

    best_row = results_df.iloc[0]
    best_name = best_row["name"]
    best_cfg = next(c for c in configs if c.name == best_name)
    log.info("Best fast config: %s (val_f1_macro=%.4f)", best_name, best_row["val_f1_macro"])

    trainval_df = pd.concat([train_df, val_df], ignore_index=True)
    X_tv = trainval_df[best_cfg.prep_col].fillna("").tolist()
    y_tv = trainval_df["label"].astype(int).to_numpy()
    best_est = clone(best_cfg.estimator)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        best_est.fit(X_tv, y_tv)

    X_test = test_df[best_cfg.prep_col].fillna("").tolist()
    y_test = test_df["label"].astype(int).to_numpy()
    y_pred = best_est.predict(X_test)
    test_m = compute_metrics(y_test, y_pred)

    model_out = MODELS_DIR / "sentence_embedding_fast_best.joblib"
    meta_out = MODELS_DIR / "sentence_embedding_fast_best_metadata.json"
    joblib.dump(best_est, model_out)
    meta_out.write_text(
        json.dumps(
            {
                "experiment_name": best_name,
                "group": "sentence_embedding_fast",
                "prep_col": best_cfg.prep_col,
                "description": best_cfg.description,
                "val_metrics": {
                    "val_f1_macro": float(best_row["val_f1_macro"]),
                    "val_accuracy": float(best_row["val_accuracy"]),
                },
                "test_metrics": {
                    k: v for k, v in test_m.items() if k != "classification_report"
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    log.info("Saved fast model: %s", model_out)
    log.info("Saved fast metadata: %s", meta_out)
    log.info("Test accuracy=%.4f | Test F1-macro=%.4f", test_m["accuracy"], test_m["f1_macro"])


if __name__ == "__main__":
    main()
