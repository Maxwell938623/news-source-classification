#!/usr/bin/env python3
from __future__ import annotations
import json
from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from models._base import compute_metrics, load_splits
from models.hybrid import _hybrid_pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def build_candidates():
    candidates = []
    for char_ng, char_mf, word_mf in [
        ((3, 5), 5_000, 5_000),
        ((3, 6), 5_000, 5_000),
        ((4, 6), 5_000, 5_000),
        ((4, 7), 5_000, 5_000),
        ((3, 6), 10_000, 10_000),
        ((4, 6), 10_000, 10_000),
    ]:
        for c in [0.5, 1.0, 1.5, 2.0]:
            for class_weight in [None, "balanced"]:
                tag = "bal" if class_weight == "balanced" else "none"
                name = (
                    f"TUNE_HYB_word12_char{char_ng[0]}{char_ng[1]}"
                    f"_wmf{word_mf}_cmf{char_mf}_C{c}_{tag}"
                )
                clf = LogisticRegression(
                    C=c,
                    max_iter=1500,
                    solver="lbfgs",
                    random_state=42,
                    class_weight=class_weight,
                )
                pipe = _hybrid_pipeline(
                    ngram=(1, 2),
                    word_mf=word_mf,
                    char_ngram=char_ng,
                    char_mf=char_mf,
                    clf=clf,
                )
                candidates.append((name, pipe))
    return candidates


def main():
    train_df, val_df, test_df = load_splits()
    X_train = train_df["headline_minimal"].fillna("").tolist()
    y_train = train_df["label"].astype(int).to_numpy()
    X_val = val_df["headline_minimal"].fillna("").tolist()
    y_val = val_df["label"].astype(int).to_numpy()
    X_test = test_df["headline_minimal"].fillna("").tolist()
    y_test = test_df["label"].astype(int).to_numpy()
    rows = []
    best_name = None
    best_est = None
    best_val = -1.0
    for name, est in build_candidates():
        m = clone(est)
        m.fit(X_train, y_train)
        y_val_pred = m.predict(X_val)
        val = compute_metrics(y_val, y_val_pred)
        row = {
            "name": name,
            "val_f1_macro": val["f1_macro"],
            "val_accuracy": val["accuracy"],
            "val_f1_fox": val["f1_foxnews"],
            "val_f1_nbc": val["f1_nbc"],
        }
        rows.append(row)
        if val["f1_macro"] > best_val:
            best_val = val["f1_macro"]
            best_name = name
            best_est = m
    results_df = pd.DataFrame(rows).sort_values("val_f1_macro", ascending=False)
    results_df.to_csv(REPORTS_DIR / "hybrid_tuning_results.csv", index=False)

    assert best_name is not None and best_est is not None
    trainval_df = pd.concat([train_df, val_df], ignore_index=True)
    X_tv = trainval_df["headline_minimal"].fillna("").tolist()
    y_tv = trainval_df["label"].astype(int).to_numpy()
    final_est = clone(best_est)
    final_est.fit(X_tv, y_tv)
    y_test_pred = final_est.predict(X_test)
    test_m = compute_metrics(y_test, y_test_pred)

    joblib.dump(final_est, MODELS_DIR / "best_model.joblib")
    meta = {
        "experiment_name": best_name,
        "group": "G6_hybrid_tuned",
        "prep_col": "headline_minimal",
        "description": "Retuned hybrid model sweep around prior best",
        "estimator_type": type(final_est).__name__,
        "label_encoding": {"FoxNews": 0, "NBC": 1},
        "train_date": str(date.today()),
        "train_size": len(X_tv),
        "test_size": len(X_test),
        "val_metrics": {
            "val_f1_macro": float(results_df.iloc[0]["val_f1_macro"]),
            "val_accuracy": float(results_df.iloc[0]["val_accuracy"]),
            "val_f1_fox": float(results_df.iloc[0]["val_f1_fox"]),
            "val_f1_nbc": float(results_df.iloc[0]["val_f1_nbc"]),
        },
        "test_metrics": {
            k: v for k, v in test_m.items()
            if k not in ("confusion_matrix", "classification_report")
        },
    }
    (MODELS_DIR / "best_model_metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    (REPORTS_DIR / "metrics_best.json").write_text(
        json.dumps(
            {
                "model": best_name,
                "text_column": "headline_minimal",
                "train_size": len(X_tv),
                "test_size": len(X_test),
                "metrics": {
                    k: v
                    for k, v in test_m.items()
                    if k != "classification_report"
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Best tuned model: {best_name}")
    print(
        f"val_f1_macro={results_df.iloc[0]['val_f1_macro']:.4f} "
        f"test_acc={test_m['accuracy']:.4f} test_f1_macro={test_m['f1_macro']:.4f}"
    )


if __name__ == "__main__":
    main()
