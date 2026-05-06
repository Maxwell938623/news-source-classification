#!/usr/bin/env python3
"""
build_model_folders.py

Create per-model report folders under reports/model_breakdown and populate each
folder with:
  - metrics from existing report JSON files (if present)
  - full-test metrics computed from current split/test set
  - confusion matrix and ROC curve plots
  - copied related graphs that already exist in reports/figures
"""

from __future__ import annotations

import json
import shutil
import sys
import types
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODEL_BREAKDOWN_DIR = REPORTS_DIR / "model_breakdown"
MODELS_DIR = PROJECT_ROOT / "models"
SPLITS_TEST = PROJECT_ROOT / "data" / "processed" / "splits" / "test.csv"

LABEL_NAMES = {0: "FoxNews", 1: "NBC"}


def _safe_name(name: str) -> str:
    return (
        name.replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .lower()
    )


def _extract_features(text: str) -> list[float]:
    words = text.split()
    n_words = len(words)
    n_chars = len(text)
    n_alpha = sum(1 for c in text if c.isalpha()) or 1
    avg_word_len = float(np.mean([len(w) for w in words])) if words else 0.0
    return [
        float(n_words),
        float(n_chars),
        avg_word_len,
        float(text.count("!")),
        float(text.count("?")),
        float(text.count("...")),
        float(":" in text),
        float('"' in text or "'" in text),
        float("-" in text or "\u2013" in text or "\u2014" in text),
        float("(" in text),
        sum(1.0 for c in text if c.isupper()) / n_alpha,
        float(sum(1 for w in words if len(w) > 1 and w.isupper())),
        sum(1.0 for c in text if c.isdigit()) / max(n_chars, 1),
        float(sum(1 for w in words if w and w[0].isupper())) / max(n_words, 1),
        float(text.count(",")),
    ]


class StyloTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([_extract_features(t) for t in X], dtype=np.float64)


class SparseWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, transformer):
        self.transformer = transformer

    def fit(self, X, y=None):
        self.transformer.fit(X, y)
        return self

    def transform(self, X):
        out = self.transformer.transform(X)
        if sp.issparse(out):
            return out.tocsr()
        return sp.csr_matrix(np.asarray(out, dtype=np.float64))

    def fit_transform(self, X, y=None):
        out = self.transformer.fit_transform(X, y)
        if sp.issparse(out):
            return out.tocsr()
        return sp.csr_matrix(np.asarray(out, dtype=np.float64))


def register_pickle_compat() -> None:
    """
    Some model scripts were trained via `python src/models/<name>.py`, so custom
    classes were pickled under module `__main__`. Register those symbols to make
    loading robust from this script.
    """
    main_mod = sys.modules.get("__main__")
    if main_mod is None:
        return
    setattr(main_mod, "SparseWrapper", SparseWrapper)
    setattr(main_mod, "StyloTransformer", StyloTransformer)
    # Some pickles reference module paths from standalone runs.
    stylometric_mod = types.ModuleType("stylometric")
    setattr(stylometric_mod, "StyloTransformer", StyloTransformer)
    sys.modules.setdefault("stylometric", stylometric_mod)

    hybrid_mod = types.ModuleType("hybrid")
    setattr(hybrid_mod, "SparseWrapper", SparseWrapper)
    sys.modules.setdefault("hybrid", hybrid_mod)


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, Any]:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_foxnews": float(f1_score(y_true, y_pred, pos_label=0, average="binary")),
        "f1_nbc": float(f1_score(y_true, y_pred, pos_label=1, average="binary")),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro")),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro")),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
    }
    return out


def save_confusion_matrix(path: Path, y_true: np.ndarray, y_pred: np.ndarray, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y_true, y_pred),
        display_labels=[LABEL_NAMES[0], LABEL_NAMES[1]],
    )
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_roc_curve(path: Path, y_true: np.ndarray, y_score: np.ndarray, title: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copy2(src, dst)


def infer_with_sklearn(
    model_path: Path,
    texts: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    model = joblib.load(model_path)
    y_pred = model.predict(texts)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(texts)[:, 1]
    elif hasattr(model, "decision_function"):
        raw = model.decision_function(texts)
        y_score = 1.0 / (1.0 + np.exp(-raw))
    else:
        y_score = y_pred.astype(float)
    return y_pred.astype(int), y_score.astype(float)


def infer_with_transformer(
    model_dir: Path,
    texts: list[str],
    max_length: int = 96,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()
    device = torch.device("cpu")
    model.to(device)

    scores: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            scores.append(probs[:, 1])
    y_score = np.concatenate(scores, axis=0)
    y_pred = (y_score >= 0.5).astype(int)
    return y_pred, y_score


def build_one_sklearn_folder(
    name: str,
    model_path: Path,
    source_metrics_path: Path,
    text_col: str,
    test_df: pd.DataFrame,
    related_figures: list[Path],
) -> None:
    out_dir = MODEL_BREAKDOWN_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    if source_metrics_path.exists():
        copy_if_exists(source_metrics_path, out_dir / "metrics_source.json")

    y_true = test_df["label"].astype(int).to_numpy()
    texts = test_df[text_col].fillna("").astype(str).tolist()
    y_pred, y_score = infer_with_sklearn(model_path, texts)

    metrics = compute_metrics(y_true, y_pred, y_score)
    save_json(out_dir / "metrics_full_test.json", metrics)
    save_json(
        out_dir / "classification_report_full_test.json",
        classification_report(
            y_true, y_pred, target_names=[LABEL_NAMES[0], LABEL_NAMES[1]], output_dict=True
        ),
    )

    save_confusion_matrix(out_dir / "confusion_matrix_full_test.png", y_true, y_pred, f"{name} Confusion Matrix")
    save_roc_curve(out_dir / "roc_curve_full_test.png", y_true, y_score, f"{name} ROC Curve")

    for fig in related_figures:
        copy_if_exists(fig, out_dir / fig.name)


def build_one_transformer_folder(
    name: str,
    model_dir: Path,
    source_metrics_path: Path,
    text_col: str,
    test_df: pd.DataFrame,
    related_figures: list[Path],
) -> None:
    out_dir = MODEL_BREAKDOWN_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    if source_metrics_path.exists():
        copy_if_exists(source_metrics_path, out_dir / "metrics_source.json")

    y_true = test_df["label"].astype(int).to_numpy()
    texts = test_df[text_col].fillna("").astype(str).tolist()
    y_pred, y_score = infer_with_transformer(model_dir, texts)

    metrics = compute_metrics(y_true, y_pred, y_score)
    save_json(out_dir / "metrics_full_test.json", metrics)
    save_json(
        out_dir / "classification_report_full_test.json",
        classification_report(
            y_true, y_pred, target_names=[LABEL_NAMES[0], LABEL_NAMES[1]], output_dict=True
        ),
    )

    save_confusion_matrix(out_dir / "confusion_matrix_full_test.png", y_true, y_pred, f"{name} Confusion Matrix")
    save_roc_curve(out_dir / "roc_curve_full_test.png", y_true, y_score, f"{name} ROC Curve")

    for fig in related_figures:
        copy_if_exists(fig, out_dir / fig.name)


def main() -> None:
    register_pickle_compat()
    MODEL_BREAKDOWN_DIR.mkdir(parents=True, exist_ok=True)
    test_df = pd.read_csv(SPLITS_TEST)

    # Sklearn models
    build_one_sklearn_folder(
        name="baseline",
        model_path=MODELS_DIR / "baseline_pipeline.joblib",
        source_metrics_path=REPORTS_DIR / "metrics_baseline.json",
        text_col="headline_minimal",
        test_df=test_df,
        related_figures=[
            FIGURES_DIR / "baseline_confusion_matrix.png",
            FIGURES_DIR / "confusion_matrix_baseline.png",
            FIGURES_DIR / "top_features_baseline.png",
        ],
    )

    build_one_sklearn_folder(
        name="best_classical_hybrid",
        model_path=MODELS_DIR / "best_model.joblib",
        source_metrics_path=REPORTS_DIR / "metrics_best.json",
        text_col="headline_minimal",
        test_df=test_df,
        related_figures=[
            FIGURES_DIR / "best_confusion_matrix.png",
            FIGURES_DIR / "confusion_matrix_best.png",
            FIGURES_DIR / "experiment_comparison.png",
            FIGURES_DIR / "baseline_vs_best.png",
        ],
    )

    # Transformer models
    build_one_transformer_folder(
        name="distilbert",
        model_dir=MODELS_DIR / "distilbert_hf",
        source_metrics_path=REPORTS_DIR / "metrics_distilbert.json",
        text_col="headline_minimal",
        test_df=test_df,
        related_figures=[FIGURES_DIR / "confusion_matrix_distilbert.png"],
    )

    build_one_transformer_folder(
        name="deberta_v3_small",
        model_dir=MODELS_DIR / "deberta_small_hf",
        source_metrics_path=REPORTS_DIR / "metrics_deberta_small.json",
        text_col="headline_minimal",
        test_df=test_df,
        related_figures=[FIGURES_DIR / "confusion_matrix_deberta_small.png"],
    )

    # Additional standalone family-best models from src/models/* scripts.
    # Files are expected to be named like: tfidf_logreg_best.joblib
    for model_path in sorted(MODELS_DIR.glob("*_best.joblib")):
        name = _safe_name(model_path.stem)
        meta_path = model_path.with_name(f"{model_path.stem}_metadata.json")
        text_col = "headline_minimal"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                text_col = meta.get("prep_col", text_col)
            except Exception:
                pass
        build_one_sklearn_folder(
            name=name,
            model_path=model_path,
            source_metrics_path=meta_path,
            text_col=text_col,
            test_df=test_df,
            related_figures=[],
        )

    print(f"Created model breakdown folders in: {MODEL_BREAKDOWN_DIR}")


if __name__ == "__main__":
    main()
