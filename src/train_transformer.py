#!/usr/bin/env python3
"""
train_transformer.py — Train/evaluate a transformer baseline on project splits.

Designed for the locked SOTA report phase:
  - DistilBERT baseline
  - DeBERTa-v3-small baseline

Usage examples:
  python src/train_transformer.py --model-name distilbert-base-uncased --run-name distilbert
  python src/train_transformer.py --model-name microsoft/deberta-v3-small --run-name deberta_small
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

LABEL_NAMES = {0: "FoxNews", 1: "NBC"}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(run_name: str) -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[
            logging.FileHandler(LOGS_DIR / f"train_transformer_{run_name}.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


def load_split(path: Path, text_col: str) -> tuple[list[str], np.ndarray]:
    df = pd.read_csv(path)
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' missing in {path}")
    texts = df[text_col].fillna("").astype(str).tolist()
    labels = pd.to_numeric(df["label"], errors="coerce").astype(int).to_numpy()
    return texts, labels


def maybe_subsample(
    texts: list[str],
    labels: np.ndarray,
    max_samples: int | None,
    seed: int,
) -> tuple[list[str], np.ndarray]:
    if max_samples is None or max_samples <= 0 or len(labels) <= max_samples:
        return texts, labels
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(labels), size=max_samples, replace=False)
    idx = np.sort(idx)
    return [texts[i] for i in idx], labels[idx]


def compute_metrics_from_logits(eval_pred) -> dict:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro")),
        "f1_foxnews": float(f1_score(labels, preds, pos_label=0, average="binary")),
        "f1_nbc": float(f1_score(labels, preds, pos_label=1, average="binary")),
        "precision_macro": float(precision_score(labels, preds, average="macro")),
        "recall_macro": float(recall_score(labels, preds, average="macro")),
    }


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y_true, y_pred),
        display_labels=[LABEL_NAMES[0], LABEL_NAMES[1]],
    )
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def to_hf_dataset(tokenizer, texts: list[str], labels: np.ndarray, max_length: int):
    from datasets import Dataset

    ds = Dataset.from_dict({"text": texts, "label": labels.tolist()})

    def _tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    ds = ds.map(_tok, batched=True, remove_columns=["text"])
    return ds


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train/evaluate a transformer model on the news source splits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--text-col", type=str, default="headline_minimal")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    log = setup_logging(args.run_name)

    train_texts, train_labels = load_split(SPLITS_DIR / "train.csv", args.text_col)
    val_texts, val_labels = load_split(SPLITS_DIR / "val.csv", args.text_col)
    test_texts, test_labels = load_split(SPLITS_DIR / "test.csv", args.text_col)

    train_texts, train_labels = maybe_subsample(
        train_texts, train_labels,
        args.max_train_samples if args.max_train_samples > 0 else None,
        args.seed,
    )
    val_texts, val_labels = maybe_subsample(
        val_texts, val_labels,
        args.max_val_samples if args.max_val_samples > 0 else None,
        args.seed + 1,
    )
    test_texts, test_labels = maybe_subsample(
        test_texts, test_labels,
        args.max_test_samples if args.max_test_samples > 0 else None,
        args.seed + 2,
    )
    log.info("Data sizes — train:%d val:%d test:%d", len(train_labels), len(val_labels), len(test_labels))

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    except Exception as exc:
        log.warning("Fast tokenizer load failed (%s). Falling back to slow tokenizer.", exc)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    train_ds = to_hf_dataset(tokenizer, train_texts, train_labels, args.max_length)
    val_ds = to_hf_dataset(tokenizer, val_texts, val_labels, args.max_length)
    test_ds = to_hf_dataset(tokenizer, test_texts, test_labels, args.max_length)

    output_dir = MODELS_DIR / f"{args.run_name}_hf"
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=0.01,
        logging_steps=100,
        seed=args.seed,
        report_to=[],
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics_from_logits,
    )

    trainer.train()

    val_pred = trainer.predict(val_ds)
    val_metrics = compute_metrics_from_logits((val_pred.predictions, val_pred.label_ids))

    test_pred = trainer.predict(test_ds)
    y_test_pred = np.argmax(test_pred.predictions, axis=1)
    test_metrics = {
        "accuracy": float(accuracy_score(test_labels, y_test_pred)),
        "f1_macro": float(f1_score(test_labels, y_test_pred, average="macro")),
        "f1_foxnews": float(f1_score(test_labels, y_test_pred, pos_label=0, average="binary")),
        "f1_nbc": float(f1_score(test_labels, y_test_pred, pos_label=1, average="binary")),
        "precision_macro": float(precision_score(test_labels, y_test_pred, average="macro")),
        "recall_macro": float(recall_score(test_labels, y_test_pred, average="macro")),
    }

    metrics_out = {
        "run_name": args.run_name,
        "model_name": args.model_name,
        "text_col": args.text_col,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "max_train_samples": args.max_train_samples,
        "max_val_samples": args.max_val_samples,
        "max_test_samples": args.max_test_samples,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = REPORTS_DIR / f"metrics_{args.run_name}.json"
    metrics_path.write_text(json.dumps(metrics_out, indent=2), encoding="utf-8")
    log.info("Saved metrics: %s", metrics_path)

    save_confusion_matrix(
        y_true=test_labels,
        y_pred=y_test_pred,
        out_path=FIGURES_DIR / f"confusion_matrix_{args.run_name}.png",
        title=f"{args.run_name} — Confusion Matrix",
    )
    log.info("Saved confusion matrix figure for %s", args.run_name)

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    log.info("Saved model/tokenizer: %s", output_dir)


if __name__ == "__main__":
    # Avoid tokenizer thread contention warnings on Windows.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
