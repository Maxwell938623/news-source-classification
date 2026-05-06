#!/usr/bin/env python3
"""
train_modernbert.py - Fine-tune ModernBERT-base on the Fox/NBC headlines task.

Selects best checkpoint by validation macro-F1, then evaluates once on the
held-out test split. Mirrors the artifact layout used by train_best_model.py
so its outputs compose with the rest of the repo's reporting tooling.

Outputs:
    models/modernbert_base/                          fine-tuned weights + tokenizer
    models/modernbert_base_metadata.json             config + final metrics
    reports/metrics_modernbert_base.json             test-set metrics
    reports/figures/modernbert_confusion_matrix.png  test confusion matrix
    logs/train_modernbert.log

Usage:
    python src/train_modernbert.py
    python src/train_modernbert.py --smoke           # 1k/200/200 quick check
    python src/train_modernbert.py --epochs 4 --batch-size 48
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict

# MPS sometimes hits OpenMP / fork issues with HF dataloaders; keep things sane.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ROOT / "data" / "processed" / "splits"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
LOGS_DIR = ROOT / "logs"

LABEL_NAMES = {0: "FoxNews", 1: "NBC"}


def setup_logging(log_path: Path) -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True, parents=True)
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    handlers = [logging.FileHandler(log_path, mode="w"), logging.StreamHandler(sys.stdout)]
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers, force=True)
    return logging.getLogger("modernbert")


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_split(name: str, text_col: str, smoke: int = 0) -> Dataset:
    path = SPLITS / f"{name}.csv"
    df = pd.read_csv(path, usecols=[text_col, "label"])
    df = df.dropna(subset=[text_col, "label"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    df = df.rename(columns={text_col: "text"})
    if smoke:
        per_class = max(1, smoke // 2)
        df = (
            df.groupby("label", group_keys=False)
            .apply(lambda g: g.sample(min(len(g), per_class), random_state=42))
            .reset_index(drop=True)
        )
    return Dataset.from_pandas(df, preserve_index=False)


def build_compute_metrics():
    def _fn(eval_pred):
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": float(accuracy_score(labels, preds)),
            "f1_macro": float(f1_score(labels, preds, average="macro")),
            "f1_weighted": float(f1_score(labels, preds, average="weighted")),
            "precision_macro": float(precision_score(labels, preds, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(labels, preds, average="macro", zero_division=0)),
        }

    return _fn


def evaluate_full(trainer: Trainer, dataset: Dataset, split_name: str, logger: logging.Logger) -> Dict:
    pred = trainer.predict(dataset)
    logits = pred.predictions if not isinstance(pred.predictions, tuple) else pred.predictions[0]
    preds = np.argmax(logits, axis=-1)
    labels = pred.label_ids
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    metrics = {
        "split": split_name,
        "size": int(len(labels)),
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro")),
        "f1_weighted": float(f1_score(labels, preds, average="weighted")),
        "f1_foxnews": float(f1_score(labels, preds, average=None, labels=[0, 1])[0]),
        "f1_nbc": float(f1_score(labels, preds, average=None, labels=[0, 1])[1]),
        "precision_macro": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(labels, preds, average="macro", zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            labels,
            preds,
            target_names=[LABEL_NAMES[0], LABEL_NAMES[1]],
            output_dict=True,
            zero_division=0,
        ),
    }
    logger.info(
        "[%s] acc=%.4f  macro-F1=%.4f  Fox-F1=%.4f  NBC-F1=%.4f",
        split_name,
        metrics["accuracy"],
        metrics["f1_macro"],
        metrics["f1_foxnews"],
        metrics["f1_nbc"],
    )
    return metrics


def save_confusion_matrix(cm, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    disp = ConfusionMatrixDisplay(confusion_matrix=np.array(cm), display_labels=[LABEL_NAMES[0], LABEL_NAMES[1]])
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="answerdotai/ModernBERT-base")
    parser.add_argument("--text-col", default="headline_minimal", help="Text column from split CSVs.")
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-stop-patience", type=int, default=2)
    parser.add_argument("--output-dir", default=str(MODELS_DIR / "modernbert_base"))
    parser.add_argument("--logging-steps", type=int, default=200)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--smoke", action="store_true", help="Tiny subset run to validate the pipeline end-to-end.")
    parser.add_argument("--device", default=None, choices=[None, "cpu", "mps", "cuda"])
    args = parser.parse_args()

    set_seed(args.seed)
    LOGS_DIR.mkdir(exist_ok=True, parents=True)
    REPORTS_DIR.mkdir(exist_ok=True, parents=True)
    FIGURES_DIR.mkdir(exist_ok=True, parents=True)
    MODELS_DIR.mkdir(exist_ok=True, parents=True)
    log_name = "train_modernbert_smoke.log" if args.smoke else "train_modernbert.log"
    logger = setup_logging(LOGS_DIR / log_name)

    device = args.device or pick_device()
    logger.info("Device: %s", device)
    logger.info("Model: %s", args.model_name)
    logger.info("Args: %s", json.dumps(vars(args), default=str))

    smoke_n = 1000 if args.smoke else 0
    train_ds = load_split("train", args.text_col, smoke=smoke_n)
    val_ds = load_split("val", args.text_col, smoke=200 if args.smoke else 0)
    test_ds = load_split("test", args.text_col, smoke=200 if args.smoke else 0)
    logger.info("Sizes  train=%d  val=%d  test=%d", len(train_ds), len(val_ds), len(test_ds))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_len)

    train_ds = train_ds.map(tok, batched=True, remove_columns=["text"])
    val_ds = val_ds.map(tok, batched=True, remove_columns=["text"])
    test_ds = test_ds.map(tok, batched=True, remove_columns=["text"])

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label=LABEL_NAMES,
        label2id={v: k for k, v in LABEL_NAMES.items()},
    )

    # MPS does NOT support bf16 amp well across all kernels in torch 2.8; use fp32
    # (ModernBERT-base is 149M; fp32 fits comfortably in 24GB unified memory).
    use_bf16 = device == "cuda"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        seed=args.seed,
        report_to=[],
        bf16=use_bf16,
        fp16=False,
        dataloader_num_workers=0,
        remove_unused_columns=True,
        disable_tqdm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=build_compute_metrics(),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stop_patience)],
    )

    logger.info("Starting training ...")
    train_result = trainer.train()
    logger.info("Training done in %.1fs", train_result.metrics.get("train_runtime", float("nan")))

    val_metrics = evaluate_full(trainer, val_ds, "val", logger)
    test_metrics = evaluate_full(trainer, test_ds, "test", logger)

    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    logger.info("Saved final model to %s", final_dir)

    metadata = {
        "model": "modernbert_base",
        "hf_model_id": args.model_name,
        "text_column": args.text_col,
        "max_len": args.max_len,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "device": device,
        "bf16": bool(use_bf16),
        "seed": args.seed,
        "label_encoding": {v: k for k, v in LABEL_NAMES.items()},
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
        "train_runtime_s": float(train_result.metrics.get("train_runtime", 0.0)),
        "best_metric": float(trainer.state.best_metric) if trainer.state.best_metric is not None else None,
        "val_metrics": val_metrics,
    }
    metadata_path = MODELS_DIR / "modernbert_base_metadata.json"
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Wrote %s", metadata_path)

    # Mirror the metrics_*.json layout used by train_baseline / train_best_model.
    metrics_payload = {
        "model": "modernbert_base",
        "hf_model_id": args.model_name,
        "text_column": args.text_col,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
        "device": device,
        "metrics": test_metrics,
    }
    metrics_path = REPORTS_DIR / "metrics_modernbert_base.json"
    with metrics_path.open("w") as f:
        json.dump(metrics_payload, f, indent=2)
    logger.info("Wrote %s", metrics_path)

    cm_path = FIGURES_DIR / "modernbert_confusion_matrix.png"
    save_confusion_matrix(test_metrics["confusion_matrix"], cm_path, "ModernBERT-base — test confusion matrix")
    logger.info("Wrote %s", cm_path)

    print("\n=== Final test metrics (ModernBERT-base) ===")
    print(f"  accuracy   = {test_metrics['accuracy']:.4f}")
    print(f"  macro-F1   = {test_metrics['f1_macro']:.4f}")
    print(f"  Fox  F1    = {test_metrics['f1_foxnews']:.4f}")
    print(f"  NBC  F1    = {test_metrics['f1_nbc']:.4f}")


if __name__ == "__main__":
    main()
