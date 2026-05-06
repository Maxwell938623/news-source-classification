#!/usr/bin/env python3
"""
build_model_pt.py - Convert the fine-tuned ModernBERT-base HuggingFace artifact
into a state_dict at submission/modernbert/model.pt that the leaderboard
backend can load via torch.load(..., map_location="cpu") + load_state_dict.

Run after training:
    python submission/build_model_pt.py
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoConfig, ModernBertForSequenceClassification

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "models" / "modernbert_base" / "final"
DST_PT = ROOT / "submission" / "modernbert" / "model.pt"
ASSETS_DIR = ROOT / "submission" / "modernbert" / "assets"


def main() -> None:
    if not SRC_DIR.exists():
        raise FileNotFoundError(
            f"Trained model not found at {SRC_DIR}. Run src/train_modernbert.py first."
        )

    print(f"[load] {SRC_DIR}")
    config = AutoConfig.from_pretrained(
        str(SRC_DIR),
        num_labels=2,
        id2label={0: "FoxNews", 1: "NBC"},
        label2id={"FoxNews": 0, "NBC": 1},
    )

    # Architecture must exactly match what model.py instantiates.
    model = ModernBertForSequenceClassification.from_pretrained(str(SRC_DIR), config=config)

    # Build the same wrapper key prefix that model.py's NewsClassifier exposes:
    # NewsClassifier wraps ModernBertForSequenceClassification under attribute `model`,
    # so saved state_dict keys must be prefixed with "model.".
    nested_state = {f"model.{k}": v for k, v in model.state_dict().items()}

    DST_PT.parent.mkdir(parents=True, exist_ok=True)
    print(f"[save] {DST_PT}  ({len(nested_state):,} tensors)")
    torch.save(nested_state, str(DST_PT))

    size_mb = DST_PT.stat().st_size / (1024 ** 2)
    print(f"[done] wrote {DST_PT}  ({size_mb:.1f} MB)")

    # Sanity: print the first few keys to confirm the prefix.
    sample_keys = list(nested_state.keys())[:5]
    print(f"[keys] sample: {json.dumps(sample_keys, indent=2)}")

    # Also verify required assets present.
    required = ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]
    missing = [f for f in required if not (ASSETS_DIR / f).exists()]
    if missing:
        print(f"[warn] missing assets in {ASSETS_DIR}: {missing}")
    else:
        print(f"[assets] {ASSETS_DIR} contains all required tokenizer + config files")


if __name__ == "__main__":
    main()
