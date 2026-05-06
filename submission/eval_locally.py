#!/usr/bin/env python3
"""
eval_locally.py - Mimic the leaderboard backend's evaluation contract for
the ModernBERT submission so we can sanity-check it before uploading.

Backend behaviour we reproduce here (per submission.txt):
    1. Read CSV
    2. preprocess.prepare_data(csv) returns (X, y)
    3. Instantiate model (no-arg constructor)
    4. Load model.pt as state_dict via torch.load(map_location="cpu") +
       load_state_dict (robust)
    5. Run inference via model.predict(batch) if available else model(batch)
       and argmax over the final dim
    6. Compute accuracy by comparing to y
    7. Print a short JSON report

Usage:
    python submission/eval_locally.py
    python submission/eval_locally.py --csv url_only_data.csv --limit 50
    python submission/eval_locally.py --csv data/processed/splits/test.csv --limit 200 --no-scrape
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
SUB_DIR = Path(__file__).resolve().parent / "modernbert"


def _import(module_name: str, file_path: Path):
    """Mirror how the backend likely imports submitted files."""
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _make_input_csv(args, tmp_dir: Path) -> Path:
    src = Path(args.csv)
    if not src.exists():
        raise FileNotFoundError(f"CSV not found: {src}")
    df = pd.read_csv(src, dtype=str)
    if args.limit and len(df) > args.limit:
        df = df.head(args.limit)
    out = tmp_dir / "leaderboard_input.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if "url" not in df.columns:
        # Allow running on test.csv (which already has labels + scraped headlines)
        # by creating a URL-only view.
        url_col = next((c for c in ("url", "URL", "link") if c in df.columns), None)
        if url_col is None:
            raise ValueError(f"No URL column found in {src} (columns: {list(df.columns)})")
        df = df[[url_col]].rename(columns={url_col: "url"})
    else:
        df = df[["url"]]
    df.to_csv(out, index=False)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=str(ROOT / "url_only_data.csv"))
    p.add_argument("--limit", type=int, default=20, help="Number of rows to evaluate (keep small to avoid hammering live URLs).")
    p.add_argument("--no-scrape", action="store_true",
                   help="Use already-scraped headlines from data/processed/splits/test.csv instead of live scraping.")
    args = p.parse_args()

    print(f"== loading submission modules from {SUB_DIR} ==")
    preprocess = _import("preprocess", SUB_DIR / "preprocess.py")
    model_mod = _import("model", SUB_DIR / "model.py")

    if hasattr(model_mod, "get_model"):
        model = model_mod.get_model()
    elif hasattr(model_mod, "Model"):
        model = model_mod.Model()
    elif hasattr(model_mod, "NewsClassifier"):
        model = model_mod.NewsClassifier()
    else:
        raise AttributeError("model.py exposes neither get_model() nor Model/NewsClassifier")

    # Robust state_dict load (mirror likely backend logic).
    weights_path = SUB_DIR / "model.pt"
    if weights_path.exists():
        print(f"== loading {weights_path.name} as state_dict ==")
        state = torch.load(str(weights_path), map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"  [warn] {len(missing)} missing keys (showing 3): {missing[:3]}")
        if unexpected:
            print(f"  [warn] {len(unexpected)} unexpected keys (showing 3): {unexpected[:3]}")
        if not missing and not unexpected:
            print("  [ok] all keys matched cleanly")

    if args.no_scrape:
        print("== --no-scrape mode: using pre-scraped headlines from test.csv ==")
        df = pd.read_csv(ROOT / "data" / "processed" / "splits" / "test.csv")
        if args.limit and len(df) > args.limit:
            df = df.sample(args.limit, random_state=42).reset_index(drop=True)
        X = df["headline_minimal"].fillna("").astype(str).tolist()
        y = df["label"].astype(int).tolist()
    else:
        tmp_dir = ROOT / ".eval_tmp"
        input_csv = _make_input_csv(args, tmp_dir)
        print(f"== prepare_data on {input_csv.name} ({args.limit or 'all'} rows) ==")
        t0 = time.time()
        X, y = preprocess.prepare_data(str(input_csv))
        print(f"  prepare_data took {time.time() - t0:.1f}s, returned len(X)={len(X)} len(y)={len(y)}")
        if X:
            print(f"  X[0] = {X[0]!r}")
            print(f"  y[0] = {y[0]}")

    if not X:
        print("No data; aborting.")
        return

    print(f"== running inference on {len(X)} examples ==")
    t0 = time.time()
    if hasattr(model, "predict"):
        preds = model.predict(X)
    else:
        out = model(X)
        if isinstance(out, torch.Tensor):
            preds = out.argmax(dim=-1).cpu().tolist()
        else:
            preds = list(out)
    print(f"  inference took {time.time() - t0:.1f}s")

    if len(preds) != len(y):
        print(f"  [error] length mismatch: preds={len(preds)} y={len(y)}")
        return

    correct = sum(int(p == int(t)) for p, t in zip(preds, y))
    acc = correct / len(y)
    print(json.dumps({
        "n": len(y),
        "accuracy": acc,
        "per_class_counts_y": {0: int(sum(1 for t in y if int(t) == 0)),
                                1: int(sum(1 for t in y if int(t) == 1))},
        "per_class_counts_pred": {0: int(sum(1 for p in preds if p == 0)),
                                   1: int(sum(1 for p in preds if p == 1))},
    }, indent=2))


if __name__ == "__main__":
    main()
