"""
model.py - News Headline Classifier (ModernBERT-base, fine-tuned).

Submission contract (CIS 4190/5190 Spring 2026 Final Project, Project B):
    * exposes a class `NewsClassifier` (and an alias `Model`) instantiable
      without arguments,
    * exposes `get_model()` returning a fresh instance,
    * exposes `.predict(batch)` that takes an iterable of headline strings
      and returns a list of integer class ids (FoxNews=0, NBC=1).

The fine-tuned weights live in `model.pt` (state_dict) sitting next to this
file. The architecture and tokenizer are reconstructed from local files in
`./assets/` so the model works without any internet access at evaluation
time.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoTokenizer,
    ModernBertForSequenceClassification,
)

_HERE = Path(__file__).resolve().parent
_ASSETS_DIR = _HERE / "assets"
_WEIGHTS_PATH = _HERE / "model.pt"

# Canonical upstream identifier. The base config and tokenizer are unchanged
# from upstream (we only fine-tuned the encoder body + classifier head), so
# loading these from the Hub gives an exactly-compatible architecture into
# which `model.pt` deserialises cleanly.
_HF_HUB_ID = "answerdotai/ModernBERT-base"

LABEL_NAMES = {0: "FoxNews", 1: "NBC"}
_MAX_LEN = 64

_SENTINEL_NO_WEIGHTS = "__no_weights__.pth"


def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_config_source() -> str:
    """Prefer a bundled local assets/ dir; otherwise fall back to HF Hub."""
    if _ASSETS_DIR.is_dir() and (_ASSETS_DIR / "config.json").is_file():
        return str(_ASSETS_DIR)
    return _HF_HUB_ID


class NewsClassifier(nn.Module):
    """ModernBERT-base sequence classifier wrapped to satisfy the backend contract.

    The leaderboard backend probes the constructor in two ways:
        1. cls(weights_path="__no_weights__.pth")  - cold init, weights loaded later
        2. cls()                                    - cold init, weights loaded later
    Both paths must succeed; the `weights_path` kwarg is accepted but never
    consumed at construction time (the backend later calls load_state_dict
    with the actual weights file).
    """

    def __init__(self, weights_path: Optional[str] = None) -> None:
        super().__init__()
        source = _resolve_config_source()

        config = AutoConfig.from_pretrained(
            source,
            num_labels=2,
            id2label=LABEL_NAMES,
            label2id={v: k for k, v in LABEL_NAMES.items()},
        )
        self.model = ModernBertForSequenceClassification(config)
        self.tokenizer = AutoTokenizer.from_pretrained(source)
        self.max_len = _MAX_LEN
        self._device = _resolve_device()

        # If the backend doesn't run its own load_state_dict pass (or we are
        # running locally), auto-load weights from a sensible default path.
        candidate_paths = []
        if weights_path and weights_path != _SENTINEL_NO_WEIGHTS:
            candidate_paths.append(Path(weights_path))
        candidate_paths.append(_WEIGHTS_PATH)

        for p in candidate_paths:
            if p.exists():
                try:
                    state = torch.load(str(p), map_location="cpu")
                    if isinstance(state, dict) and "state_dict" in state:
                        state = state["state_dict"]
                    self.load_state_dict(state, strict=False)
                    break
                except Exception:
                    # Backend will overwrite via its own load_state_dict path; ignore.
                    pass

        self.to(self._device)
        self.eval()

    # ------------------------------------------------------------------
    # Inference contract
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, batch: Union[Sequence[str], Iterable[str]]) -> List[int]:
        """Take an iterable of headline strings, return list of int class ids."""
        if isinstance(batch, str):
            batch = [batch]
        texts = [str(x) if x is not None else "" for x in batch]
        if not texts:
            return []
        out: List[int] = []
        bs = 32
        for i in range(0, len(texts), bs):
            chunk = texts[i : i + bs]
            enc = self.tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt",
            )
            enc = {k: v.to(self._device) for k, v in enc.items()}
            logits = self.model(**enc).logits
            preds = torch.argmax(logits, dim=-1).cpu().tolist()
            out.extend(int(p) for p in preds)
        return out

    @torch.no_grad()
    def forward(self, batch):
        """Backend fallback path: returns a logits tensor.

        Backend will argmax over the final dimension if a tensor is returned.
        """
        if isinstance(batch, (list, tuple)) and (len(batch) == 0 or isinstance(batch[0], str)):
            enc = self.tokenizer(
                list(batch),
                padding=True,
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt",
            )
            enc = {k: v.to(self._device) for k, v in enc.items()}
            return self.model(**enc).logits
        # Already a tokenized dict / tensor: hand straight through.
        if isinstance(batch, dict):
            batch = {k: v.to(self._device) for k, v in batch.items()}
            return self.model(**batch).logits
        return self.model(batch).logits


# Backend looks for either of these names; expose both for compatibility.
Model = NewsClassifier


def get_model() -> NewsClassifier:
    return NewsClassifier()
