"""
Embedded artifact: StackingClassifier matching ``models/best_model_metadata.json``
(``Stack_4base_style_HGB_meta``) — four sparse/tabular bases (word TF-IDF→LR, char
TF-IDF→LR, word TF-IDF→MNB, stylometric→StandardScaler→LR) with a
HistGradientBoostingClassifier meta-learner. ``StyloTransformer`` / ``SparseWrapper``
stay in this file so pickled refs resolve as ``model.*`` under any import name.
"""

from __future__ import annotations

import pickle
import sys as _sys
from pathlib import Path
from typing import Any, Iterable, List

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from torch import nn

# Ensure pickle can always find 'model.StyloTransformer' / 'model.SparseWrapper'
# regardless of the module name the evaluator used when importing this file.
_sys.modules.setdefault("model", _sys.modules.get(__name__))

MODEL_PT_PATH = Path(__file__).resolve().with_name("model.pt")
BLOB_CAP = 2_000_000


# ---------------------------------------------------------------------------
# Stylometric and sparse-wrapper helpers (embedded so pickle refs resolve)
# ---------------------------------------------------------------------------

def _extract_stylo_features(text: str) -> list:
    words   = text.split()
    n_words = len(words)
    n_chars = len(text)
    n_alpha = sum(1 for c in text if c.isalpha()) or 1
    avg_wl  = float(np.mean([len(w) for w in words])) if words else 0.0
    return [
        float(n_words),
        float(n_chars),
        avg_wl,
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
    """15-dim stylometric feature extractor (stateless)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([_extract_stylo_features(t) for t in X], dtype=np.float64)

    def get_feature_names_out(self, input_features=None):
        return np.array([
            "word_count", "char_count", "avg_word_len",
            "exclamation_count", "question_count", "ellipsis_count",
            "colon_present", "quote_present", "dash_present", "paren_present",
            "cap_ratio", "allcaps_word_count", "digit_ratio",
            "title_case_ratio", "comma_count",
        ])


class SparseWrapper(BaseEstimator, TransformerMixin):
    """Wraps a dense-output transformer so FeatureUnion can hstack it."""

    def __init__(self, transformer):
        self.transformer = transformer

    def fit(self, X, y=None):
        self.transformer.fit(X, y)
        return self

    def transform(self, X):
        dense = self.transformer.transform(X)
        return sp.csr_matrix(dense)


class Model(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.register_buffer("pipeline_blob", torch.zeros(BLOB_CAP, dtype=torch.uint8))
        self.register_buffer("pipeline_size", torch.tensor(0, dtype=torch.int64))
        self.pipeline = None

    def _extract_blob_bytes(self, payload: Any) -> bytes:
        state = payload.get("state_dict") if isinstance(payload, dict) and "state_dict" in payload else payload
        if not isinstance(state, dict):
            raise RuntimeError("model.pt payload must be a dict or {'state_dict': dict}.")

        blob = state.get("pipeline_blob")
        size = state.get("pipeline_size")
        if not torch.is_tensor(blob):
            raise RuntimeError("model.pt missing tensor key 'pipeline_blob'.")

        if torch.is_tensor(size):
            n = int(size.item())
        else:
            n = int(blob.numel())
        if n <= 0:
            raise RuntimeError("model.pt has empty pipeline blob.")
        return blob[:n].cpu().numpy().tobytes()

    def _ensure_pipeline(self):
        if self.pipeline is not None:
            return self.pipeline
        # Read embedded sklearn pickle from checkpoint file, not registered buffers:
        # a prior load_state_dict() can skip `pipeline_blob` (key mismatch) leaving
        # zeros while lengths look valid, which breaks pickle.loads (invalid key '\\x00').
        if not MODEL_PT_PATH.exists():
            raise RuntimeError(f"Missing required artifact: {MODEL_PT_PATH.name}")
        payload = torch.load(MODEL_PT_PATH, map_location="cpu", weights_only=False)
        raw = self._extract_blob_bytes(payload)
        self.pipeline = pickle.loads(raw)
        return self.pipeline

    def eval(self) -> "Model":
        super().eval()
        return self

    def predict(self, batch: Iterable[Any]) -> List[int]:
        texts = [str(x) for x in batch]
        y_pred = self._ensure_pipeline().predict(texts)
        out: List[int] = []
        for p in y_pred:
            try:
                out.append(int(p))
            except Exception:
                # Fail-safe default in case estimator emits non-numeric labels.
                out.append(0)
        return out


def get_model() -> Model:
    return Model()
