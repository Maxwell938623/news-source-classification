from __future__ import annotations

import io
import pickle
import sys as _sys
from pathlib import Path
from typing import Any, Iterable, List

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from torch import nn

# Stable alias so pickle always resolves embeddings under "model.*", regardless
# of the dynamic module name Hugging Face uses when importing this file.
_mod = _sys.modules.setdefault("model", _sys.modules.get(__name__))

MODEL_PT_PATH = Path(__file__).resolve().with_name("model.pt")
BLOB_CAP = 20_000_000


def _dense_array(X):
    return X.toarray() if sp.issparse(X) else np.asarray(X, dtype=np.float64)


def _extract_stylo_features(text: str) -> list:
    words = text.split()
    n_words = len(words)
    n_chars = len(text)
    n_alpha = sum(1 for c in text if c.isalpha()) or 1
    avg_wl = float(np.mean([len(w) for w in words])) if words else 0.0
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
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([_extract_stylo_features(t) for t in X], dtype=np.float64)


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


StyloTransformer.__module__ = "model"
StyloTransformer.__qualname__ = "StyloTransformer"
SparseWrapper.__module__ = "model"
SparseWrapper.__qualname__ = "SparseWrapper"

_mod._dense_array = _dense_array
_dense_array.__module__ = "model"
_dense_array.__qualname__ = "_dense_array"


def _loads_pipeline(blob: bytes) -> Any:
    """Unpickle sklearn pipeline; redirect stale/global names to this submission."""
    sm = _sys.modules.get("model")
    if sm is None:
        return pickle.loads(blob)

    class _U(pickle.Unpickler):
        def find_class(self, module: str, name: str):
            if name == "_dense_array":
                return sm._dense_array
            if name == "SparseWrapper":
                return sm.SparseWrapper
            if name == "StyloTransformer":
                return sm.StyloTransformer
            return super().find_class(module, name)

    return _U(io.BytesIO(blob), fix_imports=True).load()


class Model(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.register_buffer("pipeline_blob", torch.zeros(BLOB_CAP, dtype=torch.uint8))
        self.register_buffer("pipeline_size", torch.tensor(0, dtype=torch.int64))
        self.pipeline = None

    def _extract_blob_bytes(self, payload: Any) -> bytes:
        state = payload.get("state_dict") if isinstance(payload, dict) and "state_dict" in payload else payload
        blob = state.get("pipeline_blob") if isinstance(state, dict) else None
        size = state.get("pipeline_size") if isinstance(state, dict) else None
        if not torch.is_tensor(blob):
            raise RuntimeError("model.pt missing tensor key 'pipeline_blob'.")
        n = int(size.item()) if torch.is_tensor(size) else int(blob.numel())
        if n <= 0:
            raise RuntimeError("model.pt has empty pipeline blob.")
        return blob[:n].cpu().numpy().tobytes()

    def _ensure_pipeline(self):
        if self.pipeline is not None:
            return self.pipeline
        # Always read pickle bytes from disk. The evaluator may call
        # load_state_dict() with keys that omit or rename `pipeline_blob`, leaving
        # registered buffers zero-filled while `pipeline_size` still matches — then
        # unpickling buffered bytes raises UnpicklingError (e.g. invalid load key '\\x00').
        if not MODEL_PT_PATH.exists():
            raise RuntimeError(f"Missing required artifact: {MODEL_PT_PATH.name}")
        payload = torch.load(MODEL_PT_PATH, map_location="cpu", weights_only=False)
        raw = self._extract_blob_bytes(payload)
        self.pipeline = _loads_pipeline(raw)
        return self.pipeline

    def eval(self) -> "Model":
        super().eval()
        return self

    def predict(self, batch: Iterable[Any]) -> List[int]:
        texts = [str(x) for x in batch]
        preds = self._ensure_pipeline().predict(texts)
        return [int(p) for p in preds]


def get_model() -> Model:
    return Model()
