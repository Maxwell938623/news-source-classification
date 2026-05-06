from __future__ import annotations
import io
import sys
import types
from pathlib import Path
from typing import Any, Iterable, List
import joblib
import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from sklearn.base import BaseEstimator, TransformerMixin
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PIPELINE = PROJECT_ROOT / "best_model.joblib"
FALLBACK_PIPELINE = PROJECT_ROOT / "models" / "best_model.joblib"
BLOB_CAP = 2000000
ID_TO_LABEL = {0: "FoxNews", 1: "NBC"}
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
def _register_pickle_compat() -> None:
    pkg = types.ModuleType("models")
    sys.modules.setdefault("models", pkg)
    stylometric_mod = types.ModuleType("models.stylometric")
    stylometric_mod.StyloTransformer = StyloTransformer
    sys.modules.setdefault("models.stylometric", stylometric_mod)
    hybrid_mod = types.ModuleType("models.hybrid")
    hybrid_mod.SparseWrapper = SparseWrapper
    sys.modules.setdefault("models.hybrid", hybrid_mod)
class NewsClassifier(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.register_buffer("pipeline_blob", torch.zeros(BLOB_CAP, dtype=torch.uint8))
        self.register_buffer("pipeline_size", torch.tensor(0, dtype=torch.int64))
        self.pipeline = None
    def _load_pipeline_from_blob(self):
        n = int(self.pipeline_size.item())
        if n <= 0:
            return None
        raw = self.pipeline_blob[:n].cpu().numpy().tobytes()
        return joblib.load(io.BytesIO(raw))
    def _load_pipeline_from_file(self):
        _register_pickle_compat()
        if DEFAULT_PIPELINE.exists():
            return joblib.load(DEFAULT_PIPELINE)
        if FALLBACK_PIPELINE.exists():
            return joblib.load(FALLBACK_PIPELINE)
        raise FileNotFoundError(
            "Could not find best_model.joblib. Place it next to model.py "
            "or at models/best_model.joblib."
        )
    def _ensure_pipeline(self):
        if self.pipeline is not None:
            return self.pipeline
        _register_pickle_compat()
        p = self._load_pipeline_from_blob()
        if p is not None:
            self.pipeline = p
            return self.pipeline
        self.pipeline = self._load_pipeline_from_file()
        return self.pipeline
    def eval(self) -> "NewsClassifier":
        return self
    def predict(self, batch: Iterable[Any]) -> List[str]:
        texts = [str(x) for x in batch]
        y_pred = self._ensure_pipeline().predict(texts)
        out: List[str] = []
        for p in y_pred:
            try:
                out.append(ID_TO_LABEL.get(int(p), str(p)))
            except Exception:
                out.append(str(p))
        return out
class Model(NewsClassifier):
    pass
def get_model() -> NewsClassifier:
    return NewsClassifier()
