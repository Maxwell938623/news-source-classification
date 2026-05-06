"""
sentence_embedding.py — SentenceTransformer embeddings + classifier experiments.

Pipeline:
    headline text -> sentence embedding -> classifier

Embedding models:
    - all-MiniLM-L6-v2
    - all-mpnet-base-v2
    - BAAI/bge-small-en-v1.5
    - BAAI/bge-base-en-v1.5

Classifiers:
    - LogisticRegression
    - LinearSVC
    - XGBoost (optional)
    - LightGBM (optional)
    - MLPClassifier

Standalone:
    python src/models/sentence_embedding.py
"""
from __future__ import annotations

import hashlib
import logging
import sys
from pathlib import Path
from typing import Any

# Preload torch before NumPy/SciPy stack to avoid Windows DLL init issues
# in environments with mixed native deps.
try:
    import torch as _torch_preload  # noqa: F401
except Exception:
    _torch_preload = None

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

try:
    from ._base import ModelConfig, standalone_train
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _base import ModelConfig, standalone_train


try:
    from sentence_transformers import SentenceTransformer
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except Exception:  # noqa: BLE001 — also catches RuntimeError from torchvision/DLL issues
    SentenceTransformer = None
    _SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from xgboost import XGBClassifier
    _XGBOOST_AVAILABLE = True
except ImportError:
    XGBClassifier = None
    _XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    _LIGHTGBM_AVAILABLE = True
except ImportError:
    LGBMClassifier = None
    _LIGHTGBM_AVAILABLE = False


class SentenceEmbeddingTransformer(BaseEstimator, TransformerMixin):
    """Sklearn-compatible wrapper over sentence-transformers models."""

    _MODEL_REGISTRY: dict[str, Any] = {}
    _EMBEDDING_CACHE: dict[str, np.ndarray] = {}

    def __init__(
        self,
        model_name: str,
        batch_size: int = 128,
        normalize_embeddings: bool = True,
        max_seq_length: int | None = None,
        cache_dir: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.max_seq_length = max_seq_length
        self.cache_dir = cache_dir

    def _model_cache_key(self) -> str:
        return f"{self.model_name}::maxlen={self.max_seq_length}"

    def _get_cache_dir(self) -> Path | None:
        if self.cache_dir is None:
            return None
        p = Path(self.cache_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _embedding_cache_key(self, texts: list[str]) -> str:
        h = hashlib.sha1()
        h.update(self.model_name.encode("utf-8"))
        h.update(f"|norm={int(self.normalize_embeddings)}".encode("utf-8"))
        h.update(f"|maxlen={self.max_seq_length}".encode("utf-8"))
        for t in texts:
            h.update(b"\x1f")
            h.update(t.encode("utf-8", errors="ignore"))
        return h.hexdigest()

    def fit(self, X: list[str], y: Any = None) -> "SentenceEmbeddingTransformer":
        if not _SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required for sentence embedding models. "
                "Install it via `pip install sentence-transformers`."
            )

        key = self._model_cache_key()
        if key in self._MODEL_REGISTRY:
            self._model = self._MODEL_REGISTRY[key]
            return self

        model = SentenceTransformer(self.model_name)
        if self.max_seq_length is not None:
            model.max_seq_length = int(self.max_seq_length)
        self._MODEL_REGISTRY[key] = model
        self._model = model
        return self

    def transform(self, X: list[str]) -> np.ndarray:
        if not hasattr(self, "_model"):
            raise RuntimeError("SentenceEmbeddingTransformer must be fit() before transform().")
        texts = ["" if t is None else str(t) for t in X]
        key = self._embedding_cache_key(texts)

        if key in self._EMBEDDING_CACHE:
            return self._EMBEDDING_CACHE[key]

        cache_dir = self._get_cache_dir()
        if cache_dir is not None:
            cache_path = cache_dir / f"{key}.npy"
            if cache_path.exists():
                emb = np.load(cache_path)
                self._EMBEDDING_CACHE[key] = emb
                return emb

        emb = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        emb = emb.astype(np.float32, copy=False)
        self._EMBEDDING_CACHE[key] = emb

        if cache_dir is not None:
            np.save(cache_dir / f"{key}.npy", emb)
        return emb


def _lr() -> LogisticRegression:
    return LogisticRegression(
        C=1.0,
        max_iter=2500,
        solver="lbfgs",
        random_state=42,
    )


def _svm() -> LinearSVC:
    return LinearSVC(
        C=1.0,
        max_iter=5000,
        dual="auto",
        random_state=42,
    )


def _mlp() -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=(128,),
        activation="relu",
        alpha=1e-4,
        max_iter=150,
        early_stopping=True,
        n_iter_no_change=8,
        random_state=42,
    )


def _xgb() -> Any:
    if not _XGBOOST_AVAILABLE:
        return None
    return XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )


def _lgbm() -> Any:
    if not _LIGHTGBM_AVAILABLE:
        return None
    return LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )


def _pipeline(embedding_model: str, clf: Any) -> Pipeline:
    project_root = Path(__file__).resolve().parents[2]
    cache_dir = project_root / "models" / "embedding_cache"
    return Pipeline([
        (
            "embed",
            SentenceEmbeddingTransformer(
                model_name=embedding_model,
                cache_dir=str(cache_dir),
            ),
        ),
        ("clf", clf),
    ])


def get_configs() -> list[ModelConfig]:
    log = logging.getLogger(__name__)
    if not _SENTENCE_TRANSFORMERS_AVAILABLE:
        log.warning(
            "sentence-transformers not installed; returning no sentence embedding configs."
        )
        return []

    embedding_models = [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2",
        "BAAI/bge-small-en-v1.5",
        "BAAI/bge-base-en-v1.5",
    ]

    classifier_factories: list[tuple[str, Any, str]] = [
        ("logreg", _lr, "LogisticRegression over sentence embeddings"),
        ("svm", _svm, "LinearSVC over sentence embeddings"),
        ("mlp", _mlp, "MLPClassifier over sentence embeddings"),
    ]
    if _XGBOOST_AVAILABLE:
        classifier_factories.append(
            ("xgboost", _xgb, "XGBoost over sentence embeddings")
        )
    if _LIGHTGBM_AVAILABLE:
        classifier_factories.append(
            ("lightgbm", _lgbm, "LightGBM over sentence embeddings")
        )

    if not _XGBOOST_AVAILABLE:
        log.warning("xgboost not installed; skipping XGBoost sentence-embedding configs.")
    if not _LIGHTGBM_AVAILABLE:
        log.warning("lightgbm not installed; skipping LightGBM sentence-embedding configs.")

    configs: list[ModelConfig] = []
    for model_name in embedding_models:
        model_tag = model_name.split("/")[-1].replace(".", "_")
        for clf_tag, clf_factory, desc in classifier_factories:
            clf = clf_factory()
            if clf is None:
                continue
            configs.append(
                ModelConfig(
                    name=f"SE_{model_tag}_{clf_tag}",
                    group="sentence_embedding",
                    prep_col="headline_minimal",
                    estimator=_pipeline(model_name, clf),
                    description=f"{desc}; embed={model_name}",
                )
            )
    return configs


if __name__ == "__main__":
    standalone_train(
        group_name="sentence_embedding",
        get_configs_fn=get_configs,
        log_filename="train_sentence_embedding.log",
    )
