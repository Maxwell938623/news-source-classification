"""
gbdt.py — TF-IDF vocabulary features + stylometric features → Gradient Boosted Decision Trees.

Architecture:
    FeatureUnion([
        ("text",  TfidfVectorizer(...)),               → sparse (n, vocab)
        ("style", SparseWrapper(StyloTransformer()))   → sparse (n, 15)
    ])
    → horizontally stacked sparse matrix (n, vocab+15)
    → to_dense (FunctionTransformer)   ← GBDT requires dense input
    → HistGradientBoostingClassifier / XGBClassifier

All TF-IDF branches use lowercase=True; capitalisation is preserved in the
raw column (headline_minimal) so the stylometric branch can use it.

Groups:
  G7_gbdt  varied TF-IDF + GBDT hyperparameter configurations

Standalone:
    python src/gbdt.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MaxAbsScaler, FunctionTransformer
from xgboost import XGBClassifier

try:
    from models._base import ModelConfig, standalone_train
    from models.stylometric import StyloTransformer
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent / "models"))
    from _base import ModelConfig, standalone_train
    from stylometric import StyloTransformer


# ---------------------------------------------------------------------------
# Sparse wrapper (identical to hybrid.py)
# ---------------------------------------------------------------------------

class SparseWrapper(BaseEstimator, TransformerMixin):
    """Converts any transformer's dense output to csr_matrix for FeatureUnion."""

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


# ---------------------------------------------------------------------------
# to_dense step (required: GBDT classifiers do not accept sparse matrices)
# ---------------------------------------------------------------------------

def _dense_array(X):
    return X.toarray() if sp.issparse(X) else np.asarray(X, dtype=np.float64)


_to_dense = FunctionTransformer(_dense_array, accept_sparse=True)


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------

def _gbdt_pipeline(
    ngram: tuple[int, int] = (1, 2),
    word_mf: int = 5_000,
    char_ngram: tuple[int, int] | None = None,
    char_mf: int = 5_000,
    include_style: bool = True,
    clf: HistGradientBoostingClassifier | XGBClassifier | None = None,
) -> Pipeline:
    """
    Build a TF-IDF (+ optional char n-gram + optional stylometric) → GBDT pipeline.

    Parameters
    ----------
    ngram        : word n-gram range for the primary TF-IDF branch.
    word_mf      : max_features for the word TF-IDF branch.
    char_ngram   : if given, adds a char_wb TF-IDF branch with this range.
    char_mf      : max_features for the char TF-IDF branch.
    include_style: whether to include the StyloTransformer branch.
    clf          : a HistGradientBoostingClassifier instance; defaults to
                   iter=300, lr=0.05, max_depth=4.

    Pipeline steps
    --------------
    1. FeatureUnion  → sparse (n, vocab [+ char_vocab] [+ 15])
    2. MaxAbsScaler  → keeps sparsity, scales each column to [-1, 1]
    3. to_dense      → dense ndarray  (HistGBM / XGB / LGBM need this)
    4. clf           → HistGradientBoostingClassifier or XGBClassifier
    """
    if clf is None:
        clf = HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_depth=4,
            min_samples_leaf=20,
            l2_regularization=0.1,
            random_state=42,
        )

    branches: list[tuple[str, TransformerMixin]] = [
        ("word_tfidf", TfidfVectorizer(
            analyzer="word",
            ngram_range=ngram,
            max_features=word_mf,
            stop_words="english",
            lowercase=True,
            sublinear_tf=True,
        )),
    ]

    if char_ngram is not None:
        branches.append((
            "char_tfidf",
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=char_ngram,
                max_features=char_mf,
                lowercase=False,   # preserve case for character-level patterns
                sublinear_tf=True,
            ),
        ))

    if include_style:
        branches.append(("style", SparseWrapper(StyloTransformer())))

    return Pipeline([
        ("features", FeatureUnion(branches)),
        ("scaler",   MaxAbsScaler()),
        ("to_dense", _to_dense),
        ("clf",      clf),
    ])


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

def get_configs() -> list[ModelConfig]:
    configs: list[ModelConfig] = []

    # ── 0. CUDA XGBoost configs ──────────────────────────────────────────
    for max_depth, n_estimators, lr, mf in [
        (4, 350, 0.05, 5_000),
        (5, 450, 0.04, 5_000),
        (4, 450, 0.04, 10_000),
    ]:
        configs.append(ModelConfig(
            name=f"GBDT_XGB_cuda_word12_mf{mf}_style_d{max_depth}_n{n_estimators}_lr{lr}",
            group="G7_gbdt_cuda",
            prep_col="headline_minimal",
            estimator=_gbdt_pipeline(
                ngram=(1, 2),
                word_mf=mf,
                clf=XGBClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=lr,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_lambda=1.0,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                    device="cuda",
                    random_state=42,
                    n_jobs=1,
                ),
            ),
            description=(
                f"word (1,2) mf={mf} + stylo -> XGBoost GBDT "
                f"CUDA depth={max_depth} n={n_estimators} lr={lr}"
            ),
        ))

    # ── 1. learning-rate × max_iter sweep (word (1,2) + style) ──────────
    for lr, max_iter in [
        (0.10, 200),
        (0.05, 300),
        (0.05, 500),
        (0.01, 800),
    ]:
        configs.append(ModelConfig(
            name=f"GBDT_word12_mf5000_style_lr{lr}_iter{max_iter}",
            group="G7_gbdt",
            prep_col="headline_minimal",
            estimator=_gbdt_pipeline(
                ngram=(1, 2), word_mf=5_000,
                clf=HistGradientBoostingClassifier(
                    max_iter=max_iter, learning_rate=lr,
                    max_depth=4, min_samples_leaf=20,
                    l2_regularization=0.1, random_state=42,
                ),
            ),
            description=f"word (1,2) mf=5000 + stylo → HistGBM lr={lr} iter={max_iter}",
        ))

    # ── 2. max_depth sweep (word (1,2) + style) ──────────────────────────
    for depth in [3, 4, 5, 6]:
        configs.append(ModelConfig(
            name=f"GBDT_word12_mf5000_style_depth{depth}",
            group="G7_gbdt",
            prep_col="headline_minimal",
            estimator=_gbdt_pipeline(
                ngram=(1, 2), word_mf=5_000,
                clf=HistGradientBoostingClassifier(
                    max_iter=300, learning_rate=0.05,
                    max_depth=depth, min_samples_leaf=20,
                    l2_regularization=0.1, random_state=42,
                ),
            ),
            description=f"word (1,2) mf=5000 + stylo → HistGBM depth={depth}",
        ))

    # ── 3. word vocab size sweep ─────────────────────────────────────────
    for mf in [2_000, 5_000, 10_000, 20_000]:
        configs.append(ModelConfig(
            name=f"GBDT_word12_mf{mf}_style",
            group="G7_gbdt",
            prep_col="headline_minimal",
            estimator=_gbdt_pipeline(ngram=(1, 2), word_mf=mf),
            description=f"word (1,2) mf={mf} + stylo → HistGBM (default)",
        ))

    # ── 4. n-gram range sweep (word only, mf=5000 + style) ───────────────
    for ngram in [(1, 1), (1, 2), (1, 3), (2, 3)]:
        ng = f"{ngram[0]}{ngram[1]}"
        configs.append(ModelConfig(
            name=f"GBDT_word{ng}_mf5000_style",
            group="G7_gbdt",
            prep_col="headline_minimal",
            estimator=_gbdt_pipeline(ngram=ngram, word_mf=5_000),
            description=f"word {ngram} mf=5000 + stylo → HistGBM (default)",
        ))

    # ── 5. word + char TF-IDF + style (3-branch) ─────────────────────────
    for char_ng, char_mf in [((3, 5), 5_000), ((4, 6), 5_000), ((3, 6), 10_000)]:
        cng = f"{char_ng[0]}{char_ng[1]}"
        configs.append(ModelConfig(
            name=f"GBDT_word12_char{cng}_mf5000_style",
            group="G7_gbdt",
            prep_col="headline_minimal",
            estimator=_gbdt_pipeline(
                ngram=(1, 2), word_mf=5_000,
                char_ngram=char_ng, char_mf=char_mf,
            ),
            description=f"word (1,2) + char {char_ng} mf={char_mf} + stylo → HistGBM (default)",
        ))

    # ── 6. ablation: no stylometric features ─────────────────────────────
    for mf in [5_000, 10_000]:
        configs.append(ModelConfig(
            name=f"GBDT_word12_mf{mf}_nostyle",
            group="G7_gbdt",
            prep_col="headline_minimal",
            estimator=_gbdt_pipeline(ngram=(1, 2), word_mf=mf, include_style=False),
            description=f"word (1,2) mf={mf} only (no stylo) → HistGBM (default)",
        ))

    # ── 7. l2 regularisation sweep ────────────────────────────────────────
    for l2 in [0.0, 0.01, 0.1, 1.0]:
        configs.append(ModelConfig(
            name=f"GBDT_word12_mf5000_style_l2{l2}",
            group="G7_gbdt",
            prep_col="headline_minimal",
            estimator=_gbdt_pipeline(
                ngram=(1, 2), word_mf=5_000,
                clf=HistGradientBoostingClassifier(
                    max_iter=300, learning_rate=0.05,
                    max_depth=4, min_samples_leaf=20,
                    l2_regularization=l2, random_state=42,
                ),
            ),
            description=f"word (1,2) mf=5000 + stylo → HistGBM l2={l2}",
        ))

    return configs


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    standalone_train(
        group_name="gbdt",
        get_configs_fn=get_configs,
        log_filename="train_gbdt.log",
    )