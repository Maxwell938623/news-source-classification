"""
hybrid.py — TF-IDF vocabulary features + stylometric features combined.

Architecture:
    FeatureUnion([
        ("text",  TfidfVectorizer(...)),          → sparse (n, vocab)
        ("style", SparseWrapper(StyloTransformer())) → sparse (n, 15)
    ])
    → horizontally stacked sparse matrix (n, vocab+15)
    → classifier

Both branches return scipy csr_matrix so sklearn's FeatureUnion can
concatenate them with scipy.sparse.hstack without any dense conversion.

The TF-IDF branch uses lowercase=True because the input column is
headline_minimal (capitalisation preserved for stylometric branch).

Groups:
  G6_hybrid  varied TF-IDF configurations + LR or HistGB classifier

Standalone:
    python src/models/hybrid.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MaxAbsScaler

try:
    from ._base import ModelConfig, standalone_train
    from .stylometric import StyloTransformer
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _base import ModelConfig, standalone_train
    from stylometric import StyloTransformer


# ---------------------------------------------------------------------------
# Sparse wrapper
# ---------------------------------------------------------------------------

class SparseWrapper(BaseEstimator, TransformerMixin):
    """
    Wraps any transformer that returns a dense array, converting its output
    to a csr_matrix so it can be used inside a FeatureUnion alongside a
    TfidfVectorizer (which returns sparse).
    """

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
# Pipeline factory
# ---------------------------------------------------------------------------

def _hybrid_pipeline(
    ngram: tuple[int, int] = (1, 2),
    word_mf: int = 5_000,
    char_ngram: tuple[int, int] | None = None,
    char_mf: int = 5_000,
    clf=None,
) -> Pipeline:
    """
    Build a hybrid text + stylometric pipeline.

    If char_ngram is given, three feature branches are combined:
        word TF-IDF  +  char_wb TF-IDF  +  stylometric

    Otherwise two branches:
        word TF-IDF  +  stylometric

    Input expected: headline_minimal (capitalisation preserved for stylometric).
    TF-IDF branches use lowercase=True for proper vocabulary normalisation.
    """
    if clf is None:
        clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs",
                                 random_state=42)

    branches: list[tuple[str, TransformerMixin]] = [
        ("word_tfidf", TfidfVectorizer(
            analyzer="word", ngram_range=ngram, max_features=word_mf,
            stop_words="english", lowercase=True, sublinear_tf=True,
        )),
        ("style", SparseWrapper(StyloTransformer())),
    ]

    if char_ngram is not None:
        branches.insert(1, (
            "char_tfidf",
            TfidfVectorizer(
                analyzer="char_wb", ngram_range=char_ngram, max_features=char_mf,
                lowercase=False, sublinear_tf=True,
            ),
        ))

    # MaxAbsScaler normalises each feature column to [-1,1] while preserving
    # sparsity.  It prevents the stylometric features (small absolute values)
    # from being drowned out by the high-dimensional TF-IDF block.
    return Pipeline([
        ("features", FeatureUnion(branches)),
        ("scaler",   MaxAbsScaler()),
        ("clf",      clf),
    ])


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

def get_configs() -> list[ModelConfig]:
    configs: list[ModelConfig] = []

    # ---- Word TF-IDF + stylometric, LR (varied mf and ngram) -----------
    for ngram, mf in [((1, 1), 5_000), ((1, 2), 5_000), ((1, 2), 10_000),
                      ((1, 3), 5_000)]:
        ng = f"{ngram[0]}{ngram[1]}"
        configs.append(ModelConfig(
            name=f"HYB_word{ng}_mf{mf}_style_LR",
            group="G6_hybrid",
            prep_col="headline_minimal",
            estimator=_hybrid_pipeline(ngram=ngram, word_mf=mf),
            description=f"word {ngram} mf={mf} + 15 stylo features → LR C=1",
        ))

    # ---- Word TF-IDF + stylometric, LR, C sweep -------------------------
    for C in [0.1, 0.5, 2.0, 5.0]:
        configs.append(ModelConfig(
            name=f"HYB_word12_mf5000_style_LR_C{C}",
            group="G6_hybrid",
            prep_col="headline_minimal",
            estimator=_hybrid_pipeline(
                ngram=(1, 2), word_mf=5_000,
                clf=LogisticRegression(C=C, max_iter=1000, solver="lbfgs",
                                       random_state=42),
            ),
            description=f"word (1,2) mf=5000 + stylo → LR C={C}",
        ))

    # ---- Word TF-IDF + stylometric, HistGradientBoosting ----------------
    # HGB handles mixed-scale features natively; it may find non-linear
    # interactions between stylometric signals and vocabulary patterns.
    for max_iter in [200, 400]:
        configs.append(ModelConfig(
            name=f"HYB_word12_mf5000_style_HGB_iter{max_iter}",
            group="G6_hybrid",
            prep_col="headline_minimal",
            estimator=_hybrid_pipeline(
                ngram=(1, 2), word_mf=5_000,
                clf=HistGradientBoostingClassifier(
                    max_iter=max_iter, learning_rate=0.1,
                    max_depth=4, random_state=42,
                ),
            ),
            description=f"word (1,2) mf=5000 + stylo → HistGBM iter={max_iter}",
        ))

    # ---- Word + Char TF-IDF + stylometric (3 branches), LR -------------
    for char_ng, char_mf in [((4, 6), 5_000), ((3, 6), 5_000)]:
        cng = f"{char_ng[0]}{char_ng[1]}"
        configs.append(ModelConfig(
            name=f"HYB_word12_char{cng}_mf5000_style_LR",
            group="G6_hybrid",
            prep_col="headline_minimal",
            estimator=_hybrid_pipeline(
                ngram=(1, 2), word_mf=5_000,
                char_ngram=char_ng, char_mf=char_mf,
            ),
            description=(
                f"word (1,2) + char {char_ng} + stylo (3-branch) → LR C=1"
            ),
        ))

    return configs


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    standalone_train(
        group_name="hybrid",
        get_configs_fn=get_configs,
        log_filename="train_hybrid.log",
    )
