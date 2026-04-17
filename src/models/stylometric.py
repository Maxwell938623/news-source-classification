"""
stylometric.py — Handcrafted stylometric features + tree / linear classifiers.

Fox News and NBC News have measurably different writing styles that are
independent of topic: headline length, punctuation density, capitalisation
patterns, colon/quote usage, and digit density all vary systematically by
outlet.  This file extracts 15 such features and trains several classifiers
on them, providing a purely style-based signal that is orthogonal to the
vocabulary-based TF-IDF models.

Feature vector (15 dimensions):
   0  word_count         — whitespace-separated token count
   1  char_count         — total character length
   2  avg_word_len       — mean token character length
   3  exclamation_count  — number of '!'
   4  question_count     — number of '?'
   5  ellipsis_count     — number of '...'
   6  colon_present      — 1 if ':' present
   7  quote_present      — 1 if " or ' present
   8  dash_present       — 1 if -, en-dash, or em-dash present
   9  paren_present      — 1 if '(' present
  10  cap_ratio          — fraction of alphabetic chars that are uppercase
  11  allcaps_word_count — words where every char is uppercase (len > 1)
  12  digit_ratio        — fraction of chars that are digits
  13  title_case_ratio   — fraction of words whose first char is uppercase
  14  comma_count        — number of commas

Note: headline_minimal is the correct input (capitalisation preserved).
      headline_lowercase causes cap_ratio / allcaps / title_case to be ~0,
      losing those three discriminative signals.

Classifiers trained:
  LR   LogisticRegression (regularised; good baseline + interpretable coefs)
  SVM  LinearSVC           (max-margin; strong on small tabular datasets)
  RF   RandomForestClassifier (captures non-linear feature interactions)
  HGB  HistGradientBoostingClassifier (best sklearn tabular learner; no
       scaling required; handles outliers and interactions natively)

Standalone:
    python src/models/stylometric.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

try:
    from ._base import ModelConfig, standalone_train
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _base import ModelConfig, standalone_train


# ---------------------------------------------------------------------------
# Feature names (exported so callers can label axes in plots)
# ---------------------------------------------------------------------------

FEATURE_NAMES: list[str] = [
    "word_count",
    "char_count",
    "avg_word_len",
    "exclamation_count",
    "question_count",
    "ellipsis_count",
    "colon_present",
    "quote_present",
    "dash_present",
    "paren_present",
    "cap_ratio",
    "allcaps_word_count",
    "digit_ratio",
    "title_case_ratio",
    "comma_count",
]


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _extract_features(text: str) -> list[float]:
    """Return a 15-element float list for a single headline string."""
    words     = text.split()
    n_words   = len(words)
    n_chars   = len(text)
    n_alpha   = sum(1 for c in text if c.isalpha()) or 1  # avoid div/0

    avg_word_len = (
        float(np.mean([len(w) for w in words])) if words else 0.0
    )

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
    """
    Converts a list of headline strings into a (n_samples, 15) float64 array.

    Returns a *dense* numpy array.  Callers that need sparse output for use
    inside a FeatureUnion alongside a TfidfVectorizer should wrap this in
    hybrid.SparseWrapper.
    """

    def fit(self, X, y=None):          # stateless — nothing to learn
        return self

    def transform(self, X):
        return np.array([_extract_features(t) for t in X], dtype=np.float64)

    def get_feature_names_out(self, input_features=None):
        return np.array(FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Pipeline factories
# ---------------------------------------------------------------------------

def _lr_pipeline() -> Pipeline:
    return Pipeline([
        ("style",  StyloTransformer()),
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs",
                                      random_state=42)),
    ])


def _svm_pipeline(C: float = 1.0) -> Pipeline:
    return Pipeline([
        ("style",  StyloTransformer()),
        ("scaler", StandardScaler()),
        ("clf",    LinearSVC(C=C, max_iter=3000, dual="auto", random_state=42)),
    ])


def _rf_pipeline(n_estimators: int = 300) -> Pipeline:
    # RF is scale-invariant so no scaler is needed.
    return Pipeline([
        ("style", StyloTransformer()),
        ("clf",   RandomForestClassifier(n_estimators=n_estimators,
                                         max_depth=None, min_samples_leaf=2,
                                         random_state=42, n_jobs=-1)),
    ])


def _hgb_pipeline(max_iter: int = 300) -> Pipeline:
    # HistGradientBoosting is scale-invariant and handles outliers well.
    return Pipeline([
        ("style", StyloTransformer()),
        ("clf",   HistGradientBoostingClassifier(max_iter=max_iter,
                                                  learning_rate=0.1,
                                                  max_depth=4,
                                                  random_state=42)),
    ])


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

def get_configs() -> list[ModelConfig]:
    configs: list[ModelConfig] = []

    # Logistic Regression — regularisation sweep (interpretable baseline)
    for C in [0.1, 1.0, 5.0, 10.0]:
        configs.append(ModelConfig(
            name=f"STYLO_LR_C{C}",
            group="G5_stylometric",
            prep_col="headline_minimal",
            estimator=Pipeline([
                ("style",  StyloTransformer()),
                ("scaler", StandardScaler()),
                ("clf",    LogisticRegression(C=C, max_iter=1000, solver="lbfgs",
                                              random_state=42)),
            ]),
            description=f"15 stylometric features, LR C={C}",
        ))

    # LinearSVC — C sweep
    for C in [0.1, 1.0, 5.0]:
        configs.append(ModelConfig(
            name=f"STYLO_SVM_C{C}",
            group="G5_stylometric",
            prep_col="headline_minimal",
            estimator=_svm_pipeline(C=C),
            description=f"15 stylometric features, LinearSVC C={C}",
        ))

    # Random Forest — depth / trees sweep
    for n_est in [100, 300, 500]:
        configs.append(ModelConfig(
            name=f"STYLO_RF_n{n_est}",
            group="G5_stylometric",
            prep_col="headline_minimal",
            estimator=_rf_pipeline(n_estimators=n_est),
            description=f"15 stylometric features, RandomForest n={n_est}",
        ))

    # HistGradientBoosting — iteration sweep
    for max_iter in [100, 300]:
        configs.append(ModelConfig(
            name=f"STYLO_HGB_iter{max_iter}",
            group="G5_stylometric",
            prep_col="headline_minimal",
            estimator=_hgb_pipeline(max_iter=max_iter),
            description=f"15 stylometric features, HistGBM max_iter={max_iter}",
        ))

    return configs


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    standalone_train(
        group_name="stylometric",
        get_configs_fn=get_configs,
        log_filename="train_stylometric.log",
    )
