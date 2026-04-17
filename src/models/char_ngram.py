"""
char_ngram.py — Character n-gram TF-IDF experiments.

Character n-grams capture stylometric signals that word n-grams miss:
punctuation density, affixes, hyphenation patterns, and capitalisation
(when headline_minimal is used as input).

analyzer='char_wb' pads each token with spaces so n-grams never cross word
boundaries, which outperforms 'char' for short headline text.

Groups:
  G1_CNG  n-gram range sweep  (LR, headline_lowercase, mf=5000)
  G2_CNG  feature-count sweep (LR, best ngram range)
  G3_CNG  capitalisation test (same configs on headline_minimal to preserve
                               case information as an additional signal)
  G4_CNG  SVM variants        (LinearSVC on char n-grams, C sweep)
  G5_CNG  LR + char + word combined
          (FeatureUnion of char_wb and word bigrams — within a single Pipeline)

Standalone:
    python src/models/char_ngram.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

try:
    from ._base import ModelConfig, standalone_train
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _base import ModelConfig, standalone_train


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def _lr(C: float = 1.0) -> LogisticRegression:
    return LogisticRegression(C=C, max_iter=1000, solver="lbfgs", random_state=42)


def _svc(C: float = 1.0) -> LinearSVC:
    return LinearSVC(C=C, max_iter=3000, dual="auto", random_state=42)


def _char_pipeline(ngram: tuple, mf: int, C: float = 1.0,
                   use_svm: bool = False, lowercase: bool = False) -> Pipeline:
    clf = _svc(C) if use_svm else _lr(C)
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb", ngram_range=ngram, max_features=mf,
            lowercase=lowercase, sublinear_tf=True,
        )),
        ("clf", clf),
    ])


def _combined_pipeline(char_ngram: tuple, word_mf: int, char_mf: int,
                       C: float = 1.0) -> Pipeline:
    """FeatureUnion of char_wb and word bigrams feeding a single LR."""
    return Pipeline([
        ("features", FeatureUnion([
            ("char", TfidfVectorizer(
                analyzer="char_wb", ngram_range=char_ngram, max_features=char_mf,
                lowercase=False, sublinear_tf=True,
            )),
            ("word", TfidfVectorizer(
                analyzer="word", ngram_range=(1, 2), max_features=word_mf,
                stop_words="english", lowercase=False, sublinear_tf=True,
            )),
        ])),
        ("clf", _lr(C=C)),
    ])


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

def get_configs() -> list[ModelConfig]:
    configs: list[ModelConfig] = []

    # ---- G1_CNG: n-gram range sweep (LR, headline_lowercase, mf=5000) ---
    for ngram in [(2, 4), (3, 5), (3, 6), (4, 6), (4, 7), (5, 7)]:
        ng = f"{ngram[0]}{ngram[1]}"
        configs.append(ModelConfig(
            name=f"G1_CNG_LR_c{ng}_mf5000_lower",
            group="G1_CNG_ngram_range",
            prep_col="headline_lowercase",
            estimator=_char_pipeline(ngram, 5_000, lowercase=False),
            description=f"char_wb {ngram}, mf=5000, LR C=1, lowercase input",
        ))

    # ---- G2_CNG: feature-count sweep (LR, (4,6) range, headline_lowercase)
    for mf in [1_000, 2_000, 5_000, 10_000, 20_000]:
        configs.append(ModelConfig(
            name=f"G2_CNG_LR_c46_mf{mf}_lower",
            group="G2_CNG_feature_count",
            prep_col="headline_lowercase",
            estimator=_char_pipeline((4, 6), mf, lowercase=False),
            description=f"char_wb (4,6), mf={mf}, LR C=1",
        ))

    # ---- G3_CNG: capitalisation preserved (headline_minimal input) -------
    # Using headline_minimal keeps capitalisation as a discriminative signal.
    # TfidfVectorizer.lowercase=False preserves case in the n-grams.
    for ngram in [(3, 5), (3, 6), (4, 6)]:
        ng = f"{ngram[0]}{ngram[1]}"
        for mf in [5_000, 10_000]:
            configs.append(ModelConfig(
                name=f"G3_CNG_LR_c{ng}_mf{mf}_minimal",
                group="G3_CNG_capitalisation",
                prep_col="headline_minimal",
                estimator=_char_pipeline(ngram, mf, lowercase=False),
                description=f"char_wb {ngram}, mf={mf}, LR C=1, capitalisation preserved",
            ))

    # ---- G4_CNG: LinearSVC on char n-grams (C sweep) --------------------
    for C in [0.1, 0.5, 1.0, 5.0]:
        configs.append(ModelConfig(
            name=f"G4_CNG_SVM_c46_mf5000_C{C}",
            group="G4_CNG_svm",
            prep_col="headline_lowercase",
            estimator=_char_pipeline((4, 6), 5_000, C=C, use_svm=True, lowercase=False),
            description=f"char_wb (4,6), mf=5000, LinearSVC C={C}",
        ))

    # ---- G5_CNG: char + word FeatureUnion --------------------------------
    # Combining character and word n-grams often yields better accuracy than
    # either alone because the two representations capture complementary signals.
    for char_ng, word_mf, char_mf in [
        ((4, 6), 5_000, 5_000),
        ((3, 6), 5_000, 10_000),
        ((4, 6), 10_000, 10_000),
    ]:
        cng = f"{char_ng[0]}{char_ng[1]}"
        configs.append(ModelConfig(
            name=f"G5_CNG_combined_c{cng}_wmf{word_mf}_cmf{char_mf}",
            group="G5_CNG_combined",
            prep_col="headline_lowercase",
            estimator=_combined_pipeline(char_ng, word_mf, char_mf, C=1.0),
            description=(
                f"char_wb {char_ng} mf={char_mf} + word (1,2) mf={word_mf} | LR C=1"
            ),
        ))

    return configs


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    standalone_train(
        group_name="char_ngram",
        get_configs_fn=get_configs,
        log_filename="train_char_ngram.log",
    )
