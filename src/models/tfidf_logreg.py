"""
tfidf_logreg.py — TF-IDF (word n-grams) + Logistic Regression experiments.

Groups:
  G1_LR  feature-count sweep  (100 → 20 000, unigrams, C=1)
  G2_LR  n-gram range sweep   ((1,1)/(1,2)/(1,3)/(2,2) at mf=5000, C=1)
  G3_LR  C regularization     (0.01 → 20, bigrams mf=5000)
  G4_LR  preprocessing sweep  (all 5 headline variants, bigrams mf=5000, C=1)

All experiments use sublinear_tf=True (1+log(tf)) which consistently
outperforms raw tf for short headline text.

Standalone:
    python src/models/tfidf_logreg.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

try:
    from ._base import ModelConfig, standalone_train
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _base import ModelConfig, standalone_train


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def _lr(C: float = 1.0, max_iter: int = 1000) -> LogisticRegression:
    return LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs", random_state=42)


def _pipeline(vec_kw: dict, C: float = 1.0) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(**vec_kw)),
        ("clf",   _lr(C=C)),
    ])


# Shared base kwargs: word analyser, English stop-words, no extra lowercasing
# (headline_lowercase column is already lowercased), sublinear_tf for stability.
_BASE = dict(
    analyzer="word",
    stop_words="english",
    lowercase=False,
    sublinear_tf=True,
)


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

def get_configs() -> list[ModelConfig]:
    configs: list[ModelConfig] = []

    # ---- G1_LR: feature-count sweep -------------------------------------
    for mf in [100, 500, 1_000, 2_000, 5_000, 10_000, 20_000]:
        configs.append(ModelConfig(
            name=f"G1_LR_w11_mf{mf}",
            group="G1_LR_feature_count",
            prep_col="headline_lowercase",
            estimator=_pipeline({**_BASE, "ngram_range": (1, 1), "max_features": mf}),
            description=f"word unigrams, mf={mf}, C=1",
        ))

    # ---- G2_LR: n-gram sweep at mf=5000 ----------------------------------
    for ngram in [(1, 1), (1, 2), (1, 3), (2, 2)]:
        ng = f"{ngram[0]}{ngram[1]}"
        configs.append(ModelConfig(
            name=f"G2_LR_w{ng}_mf5000",
            group="G2_LR_ngrams",
            prep_col="headline_lowercase",
            estimator=_pipeline({**_BASE, "ngram_range": ngram, "max_features": 5_000}),
            description=f"word {ngram}-grams, mf=5000, C=1",
        ))

    # ---- G3_LR: C regularization sweep (bigrams, mf=5000) ---------------
    for C in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
        configs.append(ModelConfig(
            name=f"G3_LR_w12_mf5000_C{C}",
            group="G3_LR_regularization",
            prep_col="headline_lowercase",
            estimator=_pipeline(
                {**_BASE, "ngram_range": (1, 2), "max_features": 5_000}, C=C
            ),
            description=f"word (1,2), mf=5000, C={C}",
        ))

    # ---- G4_LR: preprocessing variant sweep (bigrams, mf=5000, C=1) ----
    # For minimal: TF-IDF lowercases internally (lowercase=True).
    # For nostop/lemma: stopwords already removed by preprocess.py,
    #   so we do NOT pass stop_words="english" here (would confound the comparison).
    PREP_VARIANTS = [
        ("headline_minimal",   True,  "english", "minimal"),
        ("headline_lowercase", False, "english", "lowercase"),
        ("headline_nopunct",   False, "english", "nopunct"),
        ("headline_nostop",    False, None,      "nostop"),
        ("headline_lemma",     False, None,      "lemma"),
    ]
    for col, do_lower, sw, tag in PREP_VARIANTS:
        configs.append(ModelConfig(
            name=f"G4_LR_w12_mf5000_prep_{tag}",
            group="G4_LR_preprocessing",
            prep_col=col,
            estimator=Pipeline([
                ("tfidf", TfidfVectorizer(
                    analyzer="word", ngram_range=(1, 2), max_features=5_000,
                    stop_words=sw, lowercase=do_lower, sublinear_tf=True,
                )),
                ("clf", _lr(C=1.0)),
            ]),
            description=f"word (1,2), mf=5000, C=1, prep={tag}",
        ))

    return configs


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    standalone_train(
        group_name="tfidf_logreg",
        get_configs_fn=get_configs,
        log_filename="train_tfidf_logreg.log",
    )
