"""
tfidf_nb.py — TF-IDF (word n-grams) + Multinomial Naïve Bayes experiments.

MNB treats TF-IDF weights as pseudo-counts.  L2 normalisation (sklearn's
default, norm='l2') destroys that count semantics, so we always set norm=None.

Groups:
  G1_MNB  alpha (Laplace smoothing) sweep   (bigrams, mf=5000)
  G2_MNB  feature-count sweep               (at best alpha, bigrams)
  G3_MNB  n-gram range sweep                (at best alpha, mf=5000)
  G4_MNB  preprocessing sweep               (bigrams, mf=5000, best alpha)

Standalone:
    python src/models/tfidf_nb.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

try:
    from ._base import ModelConfig, standalone_train
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _base import ModelConfig, standalone_train


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def _pipeline(vec_kw: dict, alpha: float = 0.1) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(**vec_kw)),
        ("clf",   MultinomialNB(alpha=alpha)),
    ])


# norm=None is critical: L2 normalisation collapses the count-like magnitude
# differences that MNB's likelihood model relies on.
_BASE = dict(
    analyzer="word",
    stop_words="english",
    lowercase=False,
    sublinear_tf=True,
    norm=None,
)


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

def get_configs() -> list[ModelConfig]:
    configs: list[ModelConfig] = []

    # ---- G1_MNB: alpha sweep (bigrams, mf=5000) --------------------------
    for alpha in [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0]:
        configs.append(ModelConfig(
            name=f"G1_MNB_w12_mf5000_a{alpha}",
            group="G1_MNB_alpha",
            prep_col="headline_lowercase",
            estimator=_pipeline({**_BASE, "ngram_range": (1, 2), "max_features": 5_000}, alpha),
            description=f"MNB alpha={alpha}, word (1,2), mf=5000, norm=None",
            needs_nonneg=True,
        ))

    # ---- G2_MNB: feature-count sweep at alpha=0.1 (bigrams) --------------
    for mf in [500, 1_000, 2_000, 5_000, 10_000, 20_000]:
        configs.append(ModelConfig(
            name=f"G2_MNB_w12_mf{mf}_a0.1",
            group="G2_MNB_feature_count",
            prep_col="headline_lowercase",
            estimator=_pipeline({**_BASE, "ngram_range": (1, 2), "max_features": mf}, 0.1),
            description=f"MNB alpha=0.1, word (1,2), mf={mf}, norm=None",
            needs_nonneg=True,
        ))

    # ---- G3_MNB: n-gram range sweep at alpha=0.1, mf=5000 ---------------
    for ngram in [(1, 1), (1, 2), (1, 3)]:
        ng = f"{ngram[0]}{ngram[1]}"
        configs.append(ModelConfig(
            name=f"G3_MNB_w{ng}_mf5000_a0.1",
            group="G3_MNB_ngrams",
            prep_col="headline_lowercase",
            estimator=_pipeline({**_BASE, "ngram_range": ngram, "max_features": 5_000}, 0.1),
            description=f"MNB alpha=0.1, word {ngram}, mf=5000, norm=None",
            needs_nonneg=True,
        ))

    # ---- G4_MNB: preprocessing sweep (bigrams, mf=5000, alpha=0.1) ------
    PREP_VARIANTS = [
        ("headline_minimal",   True,  "english", "minimal"),
        ("headline_lowercase", False, "english", "lowercase"),
        ("headline_nopunct",   False, "english", "nopunct"),
        ("headline_nostop",    False, None,      "nostop"),
        ("headline_lemma",     False, None,      "lemma"),
    ]
    for col, do_lower, sw, tag in PREP_VARIANTS:
        configs.append(ModelConfig(
            name=f"G4_MNB_w12_mf5000_prep_{tag}",
            group="G4_MNB_preprocessing",
            prep_col=col,
            estimator=Pipeline([
                ("tfidf", TfidfVectorizer(
                    analyzer="word", ngram_range=(1, 2), max_features=5_000,
                    stop_words=sw, lowercase=do_lower, sublinear_tf=True, norm=None,
                )),
                ("clf", MultinomialNB(alpha=0.1)),
            ]),
            description=f"MNB alpha=0.1, word (1,2), mf=5000, prep={tag}",
            needs_nonneg=True,
        ))

    return configs


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    standalone_train(
        group_name="tfidf_nb",
        get_configs_fn=get_configs,
        log_filename="train_tfidf_nb.log",
    )
