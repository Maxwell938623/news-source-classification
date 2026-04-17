"""
tfidf_svm.py — TF-IDF (word n-grams) + LinearSVC experiments.

LinearSVC consistently beats LR on sparse TF-IDF features for short-text
classification; the hinge-loss large-margin objective is well-suited to
high-dimensional, well-separated feature spaces.

Groups:
  G1_SVM  C regularization sweep  (bigrams, mf=5000)
  G2_SVM  feature-count sweep     (at best C, bigrams)
  G3_SVM  n-gram range sweep      (at best C, mf=5000)
  G4_SVM  calibrated variants     (CalibratedClassifierCV for predict_proba
                                   — needed when used as ensemble base learner)

Standalone:
    python src/models/tfidf_svm.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

try:
    from ._base import ModelConfig, standalone_train
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _base import ModelConfig, standalone_train


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def _svc(C: float = 1.0, max_iter: int = 3000) -> LinearSVC:
    return LinearSVC(C=C, max_iter=max_iter, dual="auto", random_state=42)


def _cal_svc(C: float = 1.0, cv: int = 3) -> CalibratedClassifierCV:
    """LinearSVC wrapped in Platt-scaling calibration for predict_proba support."""
    return CalibratedClassifierCV(_svc(C=C), cv=cv, method="sigmoid")


def _pipeline(vec_kw: dict, C: float = 1.0, calibrated: bool = False) -> Pipeline:
    clf = _cal_svc(C=C) if calibrated else _svc(C=C)
    return Pipeline([
        ("tfidf", TfidfVectorizer(**vec_kw)),
        ("clf",   clf),
    ])


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

    # ---- G1_SVM: C regularization sweep (bigrams, mf=5000) ---------------
    for C in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        configs.append(ModelConfig(
            name=f"G1_SVM_w12_mf5000_C{C}",
            group="G1_SVM_regularization",
            prep_col="headline_lowercase",
            estimator=_pipeline({**_BASE, "ngram_range": (1, 2), "max_features": 5_000}, C=C),
            description=f"LinearSVC, word (1,2), mf=5000, C={C}",
        ))

    # ---- G2_SVM: feature-count sweep at C=1 (bigrams) --------------------
    for mf in [500, 1_000, 2_000, 5_000, 10_000, 20_000]:
        configs.append(ModelConfig(
            name=f"G2_SVM_w12_mf{mf}_C1",
            group="G2_SVM_feature_count",
            prep_col="headline_lowercase",
            estimator=_pipeline({**_BASE, "ngram_range": (1, 2), "max_features": mf}, C=1.0),
            description=f"LinearSVC, word (1,2), mf={mf}, C=1",
        ))

    # ---- G3_SVM: n-gram range sweep at C=1, mf=5000 ---------------------
    for ngram in [(1, 1), (1, 2), (1, 3), (2, 2)]:
        ng = f"{ngram[0]}{ngram[1]}"
        configs.append(ModelConfig(
            name=f"G3_SVM_w{ng}_mf5000_C1",
            group="G3_SVM_ngrams",
            prep_col="headline_lowercase",
            estimator=_pipeline({**_BASE, "ngram_range": ngram, "max_features": 5_000}, C=1.0),
            description=f"LinearSVC, word {ngram}, mf=5000, C=1",
        ))

    # ---- G4_SVM: calibrated variants (sigmoid Platt scaling) --------------
    # Calibrated SVMs are needed when SVMs serve as base learners in a soft-
    # voting or stacking ensemble (requires predict_proba).  Training is
    # ~3× slower due to cross-validated Platt scaling.
    for C in [0.1, 0.5, 1.0, 5.0]:
        configs.append(ModelConfig(
            name=f"G4_SVM_cal_w12_mf5000_C{C}",
            group="G4_SVM_calibrated",
            prep_col="headline_lowercase",
            estimator=_pipeline(
                {**_BASE, "ngram_range": (1, 2), "max_features": 5_000},
                C=C,
                calibrated=True,
            ),
            description=f"CalibratedSVC (sigmoid), word (1,2), mf=5000, C={C}",
        ))

    return configs


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    standalone_train(
        group_name="tfidf_svm",
        get_configs_fn=get_configs,
        log_filename="train_tfidf_svm.log",
    )
