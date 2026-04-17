"""
stacking_ensemble.py — Stacking ensembles with diverse base learners.

Stacking trains a meta-learner on out-of-fold predictions from the base
models (using k-fold cross-validation), which lets it learn when each base
model is trustworthy.  This is more powerful than simple voting because the
meta-learner can up-weight confident bases and down-weight confused ones.

All base learners must support predict_proba (required by stack_method=
'predict_proba').  LinearSVC is wrapped in CalibratedClassifierCV.

Configurations:
  Stack_3base_LR_meta          word LR + char LR + MNB  →  LR meta
  Stack_3base_HGB_meta         word LR + char LR + MNB  →  HistGBM meta
                                (non-linear meta can capture interactions
                                 between base-model confidence signals)
  Stack_4base_style_LR_meta    + stylometric LR base learner
                                (uses headline_minimal for cap features)
  Stack_5base_calSVM_LR_meta   + calibrated LinearSVM base learner
  Stack_3base_charSVM_LR_meta  replace char LR with calibrated char SVM

All stacking configs use 5-fold CV (cv=5) for generating meta-features,
except the 5-base config which uses cv=3 to limit training time.

Standalone:
    python src/models/stacking_ensemble.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, StackingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

try:
    from ._base import ModelConfig, standalone_train
    from .stylometric import StyloTransformer
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _base import ModelConfig, standalone_train
    from stylometric import StyloTransformer


# ---------------------------------------------------------------------------
# Base-learner factories  (all return predict_proba-capable estimators)
# ---------------------------------------------------------------------------

def _word_lr(C: float = 1.0, mf: int = 5_000,
             lowercase: bool = False) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), max_features=mf,
            stop_words="english", lowercase=lowercase, sublinear_tf=True,
        )),
        ("clf", LogisticRegression(C=C, max_iter=1000, solver="lbfgs",
                                   random_state=42)),
    ])


def _char_lr(ngram: tuple = (4, 6), mf: int = 5_000,
             lowercase: bool = False) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb", ngram_range=ngram, max_features=mf,
            lowercase=lowercase, sublinear_tf=True,
        )),
        ("clf", LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs",
                                   random_state=42)),
    ])


def _mnb_base(mf: int = 10_000, lowercase: bool = False) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), max_features=mf,
            stop_words="english", lowercase=lowercase,
            sublinear_tf=True, norm=None,
        )),
        ("clf", MultinomialNB(alpha=0.1)),
    ])


def _cal_svm_word(C: float = 1.0, mf: int = 5_000) -> Pipeline:
    """Word-bigram TF-IDF + Platt-calibrated LinearSVC (has predict_proba)."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), max_features=mf,
            stop_words="english", lowercase=False, sublinear_tf=True,
        )),
        ("clf", CalibratedClassifierCV(
            LinearSVC(C=C, max_iter=3000, dual="auto", random_state=42),
            cv=3, method="sigmoid",
        )),
    ])


def _cal_svm_char(C: float = 1.0, ngram: tuple = (4, 6),
                  mf: int = 5_000) -> Pipeline:
    """Char-ngram TF-IDF + Platt-calibrated LinearSVC."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb", ngram_range=ngram, max_features=mf,
            lowercase=False, sublinear_tf=True,
        )),
        ("clf", CalibratedClassifierCV(
            LinearSVC(C=C, max_iter=3000, dual="auto", random_state=42),
            cv=3, method="sigmoid",
        )),
    ])


def _style_lr(C: float = 1.0) -> Pipeline:
    """Stylometric features → StandardScaler → LR (predict_proba native)."""
    return Pipeline([
        ("style",  StyloTransformer()),
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(C=C, max_iter=1000, solver="lbfgs",
                                      random_state=42)),
    ])


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

def get_configs() -> list[ModelConfig]:
    configs: list[ModelConfig] = []

    # ------------------------------------------------------------------ #
    # headline_lowercase input
    # ------------------------------------------------------------------ #

    # 3-base + LR meta  (the canonical strong ensemble)
    configs.append(ModelConfig(
        name="Stack_3base_LR_meta",
        group="G8_stacking",
        prep_col="headline_lowercase",
        estimator=StackingClassifier(
            estimators=[
                ("word_lr", _word_lr()),
                ("char_lr", _char_lr()),
                ("mnb",     _mnb_base()),
            ],
            final_estimator=LogisticRegression(C=1.0, max_iter=1000,
                                               random_state=42),
            cv=5,
            stack_method="predict_proba",
        ),
        description="Stacking 3-base (word LR + char LR + MNB) → LR meta, cv=5",
    ))

    # 3-base + HistGBM meta  (non-linear meta-learner)
    configs.append(ModelConfig(
        name="Stack_3base_HGB_meta",
        group="G8_stacking",
        prep_col="headline_lowercase",
        estimator=StackingClassifier(
            estimators=[
                ("word_lr", _word_lr()),
                ("char_lr", _char_lr()),
                ("mnb",     _mnb_base()),
            ],
            final_estimator=HistGradientBoostingClassifier(
                max_iter=200, learning_rate=0.1, max_depth=3, random_state=42,
            ),
            cv=5,
            stack_method="predict_proba",
        ),
        description="Stacking 3-base → HistGBM meta (non-linear; captures conf. interactions)",
    ))

    # 3-base + LR meta with tuned C on meta
    configs.append(ModelConfig(
        name="Stack_3base_LR_meta_C0.1",
        group="G8_stacking",
        prep_col="headline_lowercase",
        estimator=StackingClassifier(
            estimators=[
                ("word_lr", _word_lr()),
                ("char_lr", _char_lr()),
                ("mnb",     _mnb_base()),
            ],
            final_estimator=LogisticRegression(C=0.1, max_iter=1000,
                                               random_state=42),
            cv=5,
            stack_method="predict_proba",
        ),
        description="Stacking 3-base → LR meta (C=0.1, stronger regularisation)",
    ))

    # 5-base: add calibrated word-SVM + larger word LR
    configs.append(ModelConfig(
        name="Stack_5base_calSVM_LR_meta",
        group="G8_stacking",
        prep_col="headline_lowercase",
        estimator=StackingClassifier(
            estimators=[
                ("word_lr",      _word_lr()),
                ("word_lr_20k",  _word_lr(C=2.0, mf=20_000)),
                ("char_lr",      _char_lr()),
                ("mnb",          _mnb_base()),
                ("cal_svm_word", _cal_svm_word()),
            ],
            final_estimator=LogisticRegression(C=1.0, max_iter=1000,
                                               random_state=42),
            cv=3,   # cv=3 to keep runtime manageable with 5 base learners
            stack_method="predict_proba",
        ),
        description="Stacking 5-base (+ cal SVM + large word LR) → LR meta, cv=3",
    ))

    # Char-SVM variant: replace char LR with calibrated char SVM
    configs.append(ModelConfig(
        name="Stack_3base_charSVM_LR_meta",
        group="G8_stacking",
        prep_col="headline_lowercase",
        estimator=StackingClassifier(
            estimators=[
                ("word_lr",      _word_lr()),
                ("cal_svm_char", _cal_svm_char()),   # max-margin on char n-grams
                ("mnb",          _mnb_base()),
            ],
            final_estimator=LogisticRegression(C=1.0, max_iter=1000,
                                               random_state=42),
            cv=5,
            stack_method="predict_proba",
        ),
        description="Stacking: word LR + cal char SVM + MNB → LR meta",
    ))

    # ------------------------------------------------------------------ #
    # headline_minimal input — adds stylometric base learner
    # TF-IDF branches use lowercase=True for normalisation.
    # ------------------------------------------------------------------ #

    # 4-base: word LR + char LR + MNB + stylometric LR
    configs.append(ModelConfig(
        name="Stack_4base_style_LR_meta",
        group="G8_stacking",
        prep_col="headline_minimal",
        estimator=StackingClassifier(
            estimators=[
                ("word_lr",  _word_lr(lowercase=True)),
                ("char_lr",  _char_lr(lowercase=False)),
                ("mnb",      _mnb_base(lowercase=True)),
                ("style_lr", _style_lr()),
            ],
            final_estimator=LogisticRegression(C=1.0, max_iter=1000,
                                               random_state=42),
            cv=5,
            stack_method="predict_proba",
        ),
        description=(
            "Stacking 4-base (minimal input): word LR + char LR + MNB + stylo LR "
            "→ LR meta, cv=5"
        ),
    ))

    # 4-base + HistGBM meta (non-linear meta on stylometric-enriched base)
    configs.append(ModelConfig(
        name="Stack_4base_style_HGB_meta",
        group="G8_stacking",
        prep_col="headline_minimal",
        estimator=StackingClassifier(
            estimators=[
                ("word_lr",  _word_lr(lowercase=True)),
                ("char_lr",  _char_lr(lowercase=False)),
                ("mnb",      _mnb_base(lowercase=True)),
                ("style_lr", _style_lr()),
            ],
            final_estimator=HistGradientBoostingClassifier(
                max_iter=200, learning_rate=0.1, max_depth=3, random_state=42,
            ),
            cv=5,
            stack_method="predict_proba",
        ),
        description=(
            "Stacking 4-base (minimal input): word LR + char LR + MNB + stylo LR "
            "→ HistGBM meta, cv=5"
        ),
    ))

    return configs


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    standalone_train(
        group_name="stacking_ensemble",
        get_configs_fn=get_configs,
        log_filename="train_stacking_ensemble.log",
    )
