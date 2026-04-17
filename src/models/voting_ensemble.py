"""
voting_ensemble.py — Soft-voting ensembles over diverse base classifiers.

Ensemble diversity is maximised by combining base learners that differ in:
  • feature representation  (word n-grams  vs  char n-grams  vs  stylometric)
  • inductive bias          (max-margin SVM  vs  probabilistic MNB  vs  LR)

For soft voting every base model's class probabilities are averaged.
LinearSVC (no predict_proba) is wrapped in CalibratedClassifierCV with
Platt-scaling (sigmoid method) to produce calibrated probabilities.

Configurations:
  Vote_2way_word_char          word-bigram LR + char(4,6) LR
  Vote_3way_word_char_mnb      + MNB
  Vote_3way_weighted           3-way with weights [2, 1, 1] (upweight word LR)
  Vote_4way_add_calSVM         + calibrated LinearSVC
  Vote_3way_word_char_style    word LR + char LR + stylometric LR
                                (uses headline_minimal for cap-ratio features)
  Vote_4way_all_style          word LR + char LR + MNB + stylometric LR
                                (headline_minimal input)

Standalone:
    python src/models/voting_ensemble.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
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
# Reusable base-learner factories
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
    """MNB with norm=None to preserve count-like semantics."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), max_features=mf,
            stop_words="english", lowercase=lowercase,
            sublinear_tf=True, norm=None,
        )),
        ("clf", MultinomialNB(alpha=0.1)),
    ])


def _cal_svm(C: float = 1.0, mf: int = 5_000) -> Pipeline:
    """LinearSVC with Platt-scaling calibration for predict_proba support."""
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


def _style_lr(C: float = 1.0) -> Pipeline:
    """Stylometric features + StandardScaler + LR.  Accepts raw text."""
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
    # Configs using headline_lowercase as input
    # (TF-IDF base learners use lowercase=False — column already lowercased)
    # ------------------------------------------------------------------ #

    # 2-way: vocabulary (word) + style (char)
    configs.append(ModelConfig(
        name="Vote_2way_word_char",
        group="G7_voting",
        prep_col="headline_lowercase",
        estimator=VotingClassifier(
            estimators=[
                ("word_lr", _word_lr()),
                ("char_lr", _char_lr()),
            ],
            voting="soft",
        ),
        description="Soft voting 2-way: word-bigram LR + char(4,6) LR",
    ))

    # 3-way: word + char + MNB
    configs.append(ModelConfig(
        name="Vote_3way_word_char_mnb",
        group="G7_voting",
        prep_col="headline_lowercase",
        estimator=VotingClassifier(
            estimators=[
                ("word_lr", _word_lr()),
                ("char_lr", _char_lr()),
                ("mnb",     _mnb_base()),
            ],
            voting="soft",
        ),
        description="Soft voting 3-way: word LR + char LR + MNB",
    ))

    # 3-way weighted: word LR gets double weight (usually best single model)
    configs.append(ModelConfig(
        name="Vote_3way_weighted_2_1_1",
        group="G7_voting",
        prep_col="headline_lowercase",
        estimator=VotingClassifier(
            estimators=[
                ("word_lr", _word_lr()),
                ("char_lr", _char_lr()),
                ("mnb",     _mnb_base()),
            ],
            voting="soft",
            weights=[2, 1, 1],
        ),
        description="Soft voting 3-way weights=[2,1,1]: upweight word LR",
    ))

    # 4-way: add calibrated SVM (different decision boundary from LR)
    configs.append(ModelConfig(
        name="Vote_4way_word_char_mnb_calSVM",
        group="G7_voting",
        prep_col="headline_lowercase",
        estimator=VotingClassifier(
            estimators=[
                ("word_lr",  _word_lr()),
                ("char_lr",  _char_lr()),
                ("mnb",      _mnb_base()),
                ("cal_svm",  _cal_svm()),
            ],
            voting="soft",
        ),
        description="Soft voting 4-way: word LR + char LR + MNB + calibrated SVM",
    ))

    # 4-way with diverse feature counts
    configs.append(ModelConfig(
        name="Vote_4way_diverse_features",
        group="G7_voting",
        prep_col="headline_lowercase",
        estimator=VotingClassifier(
            estimators=[
                ("word_lr_5k",   _word_lr(mf=5_000)),
                ("word_lr_20k",  _word_lr(C=2.0, mf=20_000)),
                ("char_lr_10k",  _char_lr(mf=10_000)),
                ("mnb_10k",      _mnb_base(mf=10_000)),
            ],
            voting="soft",
        ),
        description="Soft voting 4-way: varied feature counts for diversity",
    ))

    # ------------------------------------------------------------------ #
    # Configs using headline_minimal as input
    # (preserves capitalisation so stylometric features are fully informative;
    #  TF-IDF branches use lowercase=True for normalisation)
    # ------------------------------------------------------------------ #

    # 3-way with stylometric branch
    configs.append(ModelConfig(
        name="Vote_3way_word_char_style",
        group="G7_voting",
        prep_col="headline_minimal",
        estimator=VotingClassifier(
            estimators=[
                ("word_lr", _word_lr(lowercase=True)),
                ("char_lr", _char_lr(lowercase=False)),
                ("style_lr", _style_lr()),
            ],
            voting="soft",
        ),
        description=(
            "Soft voting 3-way (minimal input): word LR + char LR + stylometric LR"
        ),
    ))

    # 4-way: all branches including MNB (also with lowercase=True for minimal input)
    configs.append(ModelConfig(
        name="Vote_4way_word_char_mnb_style",
        group="G7_voting",
        prep_col="headline_minimal",
        estimator=VotingClassifier(
            estimators=[
                ("word_lr",  _word_lr(lowercase=True)),
                ("char_lr",  _char_lr(lowercase=False)),
                ("mnb",      _mnb_base(lowercase=True)),
                ("style_lr", _style_lr()),
            ],
            voting="soft",
        ),
        description=(
            "Soft voting 4-way (minimal input): word LR + char LR + MNB + stylometric LR"
        ),
    ))

    return configs


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    standalone_train(
        group_name="voting_ensemble",
        get_configs_fn=get_configs,
        log_filename="train_voting_ensemble.log",
    )
