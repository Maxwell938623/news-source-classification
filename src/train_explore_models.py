#!/usr/bin/env python3
"""
Train and compare three model families:
1) TF-IDF word + char n-grams + LinearSVC
2) fastText supervised (wordNgrams=2/3)
3) Static averaged embeddings (Word2Vec/FastText) + XGBoost/MLP
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"


@dataclass
class FamilyResult:
    family: str
    best_variant: str
    val_accuracy: float
    val_f1_macro: float
    test_accuracy: float
    test_f1_macro: float
    details: dict[str, Any]


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }


def _load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    val_df = pd.read_csv(SPLITS_DIR / "val.csv")
    test_df = pd.read_csv(SPLITS_DIR / "test.csv")
    return train_df, val_df, test_df


def _clean_for_fasttext(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip().lower()
    return text if text else "<empty>"


def run_word_char_svm(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> FamilyResult:
    y_train = train_df["label"].astype(int).to_numpy()
    y_val = val_df["label"].astype(int).to_numpy()
    y_test = test_df["label"].astype(int).to_numpy()
    X_train = train_df["headline_lowercase"].fillna("").astype(str).tolist()
    X_val = val_df["headline_lowercase"].fillna("").astype(str).tolist()
    X_test = test_df["headline_lowercase"].fillna("").astype(str).tolist()

    candidates: list[tuple[str, Pipeline]] = []
    for word_mf, char_mf, C in [(20_000, 20_000, 1.0), (30_000, 30_000, 1.0), (30_000, 50_000, 1.5)]:
        name = f"wc_svm_wmf{word_mf}_cmf{char_mf}_C{C}"
        pipe = Pipeline([
            ("features", FeatureUnion([
                ("word", TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    max_features=word_mf,
                    stop_words="english",
                    sublinear_tf=True,
                    lowercase=False,
                )),
                ("char", TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 6),
                    max_features=char_mf,
                    sublinear_tf=True,
                    lowercase=False,
                )),
            ])),
            ("clf", LinearSVC(C=C, dual="auto", max_iter=6000, random_state=42)),
        ])
        candidates.append((name, pipe))

    scored: list[tuple[float, float, str, Pipeline]] = []
    for name, est in candidates:
        est.fit(X_train, y_train)
        pred = est.predict(X_val)
        m = _metrics(y_val, pred)
        scored.append((m["f1_macro"], m["accuracy"], name, est))

    scored.sort(reverse=True, key=lambda x: x[0])
    _, best_val_acc, best_name, best_est = scored[0]
    best_val_f1 = scored[0][0]

    X_tv = pd.concat([train_df["headline_lowercase"], val_df["headline_lowercase"]], ignore_index=True).fillna("").astype(str).tolist()
    y_tv = pd.concat([train_df["label"], val_df["label"]], ignore_index=True).astype(int).to_numpy()
    final_est = clone(best_est)
    final_est.fit(X_tv, y_tv)
    y_pred_test = final_est.predict(X_test)
    tm = _metrics(y_test, y_pred_test)

    out_path = MODELS_DIR / "word_char_svm_best.joblib"
    joblib.dump(final_est, out_path)

    return FamilyResult(
        family="tfidf_word_char_svm",
        best_variant=best_name,
        val_accuracy=float(best_val_acc),
        val_f1_macro=float(best_val_f1),
        test_accuracy=tm["accuracy"],
        test_f1_macro=tm["f1_macro"],
        details={"saved_model": str(out_path), "num_candidates": len(candidates)},
    )


def run_fasttext_supervised(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> FamilyResult:
    import fasttext

    y_val = val_df["label"].astype(int).to_numpy()
    y_test = test_df["label"].astype(int).to_numpy()

    def write_ft_file(path: Path, texts: list[str], labels: np.ndarray) -> None:
        lines = []
        for t, y in zip(texts, labels):
            lines.append(f"__label__{int(y)} {_clean_for_fasttext(t)}")
        path.write_text("\n".join(lines), encoding="utf-8")

    train_texts = train_df["headline_minimal"].fillna("").astype(str).tolist()
    val_texts = val_df["headline_minimal"].fillna("").astype(str).tolist()
    test_texts = test_df["headline_minimal"].fillna("").astype(str).tolist()
    y_train = train_df["label"].astype(int).to_numpy()

    with tempfile.TemporaryDirectory(prefix="fasttext_run_") as tmp:
        tmpdir = Path(tmp)
        train_file = tmpdir / "train.txt"
        trainval_file = tmpdir / "trainval.txt"
        write_ft_file(train_file, train_texts, y_train)

        configs = [
            {"wordNgrams": 2, "lr": 0.35, "epoch": 18, "dim": 120},
            {"wordNgrams": 3, "lr": 0.30, "epoch": 20, "dim": 120},
        ]
        best: tuple[float, float, dict[str, Any], Any] | None = None
        for cfg in configs:
            model = fasttext.train_supervised(
                input=str(train_file),
                loss="softmax",
                thread=max(1, os.cpu_count() or 1),
                verbose=0,
                **cfg,
            )
            pred_labels, _ = model.predict([_clean_for_fasttext(t) for t in val_texts], k=1)
            y_pred = np.array([int(lbl[0].replace("__label__", "")) for lbl in pred_labels], dtype=int)
            m = _metrics(y_val, y_pred)
            candidate = (m["f1_macro"], m["accuracy"], cfg, model)
            if best is None or candidate[0] > best[0]:
                best = candidate

        assert best is not None
        best_f1, best_acc, best_cfg, _ = best

        trainval_texts = pd.concat([train_df["headline_minimal"], val_df["headline_minimal"]], ignore_index=True).fillna("").astype(str).tolist()
        y_trainval = pd.concat([train_df["label"], val_df["label"]], ignore_index=True).astype(int).to_numpy()
        write_ft_file(trainval_file, trainval_texts, y_trainval)

        final_model = fasttext.train_supervised(
            input=str(trainval_file),
            loss="softmax",
            thread=max(1, os.cpu_count() or 1),
            verbose=0,
            **best_cfg,
        )
        pred_test, _ = final_model.predict([_clean_for_fasttext(t) for t in test_texts], k=1)
        y_test_pred = np.array([int(lbl[0].replace("__label__", "")) for lbl in pred_test], dtype=int)
        tm = _metrics(y_test, y_test_pred)

        model_path = MODELS_DIR / "fasttext_supervised_best.bin"
        final_model.save_model(str(model_path))

    return FamilyResult(
        family="fasttext_supervised",
        best_variant=f"wordNgrams={best_cfg['wordNgrams']}_dim={best_cfg['dim']}_epoch={best_cfg['epoch']}",
        val_accuracy=float(best_acc),
        val_f1_macro=float(best_f1),
        test_accuracy=tm["accuracy"],
        test_f1_macro=tm["f1_macro"],
        details={"saved_model": str(model_path), "best_cfg": best_cfg},
    )


def _tokenize(texts: list[str]) -> list[list[str]]:
    return [re.findall(r"\b\w+\b", str(t).lower()) for t in texts]


def _average_vectors(tokenized_texts: list[list[str]], keyed_vectors) -> np.ndarray:
    dim = keyed_vectors.vector_size
    out = np.zeros((len(tokenized_texts), dim), dtype=np.float32)
    for i, toks in enumerate(tokenized_texts):
        vecs = [keyed_vectors[w] for w in toks if w in keyed_vectors]
        if vecs:
            out[i] = np.mean(vecs, axis=0)
    return out


def run_static_embedding_models(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> FamilyResult:
    from gensim.models import FastText as GensimFastText
    from gensim.models import Word2Vec

    y_train = train_df["label"].astype(int).to_numpy()
    y_val = val_df["label"].astype(int).to_numpy()
    y_test = test_df["label"].astype(int).to_numpy()
    tr_texts = train_df["headline_lowercase"].fillna("").astype(str).tolist()
    va_texts = val_df["headline_lowercase"].fillna("").astype(str).tolist()
    te_texts = test_df["headline_lowercase"].fillna("").astype(str).tolist()

    tr_tok = _tokenize(tr_texts)
    va_tok = _tokenize(va_texts)
    te_tok = _tokenize(te_texts)

    w2v = Word2Vec(sentences=tr_tok, vector_size=200, window=5, min_count=2, workers=max(1, (os.cpu_count() or 2) - 1), epochs=12)
    ft = GensimFastText(sentences=tr_tok, vector_size=200, window=5, min_count=2, workers=max(1, (os.cpu_count() or 2) - 1), epochs=12)

    emb_sources = {
        "word2vec": w2v.wv,
        "fasttext": ft.wv,
    }
    clf_factories: dict[str, Any] = {
        "xgboost": lambda: XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        ),
        "mlp": lambda: MLPClassifier(
            hidden_layer_sizes=(256, 64),
            activation="relu",
            alpha=1e-4,
            max_iter=40,
            early_stopping=True,
            random_state=42,
        ),
    }

    best: tuple[float, float, str, str, Any] | None = None
    for emb_name, kv in emb_sources.items():
        Xtr = _average_vectors(tr_tok, kv)
        Xva = _average_vectors(va_tok, kv)
        for clf_name, clf_fn in clf_factories.items():
            clf = clf_fn()
            clf.fit(Xtr, y_train)
            pred = clf.predict(Xva)
            m = _metrics(y_val, pred)
            candidate = (m["f1_macro"], m["accuracy"], emb_name, clf_name, clf)
            if best is None or candidate[0] > best[0]:
                best = candidate

    assert best is not None
    best_f1, best_acc, best_emb, best_clf, _ = best

    tv_texts = pd.concat([train_df["headline_lowercase"], val_df["headline_lowercase"]], ignore_index=True).fillna("").astype(str).tolist()
    y_tv = pd.concat([train_df["label"], val_df["label"]], ignore_index=True).astype(int).to_numpy()
    tv_tok = _tokenize(tv_texts)

    if best_emb == "word2vec":
        emb_model = Word2Vec(sentences=tv_tok, vector_size=200, window=5, min_count=2, workers=max(1, (os.cpu_count() or 2) - 1), epochs=12)
    else:
        emb_model = GensimFastText(sentences=tv_tok, vector_size=200, window=5, min_count=2, workers=max(1, (os.cpu_count() or 2) - 1), epochs=12)
    kv = emb_model.wv
    X_tv = _average_vectors(tv_tok, kv)
    X_te = _average_vectors(te_tok, kv)

    final_clf = clf_factories[best_clf]()
    final_clf.fit(X_tv, y_tv)
    y_test_pred = final_clf.predict(X_te)
    tm = _metrics(y_test, y_test_pred)

    clf_path = MODELS_DIR / "static_embedding_best_classifier.joblib"
    emb_path = MODELS_DIR / "static_embedding_best_vectors.kv"
    joblib.dump(final_clf, clf_path)
    kv.save(str(emb_path))

    return FamilyResult(
        family="static_embedding_avg",
        best_variant=f"{best_emb}+{best_clf}",
        val_accuracy=float(best_acc),
        val_f1_macro=float(best_f1),
        test_accuracy=tm["accuracy"],
        test_f1_macro=tm["f1_macro"],
        details={"saved_classifier": str(clf_path), "saved_vectors": str(emb_path)},
    )


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    train_df, val_df, test_df = _load_splits()

    runs: list[FamilyResult] = []
    print("Running family 1/3: TF-IDF word+char + LinearSVC", flush=True)
    runs.append(run_word_char_svm(train_df, val_df, test_df))
    print("Running family 2/3: fastText supervised", flush=True)
    runs.append(run_fasttext_supervised(train_df, val_df, test_df))
    print("Running family 3/3: static averaged embeddings + XGBoost/MLP", flush=True)
    runs.append(run_static_embedding_models(train_df, val_df, test_df))

    summary_df = pd.DataFrame([
        {
            "family": r.family,
            "best_variant": r.best_variant,
            "val_accuracy": r.val_accuracy,
            "val_f1_macro": r.val_f1_macro,
            "test_accuracy": r.test_accuracy,
            "test_f1_macro": r.test_f1_macro,
        }
        for r in runs
    ]).sort_values("test_f1_macro", ascending=False)
    csv_path = REPORTS_DIR / "requested_model_comparison.csv"
    json_path = REPORTS_DIR / "requested_model_comparison.json"
    summary_df.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(
            {
                "results": [r.__dict__ for r in runs],
                "leaderboard": summary_df.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary_df.to_string(index=False))
    print(f"\nSaved comparison files:\n- {csv_path}\n- {json_path}")


if __name__ == "__main__":
    main()
