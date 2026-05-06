#!/usr/bin/env python3
from __future__ import annotations
import importlib.util
import json
import sys
import types
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "helpers" / "url_with_headlines.csv"
PREP_PATH = ROOT / "submission" / "preprocess.py"
OUT_CSV = ROOT / "reports" / "joblib_eval_url_with_headlines.csv"
OUT_SUMMARY = ROOT / "reports" / "joblib_eval_url_with_headlines_summary.json"


def load_submission_data():
    spec = importlib.util.spec_from_file_location("sub_pre", PREP_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.prepare_data(str(CSV_PATH))


def _extract_features(text: str) -> list[float]:
    words = text.split()
    n_words = len(words)
    n_chars = len(text)
    n_alpha = sum(1 for c in text if c.isalpha()) or 1
    avg_word_len = float(np.mean([len(w) for w in words])) if words else 0.0
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
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([_extract_features(t) for t in X], dtype=np.float64)


class SparseWrapper(BaseEstimator, TransformerMixin):
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


def register_pickle_compat() -> None:
    models_pkg = types.ModuleType("models")
    sys.modules.setdefault("models", models_pkg)
    st_mod = types.ModuleType("models.stylometric")
    st_mod.StyloTransformer = StyloTransformer
    sys.modules.setdefault("models.stylometric", st_mod)
    hy_mod = types.ModuleType("models.hybrid")
    hy_mod.SparseWrapper = SparseWrapper
    sys.modules.setdefault("models.hybrid", hy_mod)
    st_mod2 = types.ModuleType("stylometric")
    st_mod2.StyloTransformer = StyloTransformer
    sys.modules.setdefault("stylometric", st_mod2)
    hy_mod2 = types.ModuleType("hybrid")
    hy_mod2.SparseWrapper = SparseWrapper
    sys.modules.setdefault("hybrid", hy_mod2)


def as_labels(pred) -> list[str]:
    out = []
    for p in pred:
        try:
            out.append("FoxNews" if int(p) == 0 else "NBC")
        except Exception:
            out.append(str(p))
    return out


def main() -> None:
    register_pickle_compat()
    X, y = load_submission_data()
    joblibs = sorted(
        p for p in ROOT.rglob("*.joblib")
        if ".git" not in p.parts and "__pycache__" not in p.parts
    )
    rows: list[dict] = []
    for p in joblibs:
        rel = str(p.relative_to(ROOT)).replace("\\", "/")
        try:
            model = joblib.load(p)
            pred = model.predict(X)
            pred_lbl = as_labels(pred)
            rows.append(
                {
                    "model_path": rel,
                    "status": "ok",
                    "accuracy": float(accuracy_score(y, pred_lbl)),
                    "error": "",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "model_path": rel,
                    "status": "error",
                    "accuracy": None,
                    "error": str(exc)[:400],
                }
            )
    df = pd.DataFrame(rows).sort_values(
        by=["status", "accuracy"], ascending=[True, False], na_position="last"
    )
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    ok_df = df[df["status"] == "ok"]
    summary = {
        "total": len(df),
        "ok": int((df["status"] == "ok").sum()),
        "error": int((df["status"] == "error").sum()),
        "best_ok": ok_df.head(1).to_dict(orient="records"),
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
