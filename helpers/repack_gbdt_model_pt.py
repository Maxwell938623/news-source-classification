"""
Rebuild submission/gbdt_model/model.pt from models/gbdt_best.joblib.

Training saves joblib from src/gbdt.py where picklable symbols may live under
__main__, gbdt, or models.stylometric. The HF grader imports model.py under a
dynamic name; submission model.py registers sys.modules[\"model\"] and uses a
custom unpickler, but embedding the pipeline with func/__module__ = \"model\"
keeps payloads smallest and avoids relying on reducer quirks across sklearn
versions.
"""
from __future__ import annotations

import importlib.util
import pickle
import sys
from pathlib import Path

import joblib
import torch
from joblib.numpy_pickle import NumpyUnpickler
from sklearn.preprocessing import FunctionTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUBMISSION_MODEL_PY = PROJECT_ROOT / "submission" / "gbdt_model" / "model.py"
DEFAULT_JOBLIB = PROJECT_ROOT / "models" / "gbdt_best.joblib"
OUT_PT = PROJECT_ROOT / "submission" / "gbdt_model" / "model.pt"


def _load_submission_as_model() -> object:
    spec = importlib.util.spec_from_file_location("model", SUBMISSION_MODEL_PY)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["model"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _joblib_load_with_aliases(sm: object, joblib_path: Path):
    """Resolve class/function paths from training into submission `model` module."""
    orig = NumpyUnpickler.find_class

    def find_class(self, module: str, name: str):
        if name == "_dense_array":
            return sm._dense_array
        if name == "SparseWrapper":
            return sm.SparseWrapper
        if name == "StyloTransformer":
            return sm.StyloTransformer
        if module in ("__main__", "gbdt", "src.gbdt", r"src\gbdt"):
            if name == "SparseWrapper":
                return sm.SparseWrapper
            if name == "StyloTransformer":
                return sm.StyloTransformer
            if name == "_dense_array":
                return sm._dense_array
        if module.startswith("models.") or module == "stylometric":
            if name == "StyloTransformer":
                return sm.StyloTransformer
        return orig(self, module, name)

    NumpyUnpickler.find_class = find_class  # type: ignore[method-assign]
    try:
        return joblib.load(joblib_path)
    finally:
        NumpyUnpickler.find_class = orig  # type: ignore[method-assign]


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Embed gbdt_best.joblib into submission gbdt_model/model.pt.")
    p.add_argument("--joblib-path", type=Path, default=DEFAULT_JOBLIB)
    p.add_argument("--out", type=Path, default=OUT_PT)
    args = p.parse_args()

    sm = _load_submission_as_model()
    pipe = _joblib_load_with_aliases(sm, args.joblib_path)
    pipe.named_steps["to_dense"] = FunctionTransformer(
        sm._dense_array,
        accept_sparse=True,
    )
    blob = pickle.dumps(pipe, protocol=pickle.HIGHEST_PROTOCOL)
    cap = getattr(sm, "BLOB_CAP", 20_000_000)
    if len(blob) > cap:
        raise RuntimeError(f"Pickled pipeline is {len(blob)} bytes (cap {cap}). Increase BLOB_CAP in model.py.")
    tens = torch.tensor(list(blob), dtype=torch.uint8)
    size = torch.tensor(len(blob), dtype=torch.int64)
    payload = {"state_dict": {"pipeline_blob": tens, "pipeline_size": size}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)
    print(f"Wrote {args.out} ({len(blob)} byte pickle blob)")


if __name__ == "__main__":
    main()
