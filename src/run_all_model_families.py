#!/usr/bin/env python3
"""
Run every standalone model family script under src/models and record status.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SCRIPTS = [
    "src/models/tfidf_logreg.py",
    "src/models/tfidf_svm.py",
    "src/models/tfidf_nb.py",
    "src/models/char_ngram.py",
    "src/models/stylometric.py",
    "src/models/hybrid.py",
    "src/models/voting_ensemble.py",
    "src/models/stacking_ensemble.py",
    "src/models/sentence_embedding.py",
]


def main() -> None:
    results: list[dict] = []
    for script in MODEL_SCRIPTS:
        t0 = time.perf_counter()
        print(f"\n=== Running {script} ===", flush=True)
        proc = subprocess.run([sys.executable, script], cwd=PROJECT_ROOT)
        elapsed = time.perf_counter() - t0
        results.append(
            {
                "script": script,
                "exit_code": proc.returncode,
                "elapsed_seconds": round(elapsed, 2),
            }
        )
        print(
            f"=== Finished {script} | exit={proc.returncode} | elapsed={elapsed:.1f}s ===",
            flush=True,
        )

    out = {
        "total": len(results),
        "success": sum(1 for r in results if r["exit_code"] == 0),
        "failed": sum(1 for r in results if r["exit_code"] != 0),
        "results": results,
    }
    (REPORTS_DIR / "model_family_run_summary.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    print("\nWrote reports/model_family_run_summary.json", flush=True)

    if out["failed"] > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
