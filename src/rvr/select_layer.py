"""Re-select the probe layer under the F0 null constraint (§3.3).

    python -m rvr.select_layer

The first run picked the layer that maximised held-out AUROC. That criterion
actively rewards surface sensitivity: the most token-driven layer wins, and the
probe it produces separates two contexts that differ only in JSON indentation.
The F0 null control caught exactly that.

Layer choice is therefore constrained: among layers whose out-of-fold probe does
NOT separate the two cosmetic-only arms, take the one with the highest AUROC.
Runs entirely on the cached activations -- no GPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import analysis
from .analysis import ScoreTable, f0_null_check, run_primary
from .probe import fit_folds_and_score_all
from .tokenization import count as tok_count

ROOT = Path(__file__).resolve().parents[2]
SPECIAL = set("{}[]()<>\"'`,;:/\\|=+-_*&^%$#@!~?.")

EVAL_ARMS = {"base"}
DEPLOY_ARMS = {f"{f}_deploy" for f in analysis.FACTORS} | {"stacked_deploy"}


def load_cached(path: Path):
    d = np.load(path)
    X = d["X"].astype(np.float32)
    keys = [str(k) for k in d["keys"]]
    tasks = [k.split("|", 1)[0] for k in keys]
    arms = [k.split("|", 1)[1] for k in keys]
    return X, tasks, arms


def build_meta(tasks, arms):
    """Token and special-character counts for the B-len / B-char baselines."""
    from .generate import render_all
    from .taskloader import load_all

    text = {}
    for t in load_all():
        for c in render_all(t):
            text[(c.task_id, c.arm_id)] = c.text
    meta = []
    for ta, ar in zip(tasks, arms):
        s = text[(ta, ar)]
        meta.append({"task_id": ta, "arm_id": ar, "tokens": tok_count(s).n,
                     "special_chars": sum(ch in SPECIAL for ch in s)})
    return meta


def sweep(X, tasks, arms, meta, margin: float, verbose: bool = True) -> list[dict]:
    """For every layer: out-of-fold AUROC, the F0 null result, and Δ_ctrl."""
    sel = [i for i, a in enumerate(arms) if a in EVAL_ARMS | DEPLOY_ARMS]
    y = [1 if arms[i] in EVAL_ARMS else 0 for i in sel]
    g = [tasks[i] for i in sel]

    rows = []
    for L in range(X.shape[1]):
        scores, info = fit_folds_and_score_all(X[sel], y, g, X, tasks, layer=L)
        tbl = ScoreTable([{**m, "score": float(s)} for m, s in zip(meta, scores)])
        f0 = f0_null_check(tbl, margin)
        row = {
            "layer": L,
            "oof_auroc": info["cv_auroc_mean"],
            "f0_auroc": f0["auroc"],
            "f0_deviation": abs(f0["auroc"] - 0.5),
            "f0_passes": f0["passes"],
            "delta_ctrl": {f: analysis._delta_ctrl(tbl, f, "score", tbl.tasks())
                           for f in analysis.FACTORS},
            "delta_base": {f: analysis._delta_base(tbl, f, "score", tbl.tasks())
                           for f in analysis.FACTORS},
        }
        rows.append(row)
        if verbose:
            flag = "ok " if f0["passes"] else "LEAK"
            print(f"  L{L:02d}  auroc={row['oof_auroc']:.3f}  "
                  f"f0={f0['auroc']:.3f} (dev {row['f0_deviation']:.3f}) {flag}")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations", default=str(ROOT / "results" / "activations_primary.npz"))
    ap.add_argument("--results", default=str(ROOT / "results" / "results.json"))
    ap.add_argument("--out", default=str(ROOT / "results" / "layer_selection.json"))
    ap.add_argument("--resamples", type=int, default=5000)
    a = ap.parse_args()

    prev = json.load(open(a.results))
    margin = prev["tost_margin"]["delta"]
    print(f"F0 margin from the pre-registered random-direction band: {margin:.4f}\n")

    X, tasks, arms = load_cached(Path(a.activations))
    meta = build_meta(tasks, arms)
    rows = sweep(X, tasks, arms, meta, margin)

    passing = [r for r in rows if r["f0_passes"]]
    print(f"\nlayers passing the F0 null control: {len(passing)}/{len(rows)}")
    out = {"margin": margin, "n_layers": len(rows), "n_passing_f0": len(passing),
           "per_layer": rows}

    if not passing:
        out["selected"] = None
        out["verdict"] = (
            "NO layer passes the F0 null control. The probe separates two contexts "
            "differing only in JSON indentation at every depth, so no layer yields "
            "an interpretable attribution. Report the representational arm as "
            "inconclusive rather than as a null.")
        print("\n" + out["verdict"])
    else:
        best = max(passing, key=lambda r: r["oof_auroc"])
        out["selected"] = best["layer"]
        out["selected_auroc"] = best["oof_auroc"]
        out["verdict"] = (
            f"layer {best['layer']} selected: highest out-of-fold AUROC "
            f"({best['oof_auroc']:.3f}) among the {len(passing)} layers that do not "
            f"separate the cosmetic-only arms")
        print(f"\n{out['verdict']}")

        sel = [i for i, ar in enumerate(arms) if ar in EVAL_ARMS | DEPLOY_ARMS]
        y = [1 if arms[i] in EVAL_ARMS else 0 for i in sel]
        g = [tasks[i] for i in sel]
        scores, _ = fit_folds_and_score_all(X[sel], y, g, X, tasks, layer=best["layer"])
        tbl = ScoreTable([{**m, "score": float(s)} for m, s in zip(meta, scores)])
        out["primary_at_selected_layer"] = run_primary(tbl, margin, a.resamples)

    Path(a.out).write_text(json.dumps(out, indent=2, default=float))
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
