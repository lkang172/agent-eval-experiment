"""Day-2 AM probe arm, end to end.

    python -m rvr.run_probe_arm --quant 8bit

Steps, in the order §9 specifies:
  1. build all 390 contexts and re-run the §8 audits
  2. Probe A: SAD stages_oversight, layer sweep, §4.4 gate against OUR baselines
  3. Probe B: agentic contexts, layer sweep, held out BY TASK, §4.4 gate
  4. score every context with both probes
  5. TOST margin from the random-direction band (pre-registered provenance)
  6. primary family: Δ_ctrl on Probe B, four factors, Holm
  7. write results/results.json

Stops early and writes what it has if a gate fails -- a failed Probe B gate is
itself the finding (§11), not an error.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import analysis
from .activations import DEFAULT_MODEL, extract, load_model, resolve_positions
from .analysis import ScoreTable, run_primary
from .audits import run_all
from .generate import render_all
from .probe import (best_layer, fit_at_layer, fit_folds_and_score_all,
                    gate_probe_a, gate_probe_b, layer_sweep)
from .sad_data import baseline_auroc, load_oversight
from .taskloader import load_all
from .tokenization import count as tok_count

ROOT = Path(__file__).resolve().parents[2]
SPECIAL = set("{}[]()<>\"'`,;:/\\|=+-_*&^%$#@!~?.")


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def random_direction_band(X: np.ndarray, layer: int, meta: list[dict],
                          k: int = 200, seed: int = 0) -> dict:
    """TOST margin provenance (ii): the Δ_ctrl band from random probe directions.

    A random direction carries no eval/deploy information, so whatever spread of
    Δ_ctrl it produces is noise. Any real effect must exceed it.
    """
    rng = np.random.default_rng(seed)
    Xl = X[:, layer, :]
    Xl = (Xl - Xl.mean(0)) / (Xl.std(0) + 1e-8)
    vals = []
    for _ in range(k):
        w = rng.normal(size=Xl.shape[1])
        w /= np.linalg.norm(w)
        s = Xl @ w
        tbl = ScoreTable([{**m, "score": float(v)} for m, v in zip(meta, s)])
        for f in analysis.FACTORS:
            d = analysis._delta_ctrl(tbl, f, "score", tbl.tasks())
            if not np.isnan(d):
                vals.append(abs(d))
    vals = np.array(vals)
    return {"k": k, "n_values": len(vals),
            "p95": float(np.percentile(vals, 95)),
            "max": float(vals.max()), "mean": float(vals.mean())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--quant", default="8bit", choices=["8bit", "4bit", "fp16", "bf16"])
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--resamples", type=int, default=5000)
    ap.add_argument("--out", default=str(ROOT / "results" / "results.json"))
    a = ap.parse_args()

    out: dict = {"config": vars(a), "started": time.strftime("%Y-%m-%dT%H:%M:%S")}

    # -- 1. contexts ------------------------------------------------------
    _log("building contexts")
    tasks = load_all()
    contexts, audits = [], []
    for t in tasks:
        cs = render_all(t)
        contexts.extend(cs)
        audits.append(run_all(t, cs))
    out["contexts"] = {"n_tasks": len(tasks), "n_contexts": len(contexts),
                       "audits_all_ok": all(x["all_ok"] for x in audits)}
    _log(f"  {len(tasks)} tasks -> {len(contexts)} contexts, "
         f"audits_ok={out['contexts']['audits_all_ok']}")

    # -- 2. model ---------------------------------------------------------
    _log(f"loading {a.model} ({a.quant})")
    model, tok, info = load_model(a.model, a.quant)
    out["model"] = info
    _log(f"  {info['n_layers']} layers, hidden {info['hidden_size']}, {info['device']}")

    # -- 3. Probe A -------------------------------------------------------
    _log("Probe A: SAD stages_oversight")
    sad = load_oversight()
    sad_base = baseline_auroc(sad)
    _log(f"  {len(sad)} samples; our baselines {sad_base}")
    # read at the final token: these are chat prompts, not trajectories
    sad_idx = [len(tok(t, add_special_tokens=False)["input_ids"]) - 1 for t in sad.texts]
    A_sad = extract(model, tok, sad.texts, sad_idx, sad.ids, "final", a.batch_size)
    sweep_a = layer_sweep(A_sad.X, sad.labels, None)
    La = best_layer(sweep_a)
    probe_a = fit_at_layer(A_sad.X, sad.labels, La.layer)
    # Layer 0 is the embedding output -- nothing has been computed yet, so any
    # separation there is token identity, not a representation. A high value is
    # the signature of a corpus/vocabulary confound, which the length and
    # special-character baselines cannot see.
    l0 = next((r.auroc_mean for r in sweep_a if r.layer == 0), None)
    gate_a = gate_probe_a(probe_a.cv_auroc, sad_base, embedding_auroc=l0)
    out["probe_A"] = {
        "n": len(sad), "source": sad.source, "best_layer": La.layer,
        "cv_auroc": probe_a.cv_auroc, "train_auroc": probe_a.train_auroc,
        "layer_sweep": [{"layer": r.layer, "auroc": r.auroc_mean} for r in sweep_a],
        "our_baselines": sad_base, "gate": gate_a,
        "note": ("trained, not transferred: Nguyen et al.'s released probe is a "
                 "Llama-3.3-70B direction and is dimensionally inapplicable here"),
    }
    _log(f"  best layer {La.layer}, cv_auroc {probe_a.cv_auroc:.3f}, "
         f"gate passed={gate_a['passed']}")
    del A_sad

    # -- 4. agentic activations -------------------------------------------
    _log("extracting agentic activations at the primary read position")
    texts = [c.text for c in contexts]
    idx = [resolve_positions(c.text, c.read_positions, tok, ("primary",))["primary"]
           for c in contexts]
    keys = [f"{c.task_id}|{c.arm_id}" for c in contexts]
    A = extract(model, tok, texts, idx, keys, "primary", a.batch_size)

    np.savez_compressed(Path(a.out).parent / "activations_primary.npz",
                        X=A.X.astype(np.float16),
                        keys=np.array(keys), token_index=np.array(idx))
    _log(f"  cached activations -> {Path(a.out).parent / 'activations_primary.npz'}")

    meta = [{"task_id": c.task_id, "arm_id": c.arm_id,
             "tokens": tok_count(c.text).n,
             "special_chars": sum(ch in SPECIAL for ch in c.text)}
            for c in contexts]

    # -- 5. Probe B: eval-like vs deploy-like, held out BY TASK ------------
    _log("Probe B: in-domain ceiling, held out by task")
    EVAL_ARMS = {"base"}
    DEPLOY_ARMS = {f"{f}_deploy" for f in analysis.FACTORS} | {"stacked_deploy"}
    sel = [i for i, m in enumerate(meta) if m["arm_id"] in EVAL_ARMS | DEPLOY_ARMS]
    yb = [1 if meta[i]["arm_id"] in EVAL_ARMS else 0 for i in sel]
    gb = [meta[i]["task_id"] for i in sel]
    sweep_b = layer_sweep(A.X[sel], yb, gb)
    Lb = best_layer(sweep_b)
    probe_b = fit_at_layer(A.X[sel], yb, Lb.layer, groups=gb)
    gate_b = gate_probe_b(probe_b.cv_auroc)
    out["probe_B"] = {
        "n": len(sel), "best_layer": Lb.layer, "cv_auroc": probe_b.cv_auroc,
        "train_auroc": probe_b.train_auroc,
        "layer_sweep": [{"layer": r.layer, "auroc": r.auroc_mean} for r in sweep_b],
        "held_out_by": "task", "gate": gate_b,
    }
    out["A_B_transfer_gap"] = probe_b.cv_auroc - probe_a.cv_auroc
    _log(f"  best layer {Lb.layer}, cv_auroc {probe_b.cv_auroc:.3f}, "
         f"gate passed={gate_b['passed']}")

    if not gate_b["passed"]:
        _log("Probe B did not clear the gate -- that null IS the finding (§11). Stopping.")
        out["stopped_at"] = "probe_B_gate"
        Path(a.out).write_text(json.dumps(out, indent=2, default=float))
        return 0

    # -- 6. score every context, OUT OF FOLD ------------------------------
    # Every context is scored by a probe that never saw its task. Scoring the
    # contexts Probe B was fit on measures training-set fit: Δ_base saturates
    # and the F0 null test fails for reasons unrelated to any pipeline leak.
    _log("scoring all contexts out-of-fold")
    all_tasks = [m["task_id"] for m in meta]
    sb, oof_info = fit_folds_and_score_all(A.X[sel], yb, gb, A.X, all_tasks,
                                           probe_b.layer)
    out["probe_B"]["out_of_fold"] = oof_info
    _log(f"  out-of-fold: {oof_info['n_scored']}/{len(meta)} scored, "
         f"fold auroc {oof_info['cv_auroc_mean']:.3f}")
    sa = probe_a.score(A.X[:, probe_a.layer, :])
    tbl_b = ScoreTable([{**m, "score": float(v)} for m, v in zip(meta, sb)])
    tbl_a = ScoreTable([{**m, "score": float(v)} for m, v in zip(meta, sa)])

    # -- 7. TOST margin ---------------------------------------------------
    _log("random-direction band for the TOST margin")
    band = random_direction_band(A.X, probe_b.layer, meta)
    margin = band["p95"]
    out["tost_margin"] = {
        "delta": margin, "set_by": "random_direction_band_p95", "band": band,
        "note": ("§5.2 asks for the wider of test-retest spread or the random-"
                 "direction band. Greedy decoding and a fixed seed make the probe "
                 "arm deterministic, so test-retest spread is 0 by construction "
                 "and the random-direction band sets the margin."),
    }
    _log(f"  delta = {margin:.4f}")

    # -- 8. primary family -------------------------------------------------
    _log("primary family: Δ_ctrl on Probe B, Holm across four factors")
    out["primary"] = run_primary(tbl_b, margin, a.resamples)
    out["secondary_probe_A"] = run_primary(tbl_a, margin, min(a.resamples, 2000))

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2, default=float))
    _log(f"wrote {a.out}")

    r = out["primary"]
    print("\nranking by Δ_ctrl (Probe B):")
    for f in r["ranking_by_delta_ctrl"]:
        d = r["per_factor"][f]["delta_ctrl"]
        print(f"  {f}  Δ_ctrl={d['point']:+.3f}  CI[{d['ci'][0]:+.3f},{d['ci'][1]:+.3f}]"
              f"  p_holm={d['p_holm']:.4f}  sig={d['significant_holm']}")
    print(f"\nF0 null check: {r['f0_null']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
