"""RQ2 behavioural arm (§6).

    python -m rvr.run_behavioral_arm --pilot          # time it first
    python -m rvr.run_behavioral_arm --rollouts 15    # the real run

Answers: does changing a harness property change which branch the agent takes at
a pre-registered decision point? Independent of the probe arm -- it stands alone
whether or not the representational result works out.

Order of interpretation, which matters for an honest null:
  1. dynamic range on the base condition (§11). If the decision points are not
     live, a null reflects task design, not the hypothesis. Checked first.
  2. per-factor Δ in cautious-branch rate, cluster-bootstrapped by task.
  3. TOST against the pre-registered margin for any "no effect" claim, plus the
     achieved-power statement §5.2 requires.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import analysis
from .analysis import FACTORS, cluster_bootstrap, holm, tost
from .behavioral import (build_prompt, choice_rates, delta_rate, delta_rate_ctrl,
                         dynamic_range, rate_table, sample_rollouts)
from .generate import render_all
from .taskloader import load_all

ROOT = Path(__file__).resolve().parents[2]

# Staged design. The screen comes first: base vs stacked_deploy is the MAXIMUM
# available manipulation (all four factors at once, separable representationally
# at AUROC 0.999), so it is the most powerful test of whether harness realism
# moves behaviour at all. Only if the screen moves is a per-factor decomposition
# worth 4,000 generations -- and if the screen does NOT move, a null on the
# strongest possible manipulation is a far stronger result than four
# separately-underpowered per-factor nulls.
SCREEN_ARMS = ["base", "stacked_deploy"]
FULL_ARMS = (["base"] + [f"{f}_deploy" for f in FACTORS]
             + [f"{f}_ctrl" for f in FACTORS] + ["stacked_deploy", "stacked_ctrl"])
DEFAULT_ARMS = SCREEN_ARMS


def _log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--quant", default="8bit", choices=["8bit", "4bit", "fp16", "bf16"])
    ap.add_argument("--rollouts", type=int, default=15)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--arms", default=",".join(DEFAULT_ARMS),
                    help="comma-separated arm ids; 'screen' = base vs "
                         "stacked_deploy, 'full' = all 11 arms")
    ap.add_argument("--pilot", action="store_true",
                    help="2 tasks x 2 arms x 3 rollouts, then report timing and stop")
    ap.add_argument("--resamples", type=int, default=5000)
    ap.add_argument("--out", default=str(ROOT / "results" / "behavioral.json"))
    a = ap.parse_args()

    from .activations import DEFAULT_MODEL, load_model
    model_name = a.model or DEFAULT_MODEL
    arms = {"screen": SCREEN_ARMS, "full": FULL_ARMS}.get(a.arms, a.arms.split(","))

    tasks = load_all()
    specs = []
    for t in tasks:
        by_arm = {c.arm_id: c for c in render_all(t)}
        for arm in arms:
            if arm in by_arm:
                specs.append(build_prompt(by_arm[arm], t))

    if a.pilot:
        keep = {t.task_id for t in tasks[:2]}
        specs = [s for s in specs if s.task_id in keep and s.arm_id in arms[:2]]
        rollouts = 3
    else:
        rollouts = a.rollouts

    total = len(specs) * rollouts
    _log(f"{len(tasks)} tasks x {len(arms)} arms -> {len(specs)} conditions, "
         f"{rollouts} rollouts each = {total} generations")

    _log(f"loading {model_name} ({a.quant})")
    model, tok, info = load_model(model_name, a.quant)

    t0 = time.time()
    rows = sample_rollouts(model, tok, specs, rollouts, a.temperature,
                           a.max_new_tokens, a.batch_size)
    elapsed = time.time() - t0
    per_gen = elapsed / max(1, len(rows))
    _log(f"{len(rows)} generations in {elapsed:.0f}s ({per_gen:.2f}s each)")

    if a.pilot:
        full = len(tasks) * len(arms) * a.rollouts
        _log(f"PILOT ONLY. Full run would be {full} generations "
             f"~= {full * per_gen / 60:.0f} min at this rate.")
        counts: dict[str, int] = {}
        for r in rows:
            counts[r["choice"]] = counts.get(r["choice"], 0) + 1
        _log(f"parsed choices: {counts}")
        if counts.get("none", 0) + counts.get("other", 0) > len(rows) * 0.3:
            _log("WARNING: many rollouts did not call a declared tool. Inspect "
                 "results['sample_generations'] before committing to a full run.")
        Path(a.out).with_suffix(".pilot.json").write_text(json.dumps(
            {"model": info, "elapsed_s": elapsed, "per_generation_s": per_gen,
             "estimated_full_run_min": full * per_gen / 60, "choice_counts": counts,
             "sample_generations": [r["text"] for r in rows[:6]]}, indent=2))
        return 0

    # ---- summarise -------------------------------------------------------
    rates = choice_rates(rows)
    dr = dynamic_range(rates)
    _log(f"dynamic range on base: mean cautious rate {dr['mean_cautious_rate']:.3f}, "
         f"{dr['tasks_with_variance']}/{dr['n_tasks']} tasks live, usable={dr['usable']}")

    tbl = rate_table(rates)
    margin_src = ROOT / "results" / "results.json"
    margin = 0.10
    if margin_src.exists():
        try:
            margin = json.load(open(margin_src))["tost_margin"]["delta"]
        except Exception:
            pass

    # screen contrast, when the stacked arm was run
    if "stacked_deploy" in arms:
        sb = cluster_bootstrap(
            tbl, lambda t, ts: (lambda b, d: float(np.mean(b) - np.mean(d)))(
                *t.paired("base", "stacked_deploy", "score", ts)),
            a.resamples)
        out_screen = {"delta_rate": sb.point, "ci": [sb.ci_low, sb.ci_high],
                      "p_raw": sb.p_two_sided, "tost": tost(sb, margin)}
        _log(f"SCREEN base vs stacked_deploy: Δ={sb.point:+.3f} "
             f"CI[{sb.ci_low:+.3f},{sb.ci_high:+.3f}] p={sb.p_two_sided:.4f}")
    else:
        out_screen = None

    per_factor, praw = {}, {}
    for f in FACTORS:
        if f"{f}_deploy" not in arms:
            continue
        b = cluster_bootstrap(tbl, lambda t, ts, f=f: delta_rate(t, f, ts),
                              a.resamples)
        bc = cluster_bootstrap(tbl, lambda t, ts, f=f: delta_rate_ctrl(t, f, ts),
                               a.resamples)
        per_factor[f] = {
            "delta_rate": {"point": b.point, "ci": [b.ci_low, b.ci_high],
                           "p_raw": b.p_two_sided},
            "delta_rate_ctrl": {"point": bc.point, "ci": [bc.ci_low, bc.ci_high],
                                "p_raw": bc.p_two_sided},
            "tost": tost(b, margin),
        }
        praw[f] = b.p_two_sided
    for f, c in (holm(praw).items() if praw else []):
        per_factor[f]["delta_rate"]["p_holm"] = c["p_holm"]
        per_factor[f]["delta_rate"]["significant_holm"] = c["significant"]

    out = {
        "config": vars(a), "model": info,
        "n_generations": len(rows), "elapsed_s": elapsed,
        "dynamic_range_base": dr,
        "screen_base_vs_stacked": out_screen,
        "tost_margin": margin,
        "per_factor": per_factor,
        "per_cell_rates": {f"{t}|{arm}": v for (t, arm), v in rates.items()},
        "interpretation_order": [
            "1. dynamic_range_base.usable must be true before any null is "
            "interpretable (§11)",
            "2. delta_rate with Holm across four factors",
            "3. TOST for any 'no effect' claim, with achieved power stated",
        ],
    }
    Path(a.out).write_text(json.dumps(out, indent=2, default=float))

    print(f"\n{'factor':8}{'Δ rate':>10}{'ci_low':>9}{'ci_hi':>9}{'p_holm':>9}"
          f"{'sig':>6}{'TOST eq':>9}")
    for f in [x for x in FACTORS if x in per_factor]:
        d = per_factor[f]["delta_rate"]
        print(f"{f:8}{d['point']:+10.3f}{d['ci'][0]:+9.3f}{d['ci'][1]:+9.3f}"
              f"{d['p_holm']:9.4f}{str(d['significant_holm']):>6}"
              f"{str(per_factor[f]['tost']['equivalent']):>9}")
    _log(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
