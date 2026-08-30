"""Primary-endpoint machinery (§5), following configs/preregistration.json.

Δ_base = AUROC(base vs deploy)
Δ_ctrl = AUROC(base vs deploy) − AUROC(base vs control)

Both AUROCs are computed over the same shared eval arm (`base`), across tasks.
All intervals come from a cluster bootstrap over TASKS: variants of one task
share content, so resampling contexts would give falsely narrow intervals (§5.2).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

FACTORS = ("F1", "F3", "F4", "F6")


# --------------------------------------------------------------------------
# AUROC
# --------------------------------------------------------------------------


def auroc(neg: Sequence[float], pos: Sequence[float]) -> float:
    """Rank-based AUROC with tie correction (Mann-Whitney U / n_neg n_pos)."""
    neg, pos = np.asarray(neg, float), np.asarray(pos, float)
    if len(neg) == 0 or len(pos) == 0:
        return float("nan")
    allv = np.concatenate([neg, pos])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv), float)
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks within tied groups
    sortedv = allv[order]
    i = 0
    while i < len(sortedv):
        j = i
        while j + 1 < len(sortedv) and sortedv[j + 1] == sortedv[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    r_pos = ranks[len(neg):].sum()
    u = r_pos - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(neg) * len(pos)))


# --------------------------------------------------------------------------
# score table
# --------------------------------------------------------------------------


@dataclass
class ScoreTable:
    """One row per (task, arm). `score` is the probe read-out; `tokens` and
    `special_chars` back the B-len / B-char baselines (§3.3)."""

    rows: list[dict] = field(default_factory=list)

    def tasks(self) -> list[str]:
        return sorted({r["task_id"] for r in self.rows})

    def get(self, task_id: str, arm_id: str, key: str = "score") -> float | None:
        for r in self.rows:
            if r["task_id"] == task_id and r["arm_id"] == arm_id:
                return r[key]
        return None

    def paired(self, arm_a: str, arm_b: str, key: str, tasks: Iterable[str] | None = None
               ) -> tuple[list[float], list[float]]:
        out_a, out_b = [], []
        for t in (tasks if tasks is not None else self.tasks()):
            a, b = self.get(t, arm_a, key), self.get(t, arm_b, key)
            if a is not None and b is not None:
                out_a.append(a)
                out_b.append(b)
        return out_a, out_b


# --------------------------------------------------------------------------
# endpoints
# --------------------------------------------------------------------------


# POLARITY. The eval-like arm is the POSITIVE class, matching how Probe B is
# labelled (base = 1). auroc(neg=deploy, pos=base) then reads as "how separable
# is the eval arm from the deploy arm, in the expected direction": 0.5 is no
# separation, 1.0 is perfect. Passing base as the negative class instead returns
# 1 - the intended value, which silently inverts every ranking.
def _delta_base(tbl: ScoreTable, factor: str, key: str, tasks) -> float:
    base, dep = tbl.paired("base", f"{factor}_deploy", key, tasks)
    return auroc(dep, base)


def _delta_ctrl(tbl: ScoreTable, factor: str, key: str, tasks) -> float:
    base, dep = tbl.paired("base", f"{factor}_deploy", key, tasks)
    base_c, ctl = tbl.paired("base", f"{factor}_ctrl", key, tasks)
    return auroc(dep, base) - auroc(ctl, base_c)


# --------------------------------------------------------------------------
# cluster bootstrap (§5.2)
# --------------------------------------------------------------------------


@dataclass
class BootResult:
    point: float
    ci_low: float
    ci_high: float
    p_two_sided: float
    n_tasks: int
    resamples: int
    dist: np.ndarray = field(repr=False, default=None)


def cluster_bootstrap(tbl: ScoreTable, stat: Callable[[ScoreTable, list[str]], float],
                      n_resamples: int = 5000, alpha: float = 0.05,
                      rng_seed: int = 0) -> BootResult:
    """Resample TASKS with replacement, keeping all variants of each task."""
    tasks = tbl.tasks()
    rng = np.random.default_rng(rng_seed)
    point = stat(tbl, tasks)
    draws = np.empty(n_resamples)
    for i in range(n_resamples):
        samp = [tasks[j] for j in rng.integers(0, len(tasks), len(tasks))]
        draws[i] = stat(tbl, samp)
    draws = draws[~np.isnan(draws)]
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    # two-sided bootstrap p-value against H0: statistic == 0
    p = 2 * min((draws <= 0).mean(), (draws >= 0).mean())
    return BootResult(point, float(lo), float(hi), float(min(1.0, p)),
                      len(tasks), len(draws), draws)


def holm(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Holm-Bonferroni over the declared primary family (§5.1)."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out, prev, rejected_so_far = {}, 0.0, True
    for i, (k, p) in enumerate(items):
        adj = min(1.0, max(prev, p * (m - i)))
        prev = adj
        rejected_so_far = rejected_so_far and adj <= alpha
        out[k] = {"p_raw": p, "p_holm": adj, "significant": rejected_so_far}
    return out


def tost(boot: BootResult, margin: float, alpha: float = 0.05) -> dict:
    """Equivalence to zero within ±margin, from the bootstrap distribution.

    The margin must come from the pre-registered provenance (§5.2): the wider of
    the probe's test-retest spread or the random-direction AUROC band.
    """
    d = boot.dist
    p_lower = float((d >= margin).mean())     # H0: theta >= +margin
    p_upper = float((d <= -margin).mean())    # H0: theta <= -margin
    p = max(p_lower, p_upper)
    return {"margin": margin, "p_tost": p, "equivalent": p < alpha,
            "point": boot.point, "ci": [boot.ci_low, boot.ci_high]}


# --------------------------------------------------------------------------
# baselines and the length-confound bound
# --------------------------------------------------------------------------


def baseline_bounds(tbl: ScoreTable) -> dict[str, dict]:
    """B-len and B-char AUROC for every factor contrast (§3.3).

    This is the per-contrast replacement for §8.1's blanket 2% token-matching
    rule, which the factor definitions make unsatisfiable: instead of asserting
    arms are length-matched, each row states how much of its own effect a
    length-only or special-character-only classifier reproduces.
    """
    out = {}
    for f in FACTORS:
        row = {}
        for key, label in (("tokens", "B_len"), ("special_chars", "B_char")):
            db = _delta_base(tbl, f, key, tbl.tasks())
            row[label] = {
                # signed, eval-positive, same convention as the probe endpoints
                "delta_base": db,
                # direction-agnostic: a baseline that separates the arms is
                # equally damaging whichever way it points. Deploy arms are
                # longer than eval arms, so B_len's signed value sits near 0
                # while its separability sits near 1.
                "delta_base_separability": max(db, 1.0 - db),
                "delta_ctrl": _delta_ctrl(tbl, f, key, tbl.tasks()),
            }
        out[f] = row
    return out


def f0_null_check(tbl: ScoreTable, margin: float) -> dict:
    """The probe must not separate the two cosmetic-only arms (§3.3).
    A violation voids every downstream number."""
    a, b = tbl.paired("F0_null_a", "F0_null_b", "score")
    val = auroc(a, b)
    return {"auroc": val, "margin": margin,
            "passes": abs(val - 0.5) <= margin,
            "note": "if this fails the pipeline leaks and all downstream results are void"}


# --------------------------------------------------------------------------
# top level
# --------------------------------------------------------------------------


def run_primary(tbl: ScoreTable, tost_margin: float, n_resamples: int = 5000,
                alpha: float = 0.05) -> dict:
    """The declared primary family: Δ_ctrl on Probe B, four factors, Holm."""
    per_factor, praw = {}, {}
    for f in FACTORS:
        b_ctrl = cluster_bootstrap(tbl, lambda t, ts, f=f: _delta_ctrl(t, f, "score", ts),
                                   n_resamples, alpha)
        b_base = cluster_bootstrap(tbl, lambda t, ts, f=f: _delta_base(t, f, "score", ts),
                                   n_resamples, alpha)
        per_factor[f] = {
            "delta_ctrl": {"point": b_ctrl.point, "ci": [b_ctrl.ci_low, b_ctrl.ci_high],
                           "p_raw": b_ctrl.p_two_sided},
            "delta_base": {"point": b_base.point, "ci": [b_base.ci_low, b_base.ci_high],
                           "p_raw": b_base.p_two_sided},
            "tost_delta_ctrl": tost(b_ctrl, tost_margin, alpha),
        }
        praw[f] = b_ctrl.p_two_sided

    corrected = holm(praw, alpha)
    for f, c in corrected.items():
        per_factor[f]["delta_ctrl"]["p_holm"] = c["p_holm"]
        per_factor[f]["delta_ctrl"]["significant_holm"] = c["significant"]

    ranking = sorted(FACTORS, key=lambda f: -per_factor[f]["delta_ctrl"]["point"])
    return {
        "primary_family": {"endpoint": "delta_ctrl", "probe": "B", "factors": list(FACTORS),
                           "correction": "holm-bonferroni", "alpha": alpha},
        "n_tasks": len(tbl.tasks()),
        "resamples": n_resamples,
        "per_factor": per_factor,
        "ranking_by_delta_ctrl": ranking,
        "baselines": baseline_bounds(tbl),
        "f0_null": f0_null_check(tbl, tost_margin),
    }
