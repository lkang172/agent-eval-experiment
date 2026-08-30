"""Validate the endpoint machinery against data with known ground truth."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rvr.analysis import ScoreTable, auroc, cluster_bootstrap, holm, run_primary, tost


def test_auroc_known_cases():
    assert auroc([0, 1, 2], [3, 4, 5]) == 1.0           # perfect separation
    assert auroc([3, 4, 5], [0, 1, 2]) == 0.0           # perfectly inverted
    assert auroc([0, 1], [0, 1]) == 0.5                 # identical -> chance
    assert auroc([1, 1, 1], [1, 1, 1]) == 0.5           # all ties -> chance
    # one clean crossing: neg={1,2}, pos={1.5,3} -> 3/4
    assert auroc([1, 2], [1.5, 3]) == 0.75


def _simulate(n_tasks=30, seed=0):
    """Effects are injected with known sizes so the endpoint can be checked.

    F1  large effect, control barely moves          -> Draft A shape
    F3  effect fully reproduced by its control      -> Draft B shape
    F4  moderate effect, control moves a little
    F6  negligible effect                           -> should be TOST-equivalent
    """
    rng = np.random.default_rng(seed)
    spec = {"F1": (2.0, 0.2), "F3": (1.0, 0.95), "F4": (0.8, 0.15), "F6": (0.05, 0.03)}
    rows = []
    for i in range(n_tasks):
        t = f"task_{i:02d}"
        base = rng.normal(0, 1)
        n_tok = int(rng.normal(300, 20))

        def add(arm, score, tokens):
            rows.append({"task_id": t, "arm_id": arm, "score": score,
                         "tokens": tokens, "special_chars": int(tokens * 0.4)})

        add("base", base, n_tok)
        for f, (d_dep, d_ctl) in spec.items():
            add(f"{f}_deploy", base + d_dep + rng.normal(0, .3), int(n_tok * 1.3))
            add(f"{f}_ctrl", base + d_ctl + rng.normal(0, .3), int(n_tok * 1.3))
        add("F0_null_a", base + rng.normal(0, .02), n_tok)
        add("F0_null_b", base + rng.normal(0, .02), n_tok)
    return ScoreTable(rows)


def test_primary_recovers_known_structure():
    tbl = _simulate()
    res = run_primary(tbl, tost_margin=0.10, n_resamples=2000)

    pf = res["per_factor"]
    d = {f: pf[f]["delta_ctrl"]["point"] for f in ("F1", "F3", "F4", "F6")}

    # F1's control barely moves, so nearly all of its effect survives subtraction.
    # Practical range note: delta_ctrl is a difference of AUROCs, and the control
    # contrast rarely falls below 0.5, so the attainable ceiling is ~0.5, not 1.0.
    # With base ~ N(0,1) and d=2.0, theory gives 0.92 - 0.55 ~= 0.36.
    assert d["F1"] > 0.30, d
    assert d["F1"] > d["F4"] > d["F6"] > d["F3"], d
    # F3's control reproduces the effect -> Draft B shape, delta_ctrl near zero.
    assert abs(d["F3"]) < 0.10, d
    # Ranking should put F1 first and F3 last of the two extremes.
    assert res["ranking_by_delta_ctrl"][0] == "F1", res["ranking_by_delta_ctrl"]
    # F1 survives Holm; F3 does not.
    assert pf["F1"]["delta_ctrl"]["significant_holm"] is True
    assert pf["F3"]["delta_ctrl"]["significant_holm"] is False
    # F3 and F6 are equivalent-to-zero at the margin; F1 is not.
    assert pf["F3"]["tost_delta_ctrl"]["equivalent"] is True
    assert pf["F1"]["tost_delta_ctrl"]["equivalent"] is False
    # The cosmetic-only arms must not separate.
    assert res["f0_null"]["passes"] is True
    assert abs(res["f0_null"]["auroc"] - 0.5) < 0.10, res["f0_null"]


def test_blen_bound_flags_length_confound():
    """Arms differ in length by construction, so B-len must show a large
    delta_base -- that is exactly the confound the bound exists to quantify."""
    res = run_primary(_simulate(), tost_margin=0.10, n_resamples=500)
    b = res["baselines"]["F1"]["B_len"]
    assert b["delta_base"] > 0.9, b        # length alone separates base from deploy
    assert abs(b["delta_ctrl"]) < 0.1, b   # but explains none of delta_ctrl


def test_holm_monotone_and_ordered():
    out = holm({"a": 0.001, "b": 0.02, "c": 0.30, "d": 0.9})
    assert out["a"]["p_holm"] <= out["b"]["p_holm"] <= out["c"]["p_holm"] <= out["d"]["p_holm"]
    assert out["a"]["significant"] and not out["c"]["significant"]
    # once one fails, nothing later may be reported significant
    assert not out["d"]["significant"]


def test_bootstrap_clusters_by_task():
    """Cluster bootstrap must be wider than a naive context-level bootstrap;
    §5.2 is explicit that variants sharing task content give falsely narrow CIs."""
    tbl = _simulate(n_tasks=12, seed=3)
    stat = lambda t, ts: auroc(*t.paired("base", "F1_deploy", "score", ts))
    b = cluster_bootstrap(tbl, stat, n_resamples=2000)
    assert b.ci_low < b.point < b.ci_high
    assert b.n_tasks == 12


def test_delta_ctrl_cancels_a_purely_length_driven_effect():
    """Decisive check on the endpoint's design.

    Simulate a probe that reads NOTHING but token length. The base arm is short;
    the deploy and control arms are equally long. delta_base should then be near
    1.0 (length separates base from deploy perfectly) while delta_ctrl collapses
    to ~0, because both contrasts carry the identical length confound and it
    cancels in the subtraction.

    This is why §8.1's unsatisfiable 2% base-vs-deploy matching does not
    invalidate the primary endpoint: what the endpoint actually requires is that
    each CONTROL be length-matched to its own DEPLOY arm, which is achievable.
    """
    rng = np.random.default_rng(7)
    rows = []
    for i in range(30):
        t = f"task_{i:02d}"
        short, long_ = int(rng.normal(300, 15)), int(rng.normal(478, 15))

        def add(arm, tokens):
            # probe score IS the token count: a pure length reader
            rows.append({"task_id": t, "arm_id": arm, "score": float(tokens),
                         "tokens": tokens, "special_chars": int(tokens * 0.4)})

        add("base", short)
        for f in ("F1", "F3", "F4", "F6"):
            add(f"{f}_deploy", long_)
            add(f"{f}_ctrl", long_ + int(rng.normal(0, 4)))   # matched to deploy
        add("F0_null_a", short)
        add("F0_null_b", short)

    res = run_primary(ScoreTable(rows), tost_margin=0.10, n_resamples=1000)
    for f in ("F1", "F3", "F4", "F6"):
        db = res["per_factor"][f]["delta_base"]["point"]
        dc = res["per_factor"][f]["delta_ctrl"]["point"]
        assert db > 0.95, (f, db)                        # length alone separates
        assert abs(dc) < 0.05, (f, dc)                   # but cancels in delta_ctrl
        assert res["per_factor"][f]["tost_delta_ctrl"]["equivalent"], (f, dc)


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    bad = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception:
            bad += 1
            print(f"  FAIL  {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns)-bad}/{len(fns)} passed")
    raise SystemExit(1 if bad else 0)
