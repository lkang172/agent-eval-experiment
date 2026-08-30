"""Validate probe training against synthetic activations with known structure."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rvr.activations import char_to_token
from rvr.probe import (best_layer, cross_val_auroc, fit_at_layer, gate_probe_a,
                       gate_probe_b, layer_sweep)


def _synth(n_tasks=30, n_layers=8, hidden=64, signal_layer=5, seed=0):
    """Activations where only `signal_layer` separates eval from deploy."""
    rng = np.random.default_rng(seed)
    X, y, groups = [], [], []
    for t in range(n_tasks):
        # per-task offset: shared by all of a task's variants, which is exactly
        # the structure that makes a random split leak
        task_bias = rng.normal(0, 3.0, size=hidden)
        for label in (0, 1):
            for _ in range(3):
                item = rng.normal(0, 1.0, size=(n_layers, hidden)) + task_bias
                if label == 1:
                    item[signal_layer] += 1.6
                X.append(item)
                y.append(label)
                groups.append(f"task_{t:02d}")
    return np.stack(X), np.array(y), groups


def test_char_to_token_edges():
    offs = [(0, 5), (5, 11), (11, 15)]
    assert char_to_token(offs, 0) == 0
    assert char_to_token(offs, 4) == 0
    assert char_to_token(offs, 5) == 1
    assert char_to_token(offs, 14) == 2
    assert char_to_token(offs, 9999) == 2      # past the end clamps to last


def test_layer_sweep_finds_the_signal_layer():
    X, y, groups = _synth(signal_layer=5)
    res = layer_sweep(X, y, groups, progress=False)
    assert best_layer(res).layer == 5, [str(r) for r in res]
    # layers without signal should sit near chance
    others = [r.auroc_mean for r in res if r.layer != 5]
    assert max(others) < 0.65, others


def test_grouped_cv_prevents_task_leakage():
    """The load-bearing correctness claim for Probe B.

    The realistic leak is not a label-independent task offset -- that cannot
    reveal a label. It is that each task's eval/deploy difference is partly
    TASK-IDIOSYNCRATIC, because all of a task's variants share its content. A
    random split lets the probe learn task T's private eval direction from some
    of T's variants and apply it to the rest. Holding out by task cannot.
    """
    def build(shared_w, private_w, hidden=32, n_tasks=16, per_label=10, seed=1):
        rng = np.random.default_rng(seed)
        shared = rng.normal(0, 1, hidden)
        shared /= np.linalg.norm(shared)
        X, y, groups = [], [], []
        for t in range(n_tasks):
            private = rng.normal(0, 1, hidden)
            private /= np.linalg.norm(private)
            for label in (0, 1):
                for _ in range(per_label):
                    v = rng.normal(0, 1.0, hidden)
                    if label == 1:
                        v += shared_w * shared + private_w * private
                    X.append(v)
                    y.append(label)
                    groups.append(f"task_{t:02d}")
        return np.stack(X), np.array(y), groups

    # (1) Signal is PURELY task-private: there is no generalisable structure at
    # all, so the honest answer is chance. A random split still reports a
    # confident-looking ceiling built entirely from task memorisation.
    X, y, groups = build(shared_w=0.0, private_w=4.0)
    grouped, _, _ = cross_val_auroc(X, y, groups=groups)
    ungrouped, _, _ = cross_val_auroc(X, y, groups=None)
    assert ungrouped > grouped + 0.15, (ungrouped, grouped)
    assert abs(grouped - 0.5) < 0.08, grouped

    # (2) With a genuinely shared component, grouped CV recovers it.
    X, y, groups = build(shared_w=2.5, private_w=1.0)
    grouped2, _, _ = cross_val_auroc(X, y, groups=groups)
    assert grouped2 > 0.7, grouped2


def test_fitted_probe_scores_and_generalises():
    X, y, groups = _synth(signal_layer=5)
    p = fit_at_layer(X, y, layer=5, groups=groups)
    s = p.score(X[:, 5, :])
    assert s.shape == (len(y),)
    assert s[y == 1].mean() > s[y == 0].mean()
    # train AUROC should not be wildly below CV; both should be sane
    assert p.train_auroc > p.cv_auroc - 0.05
    assert 0.5 < p.cv_auroc <= 1.0


def test_gates():
    a = gate_probe_a(0.81, {"chars": 0.546, "special_chars": 0.514})
    assert a["passed"] and all(a["beats_baselines"].values())
    # clears 0.75 but loses to a surface baseline -> must fail
    a2 = gate_probe_a(0.78, {"chars": 0.80})
    assert not a2["passed"]
    # below threshold -> fail
    assert not gate_probe_a(0.70, {"chars": 0.55})["passed"]
    assert gate_probe_b(0.72)["passed"]
    assert not gate_probe_b(0.51)["passed"]
    assert "STOP" in gate_probe_b(0.51)["consequence"]


def _agentic_sim(n_tasks=30, hidden=512, effect=0.9, seed=5):
    """Contexts shaped like the real arm set.

    Each task has a large private offset (variants share content) and a small
    shared eval/deploy direction. F0's two null arms are cosmetic copies of base.

    `hidden` defaults to 512 against 180 training points, matching the real
    run's overparameterised regime (180 points in 4096 dimensions) -- that ratio
    is what lets an in-sample probe memorise its training arms perfectly.
    """
    rng = np.random.default_rng(seed)
    shared = rng.normal(0, 1, hidden)
    shared /= np.linalg.norm(shared)
    # The cosmetic whitespace change is the SAME edit in every task, so it moves
    # activations along a consistent direction rather than randomly. That is why
    # the real run saw AUROC exactly 0.0 between the two null arms: an overfit
    # probe reads any systematic departure from its memorised base point as
    # "less eval-like", identically for all 30 tasks.
    cosmetic = rng.normal(0, 1, hidden)
    cosmetic /= np.linalg.norm(cosmetic)
    arms = (["base"] + [f"{f}_deploy" for f in ("F1", "F3", "F4", "F6")]
            + [f"{f}_ctrl" for f in ("F1", "F3", "F4", "F6")]
            + ["F0_null_a", "F0_null_b", "stacked_deploy", "stacked_ctrl"])
    X, tasks, arm_ids = [], [], []
    for t in range(n_tasks):
        task_id = f"task_{t:02d}"
        private = rng.normal(0, 4.0, hidden)          # dominates; shared by variants
        # base is a training point. F0_null_a is byte-identical to base in the
        # real pipeline, so it is the SAME vector; F0_null_b differs only
        # cosmetically. That asymmetry -- one null arm memorised, the other a
        # perturbation of it -- is what makes an in-sample probe "separate" them.
        base_vec = private + rng.normal(0, 1.0, hidden) + effect * shared
        for arm in arms:
            if arm == "base" or arm == "F0_null_a":
                v = base_vec
            elif arm == "F0_null_b":
                v = base_vec + 0.6 * cosmetic + rng.normal(0, 0.05, hidden)
            else:
                v = private + rng.normal(0, 1.0, hidden)
            X.append(v)
            tasks.append(task_id)
            arm_ids.append(arm)
    X = np.stack(X)[:, None, :]                        # single layer axis
    return X, tasks, arm_ids


def test_oof_scoring_fixes_saturated_deltas_and_the_f0_null():
    """Regression test for the two defects that invalidated the first run.

    In-sample: Probe B is fit on base+deploy then scores those same contexts, so
    Δ_base saturates and F0 fails for reasons unrelated to any leak.
    Out-of-fold: every context is scored by a probe that never saw its task.
    """
    from rvr.analysis import ScoreTable, _delta_base, f0_null_check
    from rvr.probe import fit_at_layer, fit_folds_and_score_all

    X, tasks, arms = _agentic_sim()
    train_arms = {"base", "F1_deploy", "F3_deploy", "F4_deploy", "F6_deploy",
                  "stacked_deploy"}
    sel = [i for i, a in enumerate(arms) if a in train_arms]
    y = [1 if arms[i] == "base" else 0 for i in sel]
    g = [tasks[i] for i in sel]

    def table(scores):
        return ScoreTable([{"task_id": tasks[i], "arm_id": arms[i],
                            "score": float(scores[i]), "tokens": 0,
                            "special_chars": 0} for i in range(len(arms))])

    # --- in-sample (the bug) ---
    p = fit_at_layer(X[sel], y, layer=0, groups=g)
    t_in = table(p.score(X[:, 0, :]))
    d_in = _delta_base(t_in, "F1", "score", t_in.tasks())
    f0_in = f0_null_check(t_in, margin=0.1)

    # --- out-of-fold (the fix) ---
    oof, info = fit_folds_and_score_all(X[sel], y, g, X, tasks, layer=0)
    assert info["n_unscored"] == 0, info
    t_oof = table(oof)
    d_oof = _delta_base(t_oof, "F1", "score", t_oof.tasks())
    f0_oof = f0_null_check(t_oof, margin=0.1)

    # In-sample saturates; out-of-fold does not.
    assert d_in > 0.97, d_in
    assert d_oof < d_in - 0.05, (d_in, d_oof)
    # Under out-of-fold scoring the two cosmetic arms do not separate, which is
    # what the F0 null control is supposed to show.
    assert f0_oof["passes"], f0_oof
    # NOTE: the real run's in-sample F0 AUROC of exactly 0.0 is not asserted
    # here. Reproducing it needs the whitespace-induced activation shift to be
    # non-orthogonal to the probe direction, which is a property of real
    # activation geometry, not of any defensible synthetic model. The claim this
    # test pins is the saturation and its fix; whether in-sample F0 also fails is
    # data-dependent.
    _ = f0_in


def test_delta_base_polarity_is_eval_positive():
    """Δ_base must exceed 0.5 when the eval arm scores higher, not fall below it."""
    from rvr.analysis import ScoreTable, _delta_base

    rows = []
    for t in range(20):
        rows.append({"task_id": f"t{t}", "arm_id": "base", "score": 1.0,
                     "tokens": 0, "special_chars": 0})
        rows.append({"task_id": f"t{t}", "arm_id": "F1_deploy", "score": 0.0,
                     "tokens": 0, "special_chars": 0})
    tbl = ScoreTable(rows)
    assert _delta_base(tbl, "F1", "score", tbl.tasks()) == 1.0


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
