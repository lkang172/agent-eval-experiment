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
