"""Behavioural arm: truncation, choice parsing, and the §11 dynamic-range gate."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rvr.behavioral import (ASSISTANT_CUE, build_prompt, choice_rates,
                            delta_rate, dynamic_range, parse_choice, rate_table)
from rvr.generate import render_all
from rvr.taskloader import load_all


def _spec():
    t = load_all()[0]
    ctx = next(c for c in render_all(t) if c.arm_id == "base")
    return build_prompt(ctx, t), t


def test_truncation_withholds_the_authored_choice():
    """The prompt must stop before the decision-point assistant turn: that text
    states the choice, so leaving it in would hand the model its own answer."""
    spec, task = _spec()
    assert spec.prompt.endswith(ASSISTANT_CUE)
    body = spec.prompt[: -len(ASSISTANT_CUE)]
    # neither branch may be named anywhere in the visible context
    assert spec.cautious not in body, "cautious branch leaked into the prompt"
    # the withheld assistant turn is genuinely absent
    assert body.count("<|assistant|>") == body.count("<|tool_call|>")


def test_every_task_yields_a_usable_prompt():
    tasks = load_all()
    n = 0
    for t in tasks:
        for c in render_all(t):
            s = build_prompt(c, t)
            assert s.prompt.endswith(ASSISTANT_CUE)
            assert s.cautious in s.all_tools and s.efficient in s.all_tools
            assert s.cautious != s.efficient
            n += 1
    assert n == len(tasks) * 13


def test_parse_choice_forms():
    spec, _ = _spec()
    other = next(t for t in spec.all_tools if t not in (spec.cautious, spec.efficient))
    cases = [
        (f"Checking first.\n<|tool_call|>{spec.cautious}({{}})", "cautious"),
        (f"Just doing it.\n<|tool_call|>{spec.efficient}({{}})", "efficient"),
        (f"I'll use {spec.efficient} here.", "efficient"),          # format drift
        (f"Let me re-run {other}.", "other"),
        ("I am not sure how to proceed.", "none"),
        ("", "none"),
    ]
    for text, expect in cases:
        got, _ = parse_choice(text, spec)
        assert got == expect, (text[:40], got, expect)


def test_parse_choice_prefers_the_first_tool_mentioned():
    """Format drift should still score, and the FIRST tool named is the call."""
    spec, _ = _spec()
    got, tool = parse_choice(f"Maybe {spec.efficient}, or perhaps {spec.cautious}.", spec)
    assert got == "efficient" and tool == spec.efficient


def test_dynamic_range_flags_dead_decision_points():
    """§11: a base condition pinned at 0 or 1 means the task design failed, which
    is a different finding from 'the property does not move behaviour'."""
    dead = [{"task_id": f"t{i}", "arm_id": "base", "choice": "cautious"}
            for i in range(20) for _ in range(15)]
    assert dynamic_range(choice_rates(dead))["usable"] is False

    rng = np.random.default_rng(0)
    live = []
    for i in range(20):
        p = rng.uniform(0.25, 0.75)
        for _ in range(15):
            live.append({"task_id": f"t{i}", "arm_id": "base",
                         "choice": "cautious" if rng.random() < p else "efficient"})
    d = dynamic_range(choice_rates(live))
    assert d["usable"] is True and d["tasks_with_variance"] >= 15


def test_delta_rate_direction_and_abstentions():
    rows = []
    for i in range(20):
        for _ in range(15):
            rows.append({"task_id": f"t{i}", "arm_id": "base", "choice": "cautious"})
        for k in range(15):
            rows.append({"task_id": f"t{i}", "arm_id": "F1_deploy",
                         "choice": "cautious" if k < 6 else "efficient"})
    rates = choice_rates(rows)
    tbl = rate_table(rates)
    d = delta_rate(tbl, "F1", tbl.tasks())
    assert abs(d - (1.0 - 6 / 15)) < 1e-9, d       # base cautious - deploy cautious

    # abstentions are excluded from the rate but surfaced separately
    rows2 = [{"task_id": "t0", "arm_id": "base", "choice": c}
             for c in ["cautious", "efficient", "none", "other"]]
    r = choice_rates(rows2)[("t0", "base")]
    assert r["decided"] == 2 and r["cautious_rate"] == 0.5
    assert abs(r["abstain_rate"] - 0.5) < 1e-9


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    bad = 0
    for fn in fns:
        try:
            fn(); print(f"  PASS  {fn.__name__}")
        except Exception:
            bad += 1; print(f"  FAIL  {fn.__name__}"); traceback.print_exc()
    print(f"\n{len(fns)-bad}/{len(fns)} passed")
    raise SystemExit(1 if bad else 0)
