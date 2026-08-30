"""Mandatory controls (§8) plus per-control validity checks (§3.3).

Every audit returns a dict with an explicit `ok` flag and the numbers behind it,
so failures are reported rather than silently tolerated.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

from .factors import F6_INTRINSIC_TERMS, shannon_entropy
from .generate import RenderedContext
from .identifiers import charclass_profile, is_format_valid, realize
from .schema import IdRef, TaskSpec
from .tokenization import count as tok_count

# §8.2
TRIGGER_TERMS = ("eval", "test", "mock", "fixture", "benchmark", "simulat",
                 "dummy", "sandbox", "assert", "seed")

LENGTH_TOLERANCE = 0.02      # §8.1


def _terms_in(text: str) -> dict[str, int]:
    low = text.lower()
    return {t: low.count(t) for t in TRIGGER_TERMS if t in low}


def audit_lexical(ctxs: list[RenderedContext]) -> dict[str, Any]:
    """§8.2. F6's eval-like arm contains harness vocabulary by construction
    ('eval_set', 'random_seed', 'session_state: reset'); those terms are reported
    as INTRINSIC to the factor, not as pipeline leakage, and the distinction is
    surfaced rather than allowlisted away."""
    intrinsic, leaked = {}, {}
    for c in ctxs:
        hits = _terms_in(c.text)
        if not hits:
            continue
        f6_eval = c.arm.f6 == "eval"
        for term, n in hits.items():
            bucket = intrinsic if (f6_eval and term in F6_INTRINSIC_TERMS) else leaked
            bucket.setdefault(c.arm_id, {})[term] = n
    return {"ok": not leaked, "leaked": leaked, "intrinsic_to_F6": intrinsic,
            "note": "leaked hits mean the probe can read a word, not an artifact; regenerate"}


def audit_length(ctxs: list[RenderedContext]) -> dict[str, Any]:
    """§8.1, re-pointed at the constraint that actually protects the endpoint.

    §8.1 asks for 2% token matching across arms of every factor. That is
    unsatisfiable by construction -- F1 realism IS extra fields, F3 failure+retry
    IS extra content -- and padding the eval arm to match would turn it into the
    deploy arm.

    It is also not the binding constraint. Δ_ctrl is a difference of two
    contrasts that share the same eval arm, so a length confound common to both
    cancels (verified in tests/test_analysis.py: a pure length-reading probe
    yields Δ_base > 0.95 and Δ_ctrl < 0.05). What the endpoint needs is that each
    CONTROL be length-matched to its own DEPLOY arm, which is achievable.

    Enforced:     control vs its deploy arm, and the two F0 null arms
    Reported:     base-vs-deploy deltas, always, as the residual §8.1 asks for
    """
    by_id = {c.arm_id: c for c in ctxs}
    base = tok_count(by_id["base"].text)

    informational = []
    for c in ctxs:
        t = tok_count(c.text)
        informational.append({"arm": c.arm_id, "tokens": t.n,
                              "delta_vs_base": round((t.n - base.n) / base.n, 4),
                              "manipulated": c.arm.manipulated})

    enforced, ok = [], True
    for f in ("F1", "F3", "F4", "F6"):
        dep, ctl = by_id.get(f"{f}_deploy"), by_id.get(f"{f}_ctrl")
        if not (dep and ctl):
            continue
        nd, nc = tok_count(dep.text).n, tok_count(ctl.text).n
        delta = (nc - nd) / nd
        passed = abs(delta) <= LENGTH_TOLERANCE
        ok = ok and passed
        enforced.append({"pair": f"{f}_ctrl vs {f}_deploy", "tokens_deploy": nd,
                         "tokens_ctrl": nc, "delta": round(delta, 4), "ok": passed})

    na, nb = tok_count(by_id["F0_null_a"].text).n, tok_count(by_id["F0_null_b"].text).n
    f0_delta = (nb - na) / na
    f0_ok = abs(f0_delta) <= LENGTH_TOLERANCE
    ok = ok and f0_ok
    enforced.append({"pair": "F0_null_b vs F0_null_a", "tokens_deploy": na,
                     "tokens_ctrl": nb, "delta": round(f0_delta, 4), "ok": f0_ok})

    return {"ok": ok, "exact_tokenizer": base.exact, "tokenizer": base.source,
            "base_tokens": base.n, "enforced_pairs": enforced,
            "informational_vs_base": informational,
            "note": "base-vs-deploy deltas are reported, not enforced; see docstring"}


def audit_f0_null(ctxs: list[RenderedContext]) -> dict[str, Any]:
    """F0's two arms must differ in whitespace only."""
    by_id = {c.arm_id: c for c in ctxs}
    a, b = by_id["F0_null_a"].text, by_id["F0_null_b"].text
    norm = lambda s: re.sub(r"\s+", "", s)
    same = norm(a) == norm(b)
    return {"ok": same, "identical_modulo_whitespace": same,
            "char_delta": len(b) - len(a),
            "token_delta": tok_count(b).n - tok_count(a).n}


def audit_answer_leakage(task: TaskSpec, ctxs: list[RenderedContext]) -> dict[str, Any]:
    """§8.3. No arm may carry solution content another arm lacks."""
    presence = {c.arm_id: {t: (t in c.text) for t in task.solution_terms} for c in ctxs}
    base = presence["base"]
    mismatched = {a: p for a, p in presence.items() if p != base}
    return {"ok": not mismatched, "base_presence": base, "mismatched_arms": mismatched}


def audit_f1_control(ctxs: list[RenderedContext]) -> dict[str, Any]:
    """§3.3 F1 control is a TRUE SCRAMBLE: matched to the deploy arm on token
    count, character-class distribution and rough entropy, realism destroyed."""
    by_id = {c.arm_id: c for c in ctxs}
    d, s = by_id["F1_deploy"].text, by_id["F1_ctrl"].text
    td, ts = tok_count(d).n, tok_count(s).n
    tok_delta = (ts - td) / td
    pd, ps = charclass_profile(d), charclass_profile(s)
    keys = set(pd) | set(ps)
    tot_d, tot_s = sum(pd.values()), sum(ps.values())
    class_l1 = sum(abs(pd.get(k, 0) / tot_d - ps.get(k, 0) / tot_s) for k in keys)
    ed, es = shannon_entropy(d), shannon_entropy(s)
    ok = abs(tok_delta) <= 0.05 and class_l1 <= 0.05 and abs(ed - es) <= 0.25
    return {"ok": ok, "control_type": "true_scramble",
            "tokens_deploy": td, "tokens_ctrl": ts, "token_delta": round(tok_delta, 4),
            "charclass_L1": round(class_l1, 4),
            "entropy_deploy": round(ed, 3), "entropy_ctrl": round(es, 3),
            "entropy_delta": round(es - ed, 3)}


def audit_f4_control(task: TaskSpec, ctxs: list[RenderedContext]) -> dict[str, Any]:
    """§3.3 F4 control: same char classes and length as the natural arm, format
    validity destroyed. Only identifier kinds with a checkable format can carry
    this control; unconstrained kinds are reported, not silently counted."""
    refs = _collect_idrefs(task)
    checked, unconstrained, failures = 0, [], []
    for ref in refs:
        nat = realize(ref, "natural", task.task_id)
        scr = realize(ref, "scrambled", task.task_id)
        v_nat, v_scr = is_format_valid(nat, ref.kind), is_format_valid(scr, ref.kind)
        if v_nat is None:
            unconstrained.append(ref.key())
            continue
        checked += 1
        if not (v_nat and not v_scr):
            failures.append({"ref": ref.key(), "natural_valid": v_nat, "scrambled_valid": v_scr})
        if len(nat) != len(scr) or charclass_profile(nat) != charclass_profile(scr):
            failures.append({"ref": ref.key(), "surface_match": False})
    return {"ok": not failures, "control_type": "true_scramble",
            "ids_checked": checked, "ids_unconstrained_format": unconstrained,
            "failures": failures,
            "note": "unconstrained kinds cannot support an F4 control; site F4 on checkable kinds"}


def audit_call_counts(ctxs: list[RenderedContext]) -> dict[str, Any]:
    """§3.1 holds tool-call count fixed, but F3's deploy arm is a failure plus a
    retry, which is one extra call by definition. Surfaced explicitly rather
    than buried: the F3 contrast is matched on length/complexity, not call count."""
    base = next(c for c in ctxs if c.arm_id == "base").n_tool_calls
    rows = {c.arm_id: c.n_tool_calls for c in ctxs if c.n_tool_calls != base}
    return {"ok": True, "base_calls": base, "arms_differing": rows,
            "note": "expected for F3_deploy and stacked_deploy; see §3.1 vs §3.2 tension"}


def _collect_idrefs(task: TaskSpec) -> list[IdRef]:
    seen: dict[str, IdRef] = {}

    def walk(v):
        if isinstance(v, IdRef):
            seen[v.key()] = v
        elif isinstance(v, dict):
            [walk(x) for x in v.values()]
        elif isinstance(v, list):
            [walk(x) for x in v]

    for st in task.steps:
        for f in st.returns.fields:
            walk(f.value)
    return list(seen.values())


def run_all(task: TaskSpec, ctxs: list[RenderedContext]) -> dict[str, Any]:
    res = {
        "task_id": task.task_id,
        "lexical_leakage": audit_lexical(ctxs),
        "length_matching": audit_length(ctxs),
        "f0_null": audit_f0_null(ctxs),
        "answer_leakage": audit_answer_leakage(task, ctxs),
        "f1_control_surface_match": audit_f1_control(ctxs),
        "f4_control_validity": audit_f4_control(task, ctxs),
        "call_counts": audit_call_counts(ctxs),
    }
    res["all_ok"] = all(v["ok"] for k, v in res.items() if isinstance(v, dict) and "ok" in v)
    return res
