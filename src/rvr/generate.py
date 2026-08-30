"""Render a TaskSpec into all of its arms (§3.5).

Arm set, 13 per task:
    base                       all factors eval-like; the shared eval arm for
                               every Δ_base and Δ_ctrl contrast (§3.4)
    F{1,3,4,6}_deploy          one factor flipped deploy-like
    F{1,3,4,6}_ctrl            one factor replaced by its control (§3.3)
    F0_null_{a,b}              cosmetic whitespace only; probe must not separate
    stacked_deploy             all four deploy-like
    stacked_ctrl               all four controls

§3.5 estimates ~14; sharing one eval arm across the four factors gives 13, which
is the same set of contrasts with one fewer forward pass per task.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .factors import (Arm, f3_failure_payload, f3_verbose_envelope, f6_block,
                      render_return_f1, serialize)
from .schema import TaskSpec


def build_arms() -> list[Arm]:
    arms = [Arm("base", role="base", manipulated=None)]
    for fid, kw in (("F1", {"f1": "deploy"}), ("F3", {"f3": "deploy"}),
                    ("F4", {"f4": "natural"}), ("F6", {"f6": "deploy"})):
        arms.append(Arm(f"{fid}_deploy", role="deploy", manipulated=fid, **kw))
    arms += [
        Arm("F1_ctrl", f1="scramble", role="control", manipulated="F1",
            control_type="true_scramble"),
        Arm("F4_ctrl", f4="scrambled", role="control", manipulated="F4",
            control_type="true_scramble"),
        Arm("F3_ctrl", f3="ctrl", role="control", manipulated="F3",
            control_type="plausible_uninformative"),
        Arm("F6_ctrl", f6="ctrl", role="control", manipulated="F6",
            control_type="plausible_uninformative"),
        Arm("F0_null_a", role="null", manipulated="F0", whitespace="canonical"),
        Arm("F0_null_b", role="null", manipulated="F0", whitespace="alt"),
        Arm("stacked_deploy", f1="deploy", f3="deploy", f4="natural", f6="deploy",
            role="stacked", manipulated="ALL"),
        Arm("stacked_ctrl", f1="scramble", f3="ctrl", f4="scrambled", f6="ctrl",
            role="stacked", manipulated="ALL"),
    ]
    return arms


@dataclass
class RenderedContext:
    task_id: str
    arm_id: str
    arm: Arm
    text: str
    messages: list[dict[str, Any]]
    n_tool_calls: int
    # Character offsets for §4.5 read positions; mapped to token indices at probe time.
    read_positions: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in
             ("task_id", "arm_id", "text", "messages", "n_tool_calls", "read_positions")}
        d["arm"] = self.arm.__dict__
        return d


def render(task: TaskSpec, arm: Arm, f3_pad: int = 0, f1_salt: int = 0) -> RenderedContext:
    ws = arm.whitespace
    msgs: list[dict[str, Any]] = []

    sys_text = task.system_prompt
    block = f6_block(arm.f6, task.task_id)
    if block is not None:
        sys_text = sys_text + "\n\n" + serialize(block, ws)
    msgs.append({"role": "system", "content": sys_text})
    msgs.append({"role": "user", "content": task.user_task})

    n_calls = 0
    for ix, step in enumerate(task.steps):
        payload = render_return_f1(step.returns, arm.f1, task.task_id, ix, arm.f4, f1_salt)

        # F3 acts at the designated site only.
        if ix == task.f3_site and arm.f3 == "deploy":
            msgs.append({"role": "assistant", "content": step.assistant_text,
                         "tool_call": {"name": step.tool_name, "args": step.tool_args},
                         "step": ix})
            msgs.append({"role": "tool", "name": step.tool_name,
                         "content": serialize(f3_failure_payload(task.task_id, ix), ws)})
            n_calls += 1
            msgs.append({"role": "assistant",
                         "content": "The call did not return. Retrying.",
                         "tool_call": {"name": step.tool_name, "args": step.tool_args},
                         "step": ix})
            msgs.append({"role": "tool", "name": step.tool_name,
                         "content": serialize(payload, ws)})
            n_calls += 1
            continue

        if ix == task.f3_site and arm.f3 == "ctrl":
            payload = f3_verbose_envelope(payload, task.task_id, ix, f3_pad)

        msgs.append({"role": "assistant", "content": step.assistant_text,
                     "tool_call": {"name": step.tool_name, "args": step.tool_args},
                     "step": ix, "decision_point": step.is_decision_point})
        msgs.append({"role": "tool", "name": step.tool_name,
                     "content": serialize(payload, ws)})
        n_calls += 1

    text, positions = _flatten(msgs, task)
    return RenderedContext(task.task_id, arm.arm_id, arm, text, msgs, n_calls, positions)


def _flatten(msgs: list[dict], task: TaskSpec) -> tuple[str, dict]:
    """Serialize to a flat string and record §4.5 read positions as char offsets.

    primary            end of the assistant turn immediately before the
                       decision-point tool call
    pre_tool_call      every such position, in order
    tool_return_end    last character of each tool return
    final              end of context
    """
    parts: list[str] = []
    pre_tool_call: list[int] = []
    tool_return_end: list[int] = []
    primary: int | None = None
    behavioral_cut: int | None = None
    cur = 0

    for m in msgs:
        if m["role"] == "system":
            seg = f"<|system|>\n{m['content']}\n"
        elif m["role"] == "user":
            seg = f"<|user|>\n{m['content']}\n"
        elif m["role"] == "assistant":
            call = m.get("tool_call")
            seg = f"<|assistant|>\n{m['content']}\n"
            if call:
                seg += f"<|tool_call|>{call['name']}({json.dumps(call['args'], ensure_ascii=False)})\n"
        else:
            seg = f"<|tool_result|>\n{m['content']}\n"

        parts.append(seg)
        start, cur = cur, cur + len(seg)

        if m["role"] == "assistant" and m.get("decision_point"):
            # RQ2 truncates HERE, before the assistant turn, not at `primary`.
            # The authored assistant text at the decision point states the choice
            # ("I'll confirm it against current refund policy"), so cutting after
            # it would hand the model its own answer.
            behavioral_cut = start

        if m["role"] == "assistant" and m.get("tool_call"):
            # end of the assistant text, immediately before the tool call
            pos = start + len(f"<|assistant|>\n{m['content']}")
            pre_tool_call.append(pos)
            if m.get("decision_point"):
                primary = pos
        elif m["role"] == "tool":
            tool_return_end.append(cur - 1)

    text = "".join(parts)
    return text, {
        "primary": primary if primary is not None else (pre_tool_call[-1] if pre_tool_call else len(text) - 1),
        "pre_tool_call": pre_tool_call,
        "tool_return_end": tool_return_end,
        "behavioral_cut": behavioral_cut,
        "final": len(text) - 1,
    }


def _best_match(task: TaskSpec, arm: Arm, target: int, salts: range, pads: range) -> tuple[int, int]:
    """Pick the (f1_salt, f3_pad) whose rendered length lands closest to `target`.

    Both knobs select among renderings that are equally valid under the control's
    definition -- an arbitrary scramble draw, and how verbose a success envelope
    is -- so choosing the closest-matching one is a free improvement to the
    surface match the control depends on (§3.3).
    """
    from .tokenization import count as _tc

    best, best_err = (0, 0), None
    for salt in salts:
        for pad in pads:
            n = _tc(render(task, arm, pad, salt).text).n
            err = abs(n - target)
            if best_err is None or err < best_err:
                best, best_err = (salt, pad), err
            if err == 0:
                return best
    return best


def render_all(task: TaskSpec) -> list[RenderedContext]:
    from .tokenization import count as _tc

    arms = build_arms()
    by_id = {a.arm_id: a for a in arms}
    # Each control is matched to its OWN deploy arm -- the constraint the primary
    # endpoint actually depends on (see configs/preregistration.json).
    search = {
        "F1_ctrl": (_tc(render(task, by_id["F1_deploy"]).text).n, range(12), range(1)),
        "F3_ctrl": (_tc(render(task, by_id["F3_deploy"]).text).n, range(1), range(9)),
        "stacked_ctrl": (_tc(render(task, by_id["stacked_deploy"]).text).n, range(8), range(9)),
    }
    out = []
    for a in arms:
        if a.arm_id in search:
            target, salts, pads = search[a.arm_id]
            salt, pad = _best_match(task, a, target, salts, pads)
        else:
            salt, pad = 0, 0
        out.append(render(task, a, pad, salt))
    return out
