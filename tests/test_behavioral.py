"""Behavioral arm: truncation, choice parsing, and the §11 dynamic-range gate."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rvr.behavioral import (ASSISTANT_CUE, TEMPLATE_PATH, build_prompt,
                            build_prompt_native, choice_rates, delta_rate,
                            dynamic_range, ensure_tool_template, parse_choice,
                            rate_table)
from rvr.generate import render_all
from rvr.taskloader import load_all

GEN_HEADER = "<|start_header_id|>assistant<|end_header_id|>"


class _StubTokenizer:
    """Renders a chat template offline the way transformers does.

    Lets the native-prompt path be tested without downloading a tokenizer:
    the template is the unit under test, not the vocab.
    """

    bos_token = "<|begin_of_text|>"

    def __init__(self, chat_template=None):
        self.chat_template = (chat_template if chat_template is not None
                              else TEMPLATE_PATH.read_text())

    def apply_chat_template(self, messages, tools=None,
                            add_generation_prompt=False, tokenize=False):
        from jinja2.sandbox import ImmutableSandboxedEnvironment

        def raise_exception(msg):
            raise ValueError(msg)

        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        env.globals["raise_exception"] = raise_exception
        return env.from_string(self.chat_template).render(
            messages=messages, tools=tools,
            add_generation_prompt=add_generation_prompt,
            bos_token=self.bos_token)


# a template with no `tools` support, shaped like the NousResearch mirror's:
# it renders the turns and drops the declarations without a sound
_TOOLLESS = ("{{ bos_token }}{% for m in messages %}<|start_header_id|>"
             "{{ m.role }}<|end_header_id|>\n\n{{ m.content }}<|eot_id|>"
             "{% endfor %}{% if add_generation_prompt %}<|start_header_id|>"
             "assistant<|end_header_id|>\n\n{% endif %}")


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


def test_native_prompts_carry_tools_and_withhold_the_choice():
    """The first behavioral run failed because the mirror's template silently
    dropped every tool declaration. Pin the native path end to end: tools
    declared, tool results under the trained `ipython` role, assistant calls
    rendered, and the decision-point call withheld."""
    tok = _StubTokenizer()
    n = 0
    for t in load_all():
        for c in render_all(t):
            s = build_prompt_native(c, t, tok)
            assert s.prompt.rstrip().endswith(GEN_HEADER)
            for name in s.all_tools:
                assert name in s.prompt, (c.task_id, c.arm_id, name)
            assert "<|start_header_id|>ipython<|end_header_id|>" in s.prompt
            assert "<|start_header_id|>tool<|end_header_id|>" not in s.prompt
            # at least one pre-decision assistant tool call must be rendered in
            # the compact call format the model is trained to continue
            assert '{"name": "' in s.prompt, (c.task_id, c.arm_id)
            # the authored decision-point call is withheld: neither branch may
            # appear as a rendered CALL (declarations use indented JSON, calls
            # are compact, so the compact form is unambiguous)
            for branch in (s.cautious, s.efficient):
                assert f'{{"name": "{branch}"' not in s.prompt, \
                    (c.task_id, c.arm_id, branch)
            n += 1
    assert n == len(load_all()) * 13


def test_native_guard_refuses_a_template_that_drops_tools():
    """build_prompt_native must fail loudly, not hand back an unactionable
    prompt, when the template ignores `tools`."""
    t = load_all()[0]
    ctx = next(c for c in render_all(t) if c.arm_id == "base")
    try:
        build_prompt_native(ctx, t, _StubTokenizer(_TOOLLESS))
        assert False, "expected ValueError for dropped tool declarations"
    except ValueError as e:
        assert "dropped tool declarations" in str(e)


def test_ensure_tool_template_swaps_only_when_needed():
    toolless = _StubTokenizer(_TOOLLESS)
    assert ensure_tool_template(toolless) == "vendored_llama31"
    assert "tools" in toolless.chat_template
    # and the swapped-in template actually renders a usable prompt
    t = load_all()[0]
    ctx = next(c for c in render_all(t) if c.arm_id == "base")
    s = build_prompt_native(ctx, t, toolless)
    assert all(name in s.prompt for name in s.all_tools)

    capable = _StubTokenizer()
    assert ensure_tool_template(capable) == "tokenizer"


def test_parse_choice_handles_native_llama_output():
    """The model's own tool-call format is JSON, sometimes behind a python
    tag -- both must score, and prose without a call must not."""
    spec, _ = _spec()
    cases = [
        (f'{{"name": "{spec.cautious}", "parameters": {{"id": "x"}}}}', "cautious"),
        (f'<|python_tag|>{{"name": "{spec.efficient}", "parameters": {{}}}}',
         "efficient"),
        (f'Let me verify first.\n\n{{"name": "{spec.cautious}", "parameters": {{}}}}',
         "cautious"),
        ("I cannot proceed without more information.", "none"),
    ]
    for text, expect in cases:
        got, _ = parse_choice(text, spec)
        assert got == expect, (text[:50], got, expect)


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
    is a different finding from 'the property does not move behavior'."""
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
