# §7 Day-1 harness verification — ADK 2.8.0

**Question.** Does ADK's eval path alter the system prompt, inject instructions, or
change tool descriptions relative to a live run? If it does, this study measures
prompt cues rather than harness artifacts and the design needs a pivot.

**Verdict: PASS, with one documented conditional caveat.**

## What the eval path actually adds

`EvaluationGenerator._generate_inferences_from_root_agent` builds its Runner via
`_build_eval_runner_kwargs`, merging two plugins that are absent from a live run:

| Plugin | What it does | Reaches model input? |
|---|---|---|
| `_RequestIntercepterPlugin` | Caches each `LlmRequest` and writes a UUID to `callback_context.state["__llm_request_key__"]`, so AutoRater metrics can recover the instructions and tool list shown to the model | No, by default |
| `EnsureRetryOptionsPlugin` | Sets `config.http_options.retry_options` if absent (7 attempts, backoff) | No — HTTP transport policy, not prompt content |

ADK's own comment at `evaluation_generator.py:281` confirms the asymmetry is
deliberate: "By default, live API calls do not include before_model_callback and
after_model_callback. These callbacks are needed by the plugins to include the
agent instructions and tool declarations in the eval."

## Empirical differential

`scripts/verify_harness.py` runs one agent with one tool through a live Runner
and through ADK's own eval Runner construction, capturing every `LlmRequest` with
a recording model, across two turns.

```
turn 1  system_instruction  identical      turn 2  system_instruction  identical
turn 1  tools               identical      turn 2  tools               identical
turn 1  contents            identical      turn 2  contents            identical
eval-only markers reaching model input: none
```

Two turns rather than one, because the intercepter's state write happens in
`before_model_callback` — after turn 1's request is built — so anything it
persists could only surface from turn 2.

## Sensitivity controls

A differential that always says "identical" proves nothing.
`scripts/verify_harness_sensitivity.py` establishes that it does not:

- **Positive control.** A plugin appending "This is an evaluation run." to the
  system instruction is **detected**. The differential is sensitive.
- **State-template vector.** An agent whose instruction interpolates
  `{__llm_request_key__}` **does leak**: turn 1 substitutes empty, turn 2
  substitutes a live UUID written by the eval-only intercepter.

## The caveat, stated precisely

ADK's eval path does not contaminate prompts *for agents with static
instructions*. It can contaminate them for agents that use ADK's
`{placeholder}` state templating, because `inject_session_state` substitutes any
state key the instruction names — including keys written by eval-only plugins.
`instructions_utils._render_with_regex` substitutes only explicit placeholders
and early-returns when the template contains no `{`, which is why static prompts
are immune.

This is narrow, but it is a real eval/deploy asymmetry in a widely used harness
and worth one sentence in the paper's harness section.

**Mitigation in this study.** `TaskSpec.validate()` rejects any task whose
`system_prompt` contains `{` or `}`, so no task can silently adopt the vulnerable
pattern. All current tasks use static prompts and validate clean.

## Artifacts

- `results/harness_verification.json` — full live and eval request snapshots
- `results/harness_verification_sensitivity.json` — both controls
- `scripts/verify_harness.py`, `scripts/verify_harness_sensitivity.py`
