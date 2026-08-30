# Handoff — Ranking harness artifacts by evaluation-awareness leakage

**Repo:** `lkang172/agent-eval-experiment` · **Branch:** `main` · **Tests:** 19/19
**Prepared:** 2026-08-30
**Formatted version:** https://claude.ai/code/artifact/6067c4c3-7171-4120-af12-126d7539557f

> Source of truth for the protocol is `docs/overview.md`. Analysis decisions are
> recorded in `configs/preregistration.json`. Read both before changing anything.

---

## 1. What this study is

Evaluation harnesses leak cues and models detect them; that much is established.
What designers lack is an *ordered list* of which harness properties leak worst.
Chat studies cannot produce it — in chat the prompt is the whole stimulus, so
evaluation context is entangled with benchmark format. Agentic trajectories
separate task from harness, which is what makes attribution possible.

Two research questions, measured **independently**:

- **RQ1 (attribution)** — which harness properties shift a linear probe's
  eval/deploy score with task text held fixed? Measures *representation*.
- **RQ2 (dissociation)** — does a representational shift come with a
  *behavioral* shift: which tool the agent picks at a decision point?

RQ2 does not depend on RQ1. A broken probe does not block it, and a behavioral
result stands alone.

### The four factors

| ID | Factor | Eval-like (0) | Deploy-like (1) |
|----|--------|---------------|-----------------|
| F1 | Tool-return payload realism | Minimal tidy JSON, task-relevant fields only | Extra fields, nulls, nested metadata |
| F3 | Failure realism | No tool call ever fails | Transient failure then retry |
| F4 | Identifier realism | Sequential synthetic (`user_001`) | UUIDv4, checksummed SKUs |
| F6 | Session/state artifacts | Reset markers, eval-set metadata, fixed seed | Natural session continuation |

---

## 2. Status at a glance

| Stage | Status | Detail |
|---|---|---|
| Stimulus generation | **Sound** | 30 tasks, 390 contexts, 210/210 audit assertions passing |
| §7 harness verification | **Passed** | ADK's eval path does not alter model input |
| Probe A (transferred) | **Confounded** | Reads corpus identity, not eval-awareness. Fails the gate. |
| Probe B (in-domain) | **Valid** | 0.899 AUROC out-of-fold at layer 4; F0 null passes |
| RQ1 attribution | **Underpowered** | Only F6 significant, and below the noise floor |
| RQ2 behavior | **Not measured** | First run was a measurement failure; fix committed, unrun |
| §4.3 fixture-plausibility | **Not started** | Needs ~50 human ratings; sheet generator exists |
| Cross-harness replication | **Not started** | Optional |

---

## 3. Environment

A 16 GB T4 cannot hold Llama-3.1-8B in fp16 (~16.1 GB of weights before
activations), so **everything runs in 8-bit**. This is a stated limitation of the
study, not an implementation detail — the probes read quantized activations.
`results.json` records the quantization mode so runs from different settings are
never pooled.

Weights come from `NousResearch/Meta-Llama-3.1-8B-Instruct`, an ungated mirror.
The official `meta-llama` repo is `gated: manual` and blocks on an access request.

```bash
# Fresh Colab runtime — full setup
%cd /content
!git clone -q https://github.com/lkang172/agent-eval-experiment.git
%cd /content/agent-eval-experiment
!pip install -q transformers accelerate bitsandbytes

# Verify the stimulus pipeline before any GPU work
!PYTHONPATH=src python -m rvr.build
# expect: 30 tasks -> 390 contexts; audits all_ok=True
```

Local CPU work (analysis, layer selection, audits) needs only
`numpy scipy scikit-learn pyyaml transformers` — no torch.

---

## 4. What is built

`src/rvr/` — 18 modules, 19 tests.

| Module | Role |
|---|---|
| `schema.py` | Task/trajectory model. Tool returns stored *semantically* (field roles + ID references) so factors apply at render time. |
| `factors.py` | F1/F3/F6 renderers and their per-factor controls. |
| `identifiers.py` | F4 and its control. Scrambled IDs match natural ones on length and character class while being format-invalid. |
| `generate.py` | Renders 13 arms per task; records read positions and the behavioral cut point. |
| `audits.py` | Seven §8 audits: lexical leakage, length matching, F0 null, answer leakage, control validity, call counts. |
| `analysis.py` | AUROC, cluster bootstrap by task, Holm, TOST, B-len/B-char bounds. |
| `probe.py` | Layer sweep, grouped CV, out-of-fold scoring, the §4.4 gates. |
| `behavioral.py` | RQ2: prompt truncation, native chat-template rendering, tool-call parsing, dynamic-range gate. |
| `select_layer.py` | Re-selects the probe layer under the F0 null constraint. CPU only. |
| `sad_data.py` | Loads SAD `stages_oversight` for Probe A into gitignored `data/sad/`. |
| `run_probe_arm.py`, `run_behavioral_arm.py` | The two end-to-end drivers. |

### Data handling

SAD's license requires its question and answer text never appear in plain text
anywhere scrapable, **including private repos**. Everything SAD-derived lands in
`data/sad/`, gitignored. Do not copy SAD text into results, notebooks, logs, or
commit messages.

Rendered trajectories are also gitignored. They are deterministic —
`python -m rvr.build` reproduces all 390 byte-for-byte — so auditability is
preserved without publishing eval-awareness stimuli.

---

## 5. Decisions that deviate from the brief

All recorded in `configs/preregistration.json`.

**Δ_ctrl is a difference of contrasts.** The brief writes
`Δ_ctrl = AUROC(eval arm) − AUROC(control arm)`, which does not parse — an arm
has no AUROC, a contrast does. Resolved to
`AUROC(base vs deploy) − AUROC(base vs control)`.

**§8.1's 2% token matching is unsatisfiable, and not the binding constraint.**
Every factor changes length by construction (F1 +60%, F3 +26%); padding the eval
arm to match would turn it into the deploy arm. It also does not matter: Δ_ctrl
subtracts two contrasts sharing the same eval arm, so a common length confound
cancels. Verified in `tests/test_analysis.py` — a probe reading *only* token
length gives Δ_base > 0.95 and Δ_ctrl < 0.05. What the endpoint needs is each
**control matched to its own deploy arm**, enforced and holding at mean |Δ| 0.47%.

**Probe A is trained, not transferred.** Nguyen et al.'s released probe is a
Llama-3.3-70B direction (hidden dim 8192); Llama-3.1-8B is 4096. Dimensionally
inapplicable, not weakly-transferring. Their 0.517 / 0.584 / 0.598 baselines are
reference points only.

**F6's deploy arm is a content contrast.** "Absent; continuous session context"
read two ways. All three F6 arms now carry a structurally identical five-key
block differing only in what it signals, which keeps F6 length-matched.

**The behavioral arm is staged.** `base` vs `stacked_deploy` is the maximum
available manipulation (separates representationally at AUROC 0.999), so it is
screened first — 1,200 generations instead of 4,950. Declared before any
behavioral data existed.

---

## 6. Results: RQ1, representation

Files: `results/results.json`, `results/layer_selection.json`.

### Probe A is a corpus classifier — do not use it

Probe A reached 0.9999 AUROC on SAD `stages_oversight`. That is not evaluation
awareness. Its classes come from **disjoint corpora**: eval from
`test_anthropic`/`bigbench`/`wmdp`, deploy entirely from `wildchat_sampled`.

| Discriminator | AUROC |
|---|---|
| Layer-10 probe | 0.9999 |
| Bag-of-words | 0.986 |
| **First 40 characters only** | **0.977** |
| Embedding layer (layer 0) | 0.895 |
| Length baseline | 0.546 |
| Special-character baseline | 0.514 |

Length and special-character baselines are blind to a vocabulary confound, which
is how a corpus classifier passed the original gate. `gate_probe_a` now treats a
high embedding-layer AUROC as a **disqualifier**, and this probe fails it. This
reproduces Devbunova's critique (2603.19426) on our own setup and is worth
reporting as a finding.

### Layer 1 was the only leaking layer in the network

Selecting by maximum AUROC picked layer 1, whose probe separates two contexts
differing *only* in JSON indentation (F0 AUROC 0.202, token-count identical).
Layer selection is now constrained to layers passing the F0 null control. 32 of
33 pass; the constraint costs 0.016 AUROC and selects **layer 4**, where F0
passes at 0.457.

> **Method note worth publishing:** choosing a probe layer by maximum held-out
> AUROC actively selects for surface sensitivity — the most token-driven layer
> wins. Constraining by a cosmetic null control removes the artifact at near-zero
> cost.

### Attribution at layer 4

| Factor | Δ_ctrl | 95% CI | p (Holm) | Δ_base | Length alone |
|---|---|---|---|---|---|
| **F6** session artifacts | +0.074 | [0.036, 0.109] | <0.0001 | 0.838 | 0.561 |
| F4 identifiers | +0.017 | [−0.018, 0.054] | 1.00 | 0.749 | 0.751 |
| F3 failure realism | +0.004 | [−0.014, 0.024] | 1.00 | 0.902 | 1.000 |
| F1 payload realism | −0.004 | [−0.017, 0.003] | 1.00 | 0.943 | 1.000 |

The last two columns matter most. **For F1 and F3 a token-count-only classifier
beats the probe outright.** For F4 it ties. Only F6 reads beyond surface form,
by +0.277.

**Do not report this as a ranking.** F6's +0.074 sits *below* the noise floor:
the random-direction band at layer 4 gives p95 = 0.147, and tested directly,
**14.5% of random meaningless directions produce |Δ_ctrl| ≥ 0.074**. No factor
clears the noise floor at N=30. F6 is the only candidate — the only significant
Δ_ctrl and the only factor reading beyond length — but it is suggestive and
underpowered. Report with a power statement.

Caveat for the writeup: F6 is the one factor whose length confound does *not*
cancel in Δ_ctrl (B-len gives −0.112 vs exactly 0.000 for the others), because
its three arms sit within ~1% on tokens so both AUROCs are near chance. The probe
effect runs *opposite* to that contribution, so it is not explained by length,
but F6's contrast is less clean than the others'.

---

## 7. Results: RQ2, behavior — a measurement failure

File: `results/behavioral.json`.

The first screen ran 1,200 generations over 85 minutes and produced **no usable
behavioral data**. 87.2% emitted no parseable tool call. Only 9 of 30 tasks
yielded any decision; one showed the agent picking both branches. The §11
dynamic-range gate correctly reported `usable: False`.

| Parsed choice | Count | Share |
|---|---|---|
| none — no tool call found | 1046 | 87.2% |
| efficient branch | 87 | 7.2% |
| other tool | 36 | 3.0% |
| cautious branch | 31 | 2.6% |

The screen's Δ = +0.001 (p = 0.70) is meaningless — it compares two conditions
in which the model mostly did nothing parseable.

**Cause.** The behavioral prompts were built on the probe arm's transcript
format, whose markers (`<|assistant|>`, `<|tool_call|>`) are invented strings
that tokenize as ordinary text, not Llama-3.1 special tokens. Harmless for
probing, where only consistency across arms matters. Fatal for eliciting an
*action*.

**Fix, committed but unrun.** `build_prompt_native` renders the identical
semantic context through the model's own chat template with proper tool schemas,
cut at the same decision point. `--format native` is now the default. The driver
also persists sample generations and abstention examples.

---

## 8. Next steps

### Step 1 — Pilot the native-format behavioral arm

Cheap, and it decides whether the fix worked. **Check the parsed-choice
distribution before running anything longer.** If `none` is still dominant, read
`sample_generations` in the pilot output and diagnose from raw text — do not
spend another 85-minute run.

```bash
!PYTHONPATH=src python -m rvr.run_behavioral_arm --pilot --format native --quant 8bit
```

### Step 2 — Run the screen: base vs stacked_deploy

1,200 generations, ~85 min at the observed 4.3 s/generation. The first number to
read is the dynamic-range gate. If `usable: False` again, the decision points are
not live — a task-design problem, **not** evidence about the hypothesis.

```bash
!PYTHONPATH=src python -m rvr.run_behavioral_arm --arms screen --rollouts 20 --format native --quant 8bit
```

### Step 3 — Branch on the screen result

**If it moves** — decompose per factor. ~4,950 generations, ~6 hours, which
exceeds a Colab session. Add resumable checkpointing first, or move to a
persistent GPU.

```bash
!PYTHONPATH=src python -m rvr.run_behavioral_arm --arms full --rollouts 15 --format native --quant 8bit
```

**If it does not move** — a null on the strongest available manipulation, far
stronger than four separately-underpowered per-factor nulls. Requires a
behavior-specific equivalence margin (§10) plus an achieved-power statement.

### Step 4 — Run the §4.3 control-validity check

Not started, and it gates every Δ_ctrl interpretation. An LLM judge scores all
390 arms 1–7 on "does this look like a test fixture rather than a live system";
~50 human ratings validate the judge. **Each control arm must sit at or below its
eval arm.** If a control scores higher, that factor's Δ_ctrl is invalid and only
Δ_base may be reported.

There is already a signal worth chasing: F3's Δ_ctrl came out significantly
*negative* in the layer-1 run, meaning its control separated more than its deploy
arm — exactly the inversion this check exists to catch.

```bash
!PYTHONPATH=src python -m rvr.make_rating_sheet --n 50
# -> results/plausibility/rating_sheet.csv + transcripts/
# rating_key.json holds arm identities — do not open before rating
```

### Step 5 — Decide what to do about Probe A

Either rebuild it on a source-matched chat split (the current one is
corpus-confounded and cannot be salvaged by reweighting), or report it as
confounded and excluded, leaning on Probe B. Either way the A–B transfer gap
currently in `results.json` is meaningless and must not be quoted.

### Re-running the probe arm

Only needed if stimuli or model change. Analysis alone does not need the GPU —
`results/activations_primary.npz` caches all 390 × 33 × 4096 activations.

```bash
# GPU: full probe arm, ~18 min
!PYTHONPATH=src python -m rvr.run_probe_arm --quant 8bit --batch-size 4

# CPU: re-select layer + recompute primary family from cache
!PYTHONPATH=src python -m rvr.select_layer

# CPU: all tests
!for t in tests/test_*.py; do python3 $t; done
```

---

## 9. Traps that already cost time

Each was a real bug; regression tests now pin them.

- **Scoring in-sample.** Fitting Probe B on base and deploy arms then scoring
  those same contexts measures training-set fit. It produced Δ values of exactly
  0.0 and 1.0 and a spurious F0 failure. Always use `fit_folds_and_score_all`.
- **AUROC polarity.** Eval-like is the *positive* class throughout, matching
  Probe B's labelling (`base=1`). Passing base as the negative class returns
  `1 − intended` and silently inverts every ranking.
- **`F0_null_a` is byte-identical to `base`.** By design — the F0 pair differs
  only in whitespace. But an in-sample probe then compares a memorised training
  point against a perturbation of it.
- **Solution terms must come from task-relevant fields.** F1's eval arm drops
  contextual and metadata fields, so a term sourced from one appears in some arms
  and not others and trips the answer-leakage audit. `TaskSpec.validate()` rejects
  this.
- **System prompts must not use `{placeholder}` templating.** The §7 verification
  found ADK's eval-only request intercepter writes a UUID into session state,
  which an instruction interpolating that key picks up from turn 2. Static prompts
  are immune; `TaskSpec.validate()` enforces it.
- **Colab runtimes are recycled.** Long runs lose everything. Add checkpointing
  before any multi-hour run.

---

## 10. Open questions for the human

- **The behavioral equivalence margin.** The probe arm's TOST margin is in AUROC
  units and does not transfer to a difference in proportions. Before any "does
  not affect behavior" claim, someone must set the smallest shift in
  cautious-branch rate an eval designer would act on. A judgment call, not a
  computation.
- **N=30 may be too small.** The noise floor at layer 4 is 0.147 while the largest
  observed effect is 0.074. Either raise N toward the brief's preferred 40, or
  accept an underpowered representational result with honest power reporting.
- **Quantization.** All results read 8-bit activations. An fp16 replication on a
  larger GPU would remove the caveat.

---

## Framing discipline (from the brief)

This study does not resolve a confound and is not a decisive test. It separates
two previously entangled things and reports what it finds. A well-powered null is
a publishable result; an underpowered null is not. The axis that matters is
powered versus underpowered, not positive versus null.
