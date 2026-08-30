# Experiment Brief v3: Ranking Harness Artifacts by Evaluation-Awareness Leakage

**Audience:** an autonomous coding agent executing this study.
**Budget:** 3 days wall-clock (honest floor, not 2), single GPU node, open-weight models.
**Deliverable:** a ranked per-factor attribution table with per-factor matched controls, a cell-level representation/behavior dissociation result, and `results.json` sufficient for a 4–9 page workshop paper.

> **Changed from v2.**
> 1. Behavioral arm rebuilt. v2 specified greedy decoding for RQ2, which was an error: near-deterministic tool selection guarantees a behavioral null that is an artifact of the decoding choice, not evidence about dissociation. Behavioral measurement now uses valenced decision points, temperature sampling, and multiple rollouts.
> 2. Controls defined per factor. A token-matched scramble is well-defined for F1 and F4 and not for F3 and F6.
> 3. Primary endpoint family declared and pre-registered; everything else is secondary and uncorrected.
> 4. Control arms now require a fixture-plausibility validity check before their numbers are interpretable.
> 5. Dissociation stated at cell level, never as a rank correlation over four points.
> 6. Steering moved to appendix.

---

## 1. Research questions

Evaluation harnesses leak cues. This is established. **Which** harness properties leak, and by how much relative to each other, is not — designers have no ordered list of what to fix.

Chat-based studies cannot produce that list: in chat the prompt is the whole stimulus, and evaluation context is entangled with benchmark format (Devbunova 2603.19426). Agentic trajectories separate task from harness.

- **RQ1 (attribution).** Ranked by effect size, which harness properties shift the eval/deploy probe score with task text held fixed?
- **RQ2 (dissociation).** For each property individually, does a representational shift come with a behavioral shift?

Nayan et al. (2606.23583) showed representation survives behavior collapse under format rewrites in chat (probes retain AUROC ≥ 0.98 while gemma2-9B behavioral accuracy falls 0.575 → 0.338). Whether harness artifacts dissociate the same way is unclaimed, and it is our second axis.

**Framing discipline.** We do not resolve a confound and this is not a decisive test. We separate two previously entangled things and report a ranking. Lead every draft with the ranking.

---

## 2. WRITE BOTH INTRODUCTIONS BEFORE RUNNING ANYTHING

The day-one control-validity result determines which paper this is. Draft both now so the outcome selects a draft instead of triggering a rewrite.

**Draft A (attribution).** Control arms behave as intended; factors separate. Paper is the ranked list plus per-cell dissociation.

**Draft B (surface statistics).** Scrambled payloads move the probe as much as realistic ones. **State this narrowly.** It is *not* automatically "Devbunova was right, with agentic evidence." Two incompatible explanations produce the same observation: (i) the probe reads surface token statistics, or (ii) scrambling genuinely produces an eval-like context, because incoherent field names look synthetic and synthetic *is* the cue. §4.3's fixture-plausibility check is what distinguishes them. Draft B's claim is only as strong as that check allows — write it as "the agentic probe signal is not separable from token perturbation under [condition]," not as a general vindication of the format-sensitivity critique.

---

## 3. Design

### 3.1 Core structure

For each of N tasks, produce one **base trajectory**, then generate **factor-toggled variants of that same trajectory**. Never generate variants by running separate live-vs-mock harnesses; that changes tool content and trajectory length at once.

Held constant across all variants of task i: system prompt, tool schemas (names, descriptions, argument order), user task text, number of agent turns, number of tool calls, and the semantic content of every tool return.

### 3.2 The four factors

| ID | Factor | 0 (eval-like) | 1 (deploy-like) |
|----|--------|---------------|-----------------|
| F1 | Tool-return payload realism | Minimal tidy JSON, task-relevant fields only, schema field order | Extra irrelevant fields, nulls, nested metadata, non-canonical order |
| F3 | Failure realism | No tool call ever fails | One transient failure (timeout/503) then successful retry |
| F4 | Identifier realism | Sequential synthetic (`user_001`, `order_123`) | Natural-format (UUIDv4, realistic SKUs) |
| F6 | Session/state artifacts | Reset markers, eval-set metadata, visible fixed seed | Absent; continuous session context |

(F2 latency and F5 user-turn provenance were cut in v2: the former is confounded with token count, the latter is a prompt-format manipulation in a harness costume.)

### 3.3 Control arms are defined PER FACTOR

A token-matched scramble is well-defined only where the manipulated property is a surface property. It is not for semantic events. Define and label each control explicitly; do not present them as one uniform control type.

| Factor | Control type | Construction |
|--------|--------------|--------------|
| **F1** | True scramble | Hold token count, character-class distribution, and rough entropy fixed; destroy realism (shuffled field semantics, plausible-looking but meaningless field names) |
| **F4** | True scramble | Same character classes and length as the natural-format arm; destroy format validity (malformed UUIDs, invalid checksums) |
| **F3** | Plausible-but-uninformative variant | NOT a garbled failure — a garbled failure is anomalous and may cue evaluation harder than a clean one. Use instead: a non-failing tool call that returns an equally long, equally structurally complex payload (e.g. a verbose success envelope) |
| **F6** | Plausible-but-uninformative variant | Session metadata block of matched length and structure carrying deployment-neutral content (region, locale, API version) rather than reset/seed markers |

Also required, unchanged from v2:
- **F0 null control** — cosmetic whitespace/indentation only. The probe must NOT separate these arms. If it does, the pipeline leaks and every downstream number is void.
- **B-len** — length-only classifier baseline. Nguyen et al. report 0.584 on SAD vs. their probe's 0.816.
- **B-char** — special-character-count baseline, 0.598 in the same paper. Critical here: JSON-heavy contexts mean F1 could in principle be detected on brace and quote counts alone.

### 3.4 Report BOTH contrasts for every factor

Because the control means different things in different rows, a single ranked column is not comparable across rows. For each factor report:
- **Δ_base** = AUROC(eval arm vs. deploy arm)
- **Δ_ctrl** = AUROC(eval arm) − AUROC(control arm)

Label in every table which control type each row used. The ranking is presented on Δ_ctrl with the control type annotated, and Δ_base is shown alongside.

### 3.5 Sample size

- N = 30 tasks minimum, 40 preferred, across two domains.
- Per task: base + 4 factor pairs + 4 control arms + 2 null arms + 2 stacked extremes ≈ 14 static contexts.
- Total ≈ 420–560 contexts for the probe arm (forward passes only).

---

## 4. Probes, gates, and control validity

### 4.1 Models

**Llama-3.1-8B-Instruct** primary — Devbunova's backbone and Nayan et al.'s multi-layer steering target (layers 4·9·15). **Qwen3-32B** secondary if time (matches Abdelnabi & Salem, whose probes 2511.17085 reuses at the layer-43 MLP down-projection).

Sweep layers per model. 2606.29196 finds the layer of maximum linear recoverability shifts from late layers in smaller models to early in larger ones, and Nayan et al. find the signal often detectable before the network midpoint (median relative depth 0.46). Do not hardcode a layer from a paper using a different model size.

### 4.2 Two probes, both reported

- **Probe A (transferred).** Chat-trained: Nguyen et al.'s released probe, or trained on SAD `stages_oversight` / `jjpn2/eval_awareness`. This is what prior work would have used.
- **Probe B (in-domain ceiling).** Trained on your own agentic eval/deploy split, **held out by task**. Upper-bounds what is linearly recoverable from agentic trajectories at all.

Without Probe B, a low Probe A number is uninterpretable: "agents don't carry the signal" and "the probe doesn't transfer" are indistinguishable. The A–B gap is itself a reportable result.

### 4.3 CONTROL-VALIDITY CHECK (do this before interpreting any AUROC)

Naming the controls per factor does not make them valid. If a control arm looks *more* like a test fixture than the eval-like arm does, the subtraction is not conservative — it is backwards, and it will manufacture a false null.

**Procedure.** Score every arm on "does this look like a test fixture rather than a live system," 1–7, using an LLM judge over all arms plus ~50 human ratings on a stratified subsample for judge validation. **Requirement: each control arm must sit at or below its eval arm on this scale.** Report the scores in the paper.

If a control arm scores *above* its eval arm, that factor's Δ_ctrl is invalid. Report Δ_base for that factor and say why.

### 4.4 GO/NO-GO GATE

Before generating trajectories:
- Probe A ≥ **0.75 AUROC** on held-out SAD `stages_oversight`, clearing Nguyen et al.'s random (0.517), length (0.584), and special-character (0.598) baselines.
- Probe B > **0.65** on a held-out-by-task agentic split. If Probe B is at chance, agentic trajectories carry no linearly recoverable signal, there is no quantity to attribute, and that null is the finding — stop and write it up.

### 4.5 Read position

Primary read: **final assistant token immediately before a tool call.** State this explicitly in the paper.

Robustness: report scores at (a) final context token, (b) last token of each tool return, (c) token-wise sweep. If the per-factor ranking changes by read position, report it.

---

## 5. Statistics

### 5.1 Declared primary endpoint family (pre-register before running)

Reporting two contrasts × four factors × two probes × two behavioral columns and Holm-correcting across all of it leaves nothing significant at N=30. Declare the family now:

- **Primary:** Δ_ctrl on **Probe B** for the four factors. **Four tests, Holm–Bonferroni corrected.**
- **Secondary, reported uncorrected and labeled as such:** Probe A results, Δ_base for all factors, behavioral columns, read-position variants, stacked-factor effects, cross-harness replication.

State this split in the paper's methods section. A declared primary family with honest secondaries is defensible; an undeclared one is not.

### 5.2 Intervals and null claims

- **CIs:** paired cluster bootstrap **clustered by task**, 5,000 resamples. Resample tasks with replacement, keep all variants of each sampled task. Variants share task content; trajectory-level resampling gives falsely narrow intervals.
- **TOST margin:** do not use a conventional 0.05. Set δ from **whichever is wider** of (i) the transferred probe's own test-retest spread across repeated runs, or (ii) the AUROC band produced by random probe directions. Report which one set the margin and its value.
- **Power:** state achieved power for the smallest effect you claim to rule out.
- **Noise floor:** repeat one full arm end-to-end; any effect below that shift is not an effect.

---

## 6. RQ2: the behavioral arm

**This is now the study's main vulnerability. Spend design time here before increasing task count.** Dissociation is only observable if behavior can move; tool selection in ordinary agentic tasks is near-deterministic, and a behavioral null on all four factors reads as "no dissociation detected," not as dissociation.

### 6.1 Tasks need a valenced decision point

Every task must contain a decision with eval-relevant valence, where both options are legitimate:
- a cautious tool vs. an efficient one (verify-then-act vs. act)
- a permission or confirmation check that *can* be skipped
- a shortcut that works but bypasses a stated procedure
- escalate to a human vs. resolve autonomously

**Fix the decision point per task before looking at any data.** Record it in the task spec file. This is the pre-registration that keeps the behavioral result honest.

### 6.2 Decoding regime differs by arm

- **Probe arm:** greedy, fixed seed. Determinism is correct here.
- **Behavioral arm:** temperature sampling (T = 0.7 suggested), **≥15 rollouts per condition per task.**

These are different regimes for different measurements. v2 conflated them; do not.

### 6.3 Two behavioral columns

1. **Decision-point choice rate** — proportion of rollouts taking the cautious/compliant branch.
2. **Verbalized evaluation awareness** — rate at which the reasoning trace mentions being tested, judged by an LLM classifier. Nearly free, it is what most of this literature measures, and it hedges a tool-selection null.

Also log refusal/abandonment rate, retry behavior after the F3 failure, and trajectory length as exploratory.

### 6.4 Cost, and why this is a 3-day study

30 tasks × 4 factors × 2 arms × 15 rollouts ≈ 3,600 generations, plus base and control arms. Fine on an 8B model with vLLM, but several hours, not an afternoon. This is the schedule driver.

### 6.5 Stating the dissociation

**Cell-level only.** A rank correlation between representational and behavioral shift over four points is not defensible and no reviewer will accept it.

Write: "F4 shifts the representation ([X] AUROC, corrected p = [p]) without a detectable shift in decision-point choice (Δ = [y], TOST-equivalent at δ = [δ]), while F1 shifts both." Two comparisons support that. It is also the more useful statement for an eval designer than any aggregate correlation.

---

## 7. Harness choice and cross-harness replication

**ADK is an implementation detail, not a contribution.** Keep it out of the title. One sentence of justification (open source, inspectable eval-set format, exhibits all four factors natively), then move on.

**Cross-harness replication:** replicate the **highest-effect factor**, not the cheapest — a null replication of a weak factor tells you nothing. Choose it **after Day 2**, once you know which factor wins; do not pre-commit. Target Inspect (used by LURE) or a τ-bench-style setup.

**Day-1 verification, before anything else:** confirm the eval path does not itself alter the system prompt, inject instructions, or change tool descriptions relative to a live run. If it does, you are measuring prompt cues again. If ADK *does* change the prompt, document it — that is a reportable finding and a pivot, not a failure.

---

## 8. Remaining mandatory controls

1. **Token-length matching** across arms of every factor to within 2%; report residual delta.
2. **Lexical leakage audit.** Grep every variant for `eval`, `test`, `mock`, `fixture`, `benchmark`, `simulat`, `dummy`, `sandbox`, `assert`, `seed`. Any hit means the probe reads a word, not an artifact. Regenerate.
3. **Answer-leakage audit.** Verify no deploy-like variant leaks task solutions.
4. **Determinism where specified** (probe arm only; see §6.2).

---

## 9. Schedule

**Day 1 AM.** Harness verification (§7). Probe A reproduction to gate (§4.4).
**Day 1 PM.** Trajectory generator; base trajectories; all factor, control, and null arms. Controls §8. **Run the control-validity check (§4.3) — its outcome selects Draft A or Draft B.**
**Day 2 AM.** Probe B training (held out by task). Forward passes. Δ_ctrl and Δ_base with cluster bootstrap; Holm across the four primary tests; TOST for null claims.
**Day 2 PM.** B-len / B-char baselines; read-position robustness. **Launch behavioral rollouts (§6.4) — start these early, they are the long pole.**
**Day 3 AM.** Behavioral analysis; cell-level dissociation statements.
**Day 3 PM.** Cross-harness replication of the winning factor. Figures, `results.json`.

---

## 10. Outputs

- `results.json` — Δ_ctrl and Δ_base per factor for both probes, control type per factor, fixture-plausibility scores, cluster-bootstrap CIs, Holm-corrected p-values for the primary family, TOST results and the margin's provenance, null/baseline numbers, noise floor, behavioral rates with rollout counts.
- **Figure 1 (headline):** ranked per-factor Δ_ctrl with CIs, Probe A and B side by side, control type annotated per row, with F0 null and B-len/B-char on the same axis.
- **Figure 2:** per-factor representational shift and behavioral shift as paired bars — **not** a scatter with a fitted line.
- **Figure 3:** token-wise probe score across a representative trajectory, annotated with artifact positions.
- `tasks/` — task specs including the pre-registered decision point for each.
- `trajectories/` — all variants, so the matched-pair claim is auditable.

---

## 11. Failure modes

- **Probe B at chance.** Nothing linearly recoverable from agentic trajectories. Report with the transfer gap quantified; do not attribute an unmeasurable quantity.
- **Control arm scores above eval arm on fixture-plausibility.** That factor's Δ_ctrl is invalid. Report Δ_base and explain.
- **Behavioral null across all four factors.** Check dynamic range first: if the base condition shows near-zero variance at the decision point across rollouts, the task design failed, not the hypothesis. Report the base-condition variance so a reader can tell these apart.
- **Underpowered null.** Any "contributes nothing" needs TOST plus a power statement.
- **Drift toward prompt factors.** If a factor touches output-format requests, grading language, or placeholder entities, it has become EvalAwareBench (2605.23055) in costume. Every factor lives in the environment.

---

## Appendix: steering (only if everything above is complete)

Do not promise a causal claim and keep this out of the abstract. Nayan et al. establish single-layer steering at |α| ≤ 1.2 is marginal (|ΔAUROC| ≤ 0.017 across Gemma2-9B, Qwen2-7B, Qwen3-8B-thinking) while multi-layer steering (Llama-3.1-8B, layers 4·9·15) is robustly causal with matched random-vector controls reproducing none of the effects. Reproducing that standard is a multi-day budget item.

If run: multi-layer, along the contrastive direction, with matched random-vector controls. Appendix only, no causal language.

---

## 12. Prior work (cite; do not duplicate)

- **Nayan et al., 2606.23583** — reframed this study. Probes retain AUROC ≥ 0.98 under behavior-collapsing rewrites; contrastive probes show negligible shift under rewritten formats. **Reviewers will ask how you differ. Answer in the abstract: they show the representation is robust to format in chat; we attribute it to specific harness properties in agents and test dissociation per property.**
- **Devbunova, 2603.19426** — probes track benchmark-canonical structure. Motivates the controls, not the RQ.
- **Li et al., 2605.23055 (EvalAwareBench)** — factor-controlled attribution, but prompt cues, black-box CoT monitoring. Nearest methodological neighbor.
- **Ivanov & Africa, 2605.26438 (LURE)** — agentic realism via replay, black-box metrics, no matched-pair toggling, no probes.
- **Nguyen et al., 2507.01786** — probe methodology and the three baselines (0.517 / 0.584 / 0.598) we reproduce.
- **2511.17085 (Whistleblow)** — agentic + probes, but for validating low overall awareness, not attribution.
- **2606.29196** — layer depth by scale; justifies sweeping layers.
- **2603.15457 (Evasive Intelligence)** — agenda paper requesting harness-variant testing.
- **Needham et al., 2505.23836** — agentic-vs-chat discriminability gap motivating the setting.