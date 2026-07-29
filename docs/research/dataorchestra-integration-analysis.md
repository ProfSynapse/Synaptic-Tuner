# DataOrchestra Integration Analysis for Synaptic-Tuner

**Paper**: [DataOrchestra: Learning to Orchestrate Per-Example Curation of Pretraining Data](https://arxiv.org/abs/2607.24717) (arXiv:2607.24717)
**Authors**: Zhen Huang, Yikun Wang, Shijie Xia, Pengfei Liu
**Date of analysis**: 2026-07-29

---

## 0. Source Access Caveat (read first)

**The full text could not be retrieved in this environment.** `arxiv.org`,
`huggingface.co`, `alphaxiv.org`, and `papers.cool` are all blocked by the
session's egress policy (HTTP 403 from the agent proxy). Everything below is
built from the abstract as surfaced by web search plus the paper listing.

Two consequences:

1. **Method details are unavailable**: how the orchestrator is *trained*
   (supervision signal, reward, rejection sampling vs. RL), the ablation table,
   per-benchmark numbers, and the size of the orchestrator model. Section 5
   ("learning the policy") is therefore explicitly deferred rather than specified.
2. **The framing in the intake blurb is slightly off from the abstract.** The
   blurb described four actions — "drop, keep, clean, or rewrite" — and claimed
   the paper "cuts processing compute". The retrieved abstract describes
   **three** top-level actions (drop / untouch / clean), with *rewriting nested
   underneath clean* as one of several downstream operations, and does not
   state a compute-savings figure in the text retrieved. The compute argument is
   an inference from the architecture (cheap router, expensive tools invoked
   selectively), not a quoted result. Treat compute savings as a hypothesis to
   measure, not a reported finding.

Anyone with arXiv access should re-read §method and §ablations before acting on
Section 5. Sections 3 and 4 stand on their own — they are about this repo's
architecture, not the paper's numbers.

---

## 1. Executive Summary

DataOrchestra's transferable idea is **one unified per-example action space**
instead of a fixed corpus-level pipeline: for each chunk, a router decides
drop / untouch / clean, and for "clean" it selects which downstream operations
to run and generates a concrete instruction for each.

Synaptic-Tuner already has both halves of this — but in two different
subsystems that cannot see each other:

- **`shared/flywheel` can route but cannot repair.** `AutoTagger` assigns
  `sft` / `kto` / `grpo` / `discard`. There is no repair action; a mediocre
  example is demoted to a KTO negative or thrown away.
- **`SynthChat` improve mode can repair but cannot route.** `ImprovementEngine`
  runs judge→improve against a fixed `default_rubrics` list for *every*
  selected example. There is no per-example decision to skip work, and no drop.

The single highest-value change is to close that gap: give the flywheel a
`repair` action wired to the improvement engine, and give the improvement
engine a cheap routing pre-pass. Everything else in the paper is either already
present or does not transfer to LoRA-scale fine-tuning.

**Recommendation**: adopt the action-space unification (R1–R3). Defer the
learned orchestrator (R4). Do not treat the pretraining benchmark gains as
predictive of anything here.

---

## 2. What DataOrchestra Does

### The Problem It Solves

Pretraining data pipelines pick a strategy at the **corpus or domain level**
and apply it uniformly. Every document in a bucket gets the same filter, the
same dedup, the same rewrite prompt — regardless of what is actually wrong with
that particular document.

### Core Insight

The right processing operation is a **property of the example, not of the
corpus**. Some chunks are fine untouched. Some are unsalvageable. Many are
salvageable but need a *specific* fix, and applying the generic fix to all of
them wastes compute on the healthy ones and under-treats the sick ones.

### The Action Space

Given a chunk, the orchestrator emits:

| Action | Meaning |
|---|---|
| `drop` | Discard the chunk entirely |
| `untouch` | Keep as-is, no processing |
| `clean` | Route to one or more downstream operations |

For `clean`, the orchestrator additionally:

- selects **one or more** downstream operations, spanning programmatic editing
  through several distinct forms of LLM-based rewriting, and
- **generates a concrete instruction** for each rewriting step, which a
  downstream tool model executes.

So it is not a classifier over a fixed op set — it composes a per-example
pipeline and writes the prompt for each stage of it.

### Reported Results

Models pretrained from scratch at **0.5B–7B** on web data processed by
DataOrchestra show **stable average gains over individual data-processing
methods across 11 benchmarks**. Per-benchmark numbers and ablations were not
retrievable (see §0).

---

## 3. Synaptic-Tuner Current State

Three curation surfaces exist today. None of them spans the full action space.

### 3.1 Flywheel `AutoTagger` — routing without repair

`shared/flywheel/tagger.py`. Pipeline position: `ingest → clean → tag → stage`
(`shared/flywheel/orchestrator.py:6`).

Rule-based bands from `FlywheelConfig` (`shared/flywheel/config.py:92-108`):

```
score >= sft_threshold (0.8)          -> "sft"
kto_min_threshold (0.3) <= score < 0.8 -> "kto"   (negative example)
score < kto_min_threshold (0.3)        -> "discard"
ambiguous_min..ambiguous_max (0.4-0.7) -> escalate to LLM judge
tools_requested == False               -> text_response_policy (sft|kto|skip)
```

This is genuinely a per-example router with cheap-first / expensive-fallback
structure — the rule path handles the clear cases and the LLM judge is only
invoked on the ambiguous band. That is already the compute discipline
DataOrchestra advocates.

**What is missing is the `clean` action.** Every path terminates in keep,
demote, or discard. An example that is 80% right and fails one rubric is
recorded as a KTO negative forever.

### 3.2 SynthChat improve mode — repair without routing

`SynthChat/modes/improve.py` + `SynthChat/engine.py:53`.

`ImprovementEngine.run()` walks `scope_processing_order`
(system_prompt → thinking → response) and for each scope runs up to
`max_iterations` (default **10**, `SynthChat/config/settings.yaml:48`) of
validate → judge → improve → apply. Rubrics come from `default_rubrics`, a
**corpus-level constant** (settings.yaml:50-54).

Selection of *which examples* to process is `--lines` / `--start-line` /
`--end-line` — i.e. manual line ranges (`_select_examples`,
`SynthChat/modes/improve.py:65`). There is no quality-derived selection.

Consequences:

- The LLM judge is invoked at least once per scope per example, even for
  examples that need nothing. The engine short-circuits *after* the first judge
  call passes, not before it.
- `on_max_iterations: skip | fail | save_partial` has **no `drop` option** —
  there is no way to say "this example is unsalvageable, discard it and record
  why".
- Every example gets the same four rubrics regardless of what is wrong with it.

This is precisely the "fixed strategy applied uniformly" pattern the paper
criticizes, one level down the stack.

### 3.3 Transcript distiller — keep/drop funnel

`.skills/transcript-distillation/scripts/distill.py`. A quality funnel
(`drop_tools`, `drop_skills`, `drop_prompt_markers`) followed by outcome-based
tiering into gold/silver and score tiers good/borderline/bad. Pure keep/drop —
no repair action, and dropped rows are counted but not recovered.

### 3.4 The wiring is already 90% there

The most actionable finding in this analysis:

**`RubricDef` already carries an `improver_prompt`, and the flywheel already
loads it — but never invokes it.**

- `shared/judge/models.py:65` — `improver_prompt: Optional[str] = None`,
  documented at line 27 as *"Prompt template for improvement (SynthChat only,
  None for Evaluator)"*.
- `shared/flywheel/judge.py:233` — the flywheel's rubric loader reads
  `improver_prompt` off the rubric dict into the `RubricDef`.
- Nothing in `shared/flywheel/` ever calls it. `grep -rn "improve\|rewrite\|repair"
  shared/flywheel/*.py` returns only unrelated matches.

Meanwhile `FlywheelJudge.judge_record()` already returns per-rubric scores plus
a `verdict_rationale` (`shared/flywheel/judge.py:20-30`). That rationale is
functionally the same object as DataOrchestra's "concrete instruction generated
for each rewriting step" — a per-example, per-defect natural-language
description of what is wrong. It is currently written to metadata and discarded.

---

## 4. Integration Opportunities

### R1 — Add a `repair` action to the flywheel (High value, Medium effort)

**The change**: a new pipeline stage between `tag` and `stage`. Logs tagged
`kto` whose score falls in a configurable repair band are sent to the
improvement engine with the judge's `verdict_rationale` as the instruction and
the rubric's `improver_prompt` as the template, then **re-scored** by
`FitnessEvaluator`.

**Outcome per repaired record**:

- Re-score clears `sft_threshold` → promote to `sft`.
- Re-score does not clear → keep the existing `kto` tag. No regression.

**Why this is the highest-value item**: it converts the middle band from a
liability into two assets. Today a 0.55-score record produces one KTO negative.
After repair it produces an SFT positive *and* the original is a natural KTO
negative — and crucially, an aligned (rejected, chosen) pair from the same
prompt, which is a strictly better preference pair than the unpaired negatives
the stager currently interleaves.

**Config sketch** (`FlywheelConfig`, mirroring existing knob style):

```yaml
repair_enabled: true
repair_band: [0.4, 0.8]        # defaults to [ambiguous_min, sft_threshold]
repair_max_attempts: 2          # NOT SynthChat's 10 - see R2 on cost
repair_ops: [programmatic, llm_rewrite]
```

**Non-negotiable provenance requirement**: repaired content is model-generated
and must be ablatable. Set `tag_source="repair"` (the field already exists,
`TaggedExample.tag_source`, `shared/flywheel/tagger.py:37`), persist the
pre-repair content in the catalog, and record which ops ran. Without this you
cannot later answer "did repaired data help or did it just launder the model's
own errors back into training" — which is the main risk of this whole approach.

**Blast radius**: `tagger.py`, `config.py`, `orchestrator.py` (one stage),
catalog schema (one column + one metadata field). `stager.py` gets the paired
KTO construction. No changes to trainers.

### R2 — Route before you rewrite in SynthChat (High value, Low effort)

**The change**: run the free programmatic validators as a pre-pass and only
escalate failing scopes to the LLM judge.

`ValidationService.validate_example()` already exists and already runs — but it
runs *inside* the iteration loop, feeding the judge prompt
(`SynthChat/engine.py:195`), not as a gate in front of it. Hoisting it is a
small refactor with a direct compute payoff: scopes with no validator failures
and no cheap-signal defects skip the judge call entirely.

This is the half of DataOrchestra that is nearly free here, because
`validate` already exists as a standalone mode (`SynthChat/modes/validate.py`).

**Also add `drop` to `on_max_iterations`.** Currently `skip | fail |
save_partial`; none of them expresses "unsalvageable". Add `drop` plus a
drop-reason ledger so the discard decision is auditable rather than a silently
passed-through original (which is what `skip` does today — see
`improve.py`, the exception path appends the unmodified example).

**Cost note that motivates R1's `repair_max_attempts: 2`**: SynthChat's default
is 10 iterations × 4 rubrics × every example. That budget is defensible for
offline dataset construction; it is not defensible inside a flywheel cycle that
is supposed to keep up with live inference logs. The flywheel repair stage
should use a much tighter budget than SynthChat's default.

### R3 — Make the op set per-example, not fixed (Medium value, Medium effort)

DataOrchestra's `clean` selects **one or more** ops spanning programmatic
editing through several distinct LLM rewrite forms. Synaptic-Tuner has the ops
— they are just not in one action space:

| Op | Where it lives now | Kind |
|---|---|---|
| Privacy/PII scrubbing | `SynthChat/modes/sanitize.py`, `PrivacyPreprocessor` | programmatic |
| Tool-schema repair | `Tools/audit_tool_schemas.py` | programmatic |
| Scope reformatting | `SynthChat/services/parsing/scope_extractor.py` | programmatic |
| Rubric-targeted rewrite | `ImprovementService.improve()` | LLM |

**The change**: declare ops in config, each with a **trigger predicate keyed to
a validator failure class**, so op selection is *derived from the diagnosis*
rather than fixed. This is the cheap, deterministic version of DataOrchestra's
learned op selection and it keeps the repo's config-driven discipline intact
(CLAUDE.md: "NO HARDCODING for specific scenarios").

Note the flywheel's `PIIDetector` is currently a no-op stub
(`NoOpPIIDetector`, `shared/flywheel/cleaner.py:51`) with the interface already
defined — it is a natural first programmatic op to make real.

### R4 — Learning the orchestrator (Defer)

The paper *trains* the orchestrator. Do not attempt this yet:

- **The supervision signal does not transfer at this scale.** DataOrchestra
  evaluates curation policies by pretraining 0.5B–7B models from scratch and
  reading 11 benchmarks. Synaptic-Tuner runs LoRA fine-tunes on datasets orders
  of magnitude smaller, where the downstream eval delta attributable to any one
  curation decision is well inside the noise floor.
- **We do not know how they trained it** (see §0) — the method section was not
  retrievable.

**What to do instead**: make R1–R3 rule-based and judge-based, and **log every
routing decision alongside the downstream eval delta**. The catalog already
persists `tag`, `tag_source`, and `fitness_score` per record, and
`shared/flywheel/experiment_loop.py` already closes the loop from dataset
version to eval score. That gives a (decision → outcome) corpus essentially for
free. Revisit learning the policy only once that corpus exists and shows
separable signal.

---

## 5. What Not to Take

- **The benchmark numbers.** From-scratch pretraining on web text at 0.5B–7B is
  a different regime from LoRA SFT/KTO on curated tool-calling data. "Stable
  average gains across 11 benchmarks" is not a promise that transfers. What
  transfers is the *shape of the decision*, not the effect size.
- **The compute-savings claim as stated in the intake blurb.** Not verified
  against the paper (§0). Measure it locally: instrument judge-call counts
  before/after R2 and report actual numbers rather than citing the paper.
- **Unbounded rewriting.** The failure mode of per-example repair at fine-tuning
  scale is training the model on its own laundered errors. Every repaired record
  must be provenance-tagged and independently ablatable, or R1 is a
  data-poisoning vector rather than an improvement.

---

## 6. Recommendation

Adopt in this order:

1. **R2 first** (lowest effort, immediate compute payoff, no data-quality risk):
   hoist validators to a routing gate in SynthChat, add `drop` to
   `on_max_iterations`.
2. **R1 next** (highest value): flywheel `repair` action with strict provenance
   and a tight attempt budget. Ship it behind `repair_enabled: false` and prove
   it on one dataset version with an A/B against the unrepaired baseline before
   defaulting it on.
3. **R3 when R1 is proven**: generalize the single rewrite op into a
   predicate-triggered op set.
4. **R4 not yet.** Accumulate (decision → outcome) pairs in the catalog first.

The framing to carry forward: Synaptic-Tuner's flywheel routes but cannot
repair, and SynthChat repairs but cannot route. DataOrchestra's contribution is
the observation that these are one decision, not two — and in this repo they are
already two subsystems that happen to load the same `improver_prompt` field.

---

## Sources

- [DataOrchestra: Learning to Orchestrate Per-Example Curation of Pretraining Data](https://arxiv.org/abs/2607.24717) — abstract only; full text blocked by egress policy (§0)
- [Cool Papers listing (cs.CL, cs.LG, cs.AI, cs.CV)](https://papers.cool/arxiv/cs.CL,cs.LG,cs.AI,cs.CV)

Repo evidence cited above:
`shared/flywheel/tagger.py`, `shared/flywheel/config.py`,
`shared/flywheel/judge.py`, `shared/flywheel/cleaner.py`,
`shared/flywheel/orchestrator.py`, `shared/judge/models.py`,
`SynthChat/engine.py`, `SynthChat/modes/improve.py`,
`SynthChat/config/settings.yaml`,
`.skills/transcript-distillation/scripts/distill.py`
