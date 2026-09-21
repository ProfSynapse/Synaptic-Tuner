# Research: google-research/envharness — integrate, borrow, or skip?

**Date**: 2026-09-21
**Repo assessed**: https://github.com/google-research/envharness @ `HEAD` (last commit 2026-08-20)
**Paper**: arXiv 2608.19880 — *EnvHarness: Awakening Static Worlds for Agent Learning*
**Context**: Second external-framework evaluation for Synaptic Tuner, after
[`higgsfield-framework-assessment.md`](./higgsfield-framework-assessment.md). Unlike that one,
this is live, well-built code worth reading carefully.

---

## Verdict

**Do not integrate. Borrow one specific mechanism, and read the paper.**

This is the opposite failure mode from higgsfield. higgsfield was dead code that solved a problem
we do not have. EnvHarness is *good, current* code that solves a problem adjacent to ours — but it
operates one layer above where we live, on a different environment interface, and its headline
results are about in-context skills rather than weights.

The borrowable mechanism is `orchestration/objectives.py` (~165 lines, zero dependencies): a
**difficulty-band controller** that reads recent rollout traces and emits a *direction* for the
next generation step. That idea maps directly onto SynthChat generation and the flywheel loop, and
we would reimplement it rather than take the dependency.

---

## 1. What it actually is

EnvHarness wraps a **frozen benchmark environment** in composable layers that reshape what the
agent sees and may do, without editing the benchmark's code or its verifier:

| Component | Reshapes | Mechanism |
|---|---|---|
| `Setup` | initial state (S0) | replays an action list through `inner.step()` before the policy starts (`harnesses/setup.py`) |
| `Rules` | action / transition / observation (A, T, O) | three pure-function hooks: `filter_action`, `modify_transition`, `filter_observation` (`harnesses/rules.py:92-105`) |
| `Link` | task composition | composes another environment's tasks in (`harnesses/link.py`, 564 lines) |

Everything speaks one interface, `ActionableEnv` (`core/actionable_env.py`) — a Gymnasium-shaped
`reset / step / observe / evaluate` ABC plus env-owned `save_state` / `from_state`. An `EnvHarness`
*is* an `ActionableEnv` wrapping another one, so layers stack by composition and an agent cannot
tell a bare benchmark from one wrapped in N layers. That decorator design is clean and correct.

An LLM **designer agent** (`agents/harness_agent.py`, 1,430 lines) runs
`propose → rollout K → decide → refine`, emitting a `_Rules(Rules)` **Python subclass as source
text**, which is compiled and loaded (`core/code_loader.py:68-105`). Failures come back as
`RulesCodeError` messages "shaped to be useful when fed back to the LLM for self-repair".

Bridges ship for six benchmarks: toy24, ALFWorld, WebArena, SWE-bench, OfficeQA, SpreadsheetBench.

## 2. Project health

| Signal | Value |
|---|---|
| Last commit | 2026-08-20 |
| Commits | **4** — a paper code drop, not an evolving project |
| Python LOC | ~28,240 across 125 files |
| Tests | 17 test modules under `tests/` (composition, hooks, code loader, objectives, persistence) |
| License | Apache-2.0 |
| Stars | ~580 |
| Core deps | `pydantic>=2`, `pyyaml`, `litellm>=1.50` — genuinely light |

The code is well documented and tested. But four commits over a month means the maintenance
commitment is a paper artifact. It is fresh rather than maintained; treat any dependency on it as a
vendored snapshot that will never receive a fix.

## 3. Why it does not plug into Synaptic Tuner

**It trains skills, not weights.** The core `envharness` package has **no training dependency** —
`grep` for torch/transformers/trl/peft hits only `infra/llm.py` and vendored third-party agent code.
The headline result (up to +9 points, ~9.8% fewer steps) is *skills learned in EnvHarness
environments* — reasoning-bank induction and in-context skill retrieval
(`reasoning_bank/induce.py`, `bank.py`, `embed.py`) against frontier models (`gpt-4.1`,
`vertex_ai/claude-sonnet-4-6`, `gemini/*`). No fine-tuned model comes out the other end. That is an
agent-harness product, not our product.

**The one path that does update weights is out of reach.** `rl/` drives GRPO through
[verl-agent](https://github.com/langfengQ/verl-agent), and per `rl/README.md` it needs a conda env
with **verl + vLLM 0.11 + flash-attn, 2 GPUs for the smoke and 8 for the full run**, and only
ALFWorld is wired (`rl/envharness_rl/alfworld/`). Our training is single-GPU by explicit gate
(`tuner/cloud/hardware_planner.py:431-442`) on an Unsloth/TRL stack. Integration is not a matter of
config; it is a different trainer on hardware we deliberately do not provision.

**Its environment interface does not match ours, in a way that is more than cosmetic.** Our
environment abstraction is `shared/environments/base.py:11-61`:

```python
class EnvironmentRuntime(ABC):
    setup(fixture) / teardown()
    mkdir / write_text / read_text / list_dir / move / copy / delete / exists / search
    snapshot(limit=200) -> Dict
```

There is **no `reset` / `step` / action space / observation space**. It is a POSIX-ish file-tree
API. The step-like object is `EnvironmentSession` (`shared/environments/validator.py:99`), whose
loop is `execute_response(completion_text) -> EnvironmentStepResult` then
`finalize(...) -> EnvironmentValidationResult`. Termination is a list of declarative assertions
(`SynthChat/schemas/environment_schema.py:33-120`: `path_exists`, `file_contains`,
`frontmatter_field_equals`, ...), not a reward signal.

Worth correcting a common assumption about this path: we do **not** use OpenEnv as an environment
abstraction. `Trainers/grpo/src/env_rollout.py:552-564` imports exactly one symbol from
`trl.experimental.openenv` — `generate_rollout_completions`, a vLLM sampling helper. TRL's actual
`environment_factory` integration is probed and deliberately unused
(`Trainers/grpo/src/env_runtime.py:46-84`, `:246-247`). So there is one environment abstraction
here, and it is ours.

That means adopting `ActionableEnv` is not "one more interface among several" — it is importing a
Gymnasium-shaped step loop into a codebase that deliberately does not have one, and writing an
adapter from `EnvironmentSession` up to it. Feasible; not free; and it buys the `Rules` hook points
(`filter_action` / `modify_transition` / `filter_observation`) that only pay off if we also adopt
the designer agent.

**The `_Rules` integration is patch-based where it touches a trainer.** `rl/integration/` ships
three `.patch` files against verl-agent pinned at commit `796ed31`, applied by a fetch script.
That is a maintenance liability by construction, and it is the pattern we would inherit.

## 4. Security note, if we ever did run designer-authored code

`core/code_loader.py:85` is a bare `exec(compiled, namespace)` — the comment says as much
(`# noqa: S102 -- intentional execution of LLM code`). The README's "isolated subprocess" is
`orchestration/episode_worker.py`, a stdin/stdout subprocess: that is **crash isolation, not
security isolation**. Designer-authored Python runs with the parent's user, filesystem, network and
environment variables.

For a research lab that is a reasonable trade. Against our rules — credential-free remote jobs,
secrets resolved by name at execution time, hash-pinned images — LLM-authored code in the training
loop would have to be confined to a Docker or Modal sandbox with no credentials mounted. We have
that infrastructure, but it is real work and it is not what the upstream design assumes.

## 5. What is worth borrowing

### 5.1 The difficulty-band controller (take this)

`orchestration/objectives.py` defines `MutationObjective.evaluate(recent_traces) -> ObjectiveSignal`
and two implementations. `DifficultyZone` (`:58-112`) holds the policy's success rate inside a
target band (default `(0.3, 0.7)`) over a rolling window, and returns:

- a **score** — distance from band center, normalized;
- a **diagnostic** — success rate, band, verdict, and the distribution of failure axes;
- a **suggestion_prompt** — "Increase difficulty." / "Decrease difficulty." / "Maintain. Prefer
  diversity over change.";
- **weights** over failure axes, computed as `1 / (recent_failures + 1)` normalized
  (`:116-123`), so axes with no recent failures get probed harder.

The design principle is stated in the module docstring and is the part worth internalizing:

> Objective only specifies DIRECTION, not concrete numerical mutations. The HarnessAgent LLM reads
> the suggestion_prompt and decides the actual values.

A deterministic, testable, cheap controller picks the direction; the expensive LLM picks the
content. That separation is what makes the loop auditable.

**Where it lands here.** This is a real gap, and the evidence is blunt: `grep -rn
"difficulty\|curriculum"` across `SynthChat/`, `Trainers/`, `shared/` and `Evaluator/` returns
**zero hits**. Concretely:

- Environments are LLM-generated **once**, offline, during SynthChat generation
  (`SynthChat/generator.py:1435 _generate_environment_spec`), gated and judged, then frozen into the
  dataset JSONL.
- At GRPO time the environment set is fixed. `Trainers/grpo/src/env_rollout.py:130-146` builds a
  prompt→`EpisodeSpec` registry once and replays `spec.environment_config` verbatim; the only
  stochasticity is sampling temperature. It even hard-raises on duplicate prompt keys (`:137`).
- `seed_count` / `rollouts_per_seed` in `SynthChat/targets.py:12-34` give seed multiplicity, which
  is variance reduction, not difficulty control.
- `shared/flywheel/experiment_loop.py` searches **hyperparameters** only (`_flatten_config` `:285`,
  `_merge_config_overrides` `:297`). No dataset or scenario dimension.

So nothing reads model performance and adjusts what gets generated. A `DifficultyZone`-style
controller over per-example eval outcomes is what closes that, and it is ~150 lines of our own code
plus plumbing into the existing generate → evaluate → analyze path.

### 5.2 Closed-vocabulary failure analysis (cheap, consistent with our conventions)

`core/types.py:154-159` types the diagnosis rather than leaving it prose:

```python
class FailureAnalysis(BaseModel):
    primary_axis: Optional[Literal["S0", "A", "O", "T", "R", "task_understanding", "none"]] = None
    label: str = ""        # short, free-form, e.g. "lost_localization"
    description: str = ""  # 1-2 sentence rationale
```

A closed axis vocabulary makes failures countable (`Counter(t.failure_analysis.primary_axis ...)`),
which is what makes the weighting in 5.1 possible at all. We already use closed vocabularies for
evidence levels (`LIVE_PROVEN` / `IMPLEMENTED_FAKE_TESTED` / `CONTRACT_ONLY` / `NOT_IMPLEMENTED`)
and Modal diagnostic codes, so this fits our existing discipline. Our judge output
(`Trainers/grpo/src/judge_reward.py`, `shared/flywheel/judge.py`) is the natural place for it.

### 5.3 We already own most of the loop machinery — note this before building anything

Two pieces of the repo are closer to EnvHarness's loop than EnvHarness is to us:

- **`shared/prompt_optimization/service.py`** already runs exactly this shape: `_run_evolutionary`
  (`:215-380`) with `population_size` / `elite_count` / `mutation_rate` / `crossover_rate`,
  `_apply_llm_rewrite` (`:584`) where an LLM mutates a genome segment, and fitness from the real
  Evaluator via `EvaluatorScoringAdapter` (`shared/prompt_optimization/evaluators.py:28`, `_score`
  `:96`). "LLM mutates artifact → Evaluator scores it → select and mutate again" is **running code
  here today**. The artifact is a system prompt. EnvHarness's contribution is the observation that
  the artifact could be the *environment*.
- **`shared/experiment_tracking/recommendation_engine.py:178-205`** already computes a weakness
  signal — `loss_spread` = top_loss / median_loss — and emits
  `{"dataset_review": {"action": "prioritize_high_loss_examples", "jsonl_hashes": [...]}}`. It is
  the only data-side recommendation in the repo, and **it has no consumer**. It names rows; nothing
  reads it.

That second point is the cheapest possible starting move: give `prioritize_high_loss_examples` a
consumer before building a new controller on top of it.

### 5.4 Baseline-then-candidate rollout protocol (already half-present)

The orchestrator measures a K-rollout baseline per task, caches it
(`orchestration/baseline_cache.py`), then measures the candidate on the same task with the same K
before accepting (`orchestrator.py:783-1048`). Our `compare-runs` / `compute-losses` path does the
analogous thing at run granularity; the per-task, cached-baseline framing is a useful refinement if
we build 5.1.

## 6. What to skip

- `bridges/` — ALFWorld, WebArena (browsergym), SWE-bench (docker), SpreadsheetBench. ~5,200 LOC of
  benchmark plumbing for domains we do not train on.
- `reasoning_bank/` and `third_party/` — in-context skill induction. Adjacent product, not ours.
- `rl/` — verl-agent, multi-GPU, patch-based, ALFWorld-only.
- `core/actionable_env.py` as an interface — we have two already.

## 7. Summary table

| Dimension | EnvHarness | Synaptic Tuner today |
|---|---|---|
| Output | In-context skills (+ a verl RL branch) | Fine-tuned weights (LoRA/SFT/KTO/DPO/GRPO) |
| Models targeted | Frontier APIs via litellm | Small open models via Unsloth |
| Env interface | `ActionableEnv` (reset/step ABC) | `EnvironmentRuntime` (file API, no step loop) |
| Env domains | 6 benchmarks (ALFWorld, WebArena, SWE-bench, ...) | 1 (markdown workspace), local or E2B backend |
| GPU shape | 0 for the main loop; 2-8 for `rl/` | 1, by explicit gate |
| Generated code | LLM-authored `_Rules` via `exec` in a plain subprocess | None in the training loop |
| Loop | diagnose → mutate env → K rollouts → accept/refine | generate → train → evaluate → recommend |
| Difficulty control | `DifficultyZone`, target SR band | **absent** |
| Maintenance | 4 commits, paper drop | active |

---

## Recommendation

1. **Read the paper** (arXiv 2608.19880) — the co-evolution framing is the valuable part, and it is
   cheaper to absorb than the code.
2. **Give `prioritize_high_loss_examples` a consumer** (§5.3). Smallest move, uses a signal we
   already compute and currently throw away.
3. **Type the judge's failure output** with a closed axis vocabulary (§5.2). Small, fits existing
   conventions, and is the prerequisite for (4).
4. **Build a difficulty-band objective**, modelled on §5.1 but written against our eval outputs and
   reusing `EvaluatorScoringAdapter` as the fitness oracle so it reads as native rather than bolted
   on. Deterministic controller sets direction; the generator model fills in content.
5. **Do not vendor, submodule, or depend on** `envharness`.

### Where it would attach, if we go further than (4)

- **Generation-time seam (lower risk, good fit).** `SynthChat/generator.py:1435` already LLM-generates
  environments against `SynthChat/schemas/environment_schema.py` with gate and judge validation. A
  difficulty controller slots in *above* `SynthChat/targets.py`, steering scenario selection and
  parameterization. The assertion vocabulary (`environment_schema.py:33-120`) and the gate types
  are the mutation grammar to aim at. Output is new JSONL rows — no change to the trainer.
- **Train-time seam (higher leverage, much more work).** Making `build_prompt_registry`
  (`env_rollout.py:130`) dynamic — perturbing `EpisodeSpec.environment_config` between GRPO steps
  keyed on observed `env_passed` / `stop_reason` — is where a true online curriculum lives. Nothing
  supports it today, and the duplicate-prompt guard at `:137` means any mutator must emit distinct
  prompt strings.

### Open question

The sharper question this raises is not about EnvHarness at all: **is our GRPO environment path
where we want to invest?** `Trainers/grpo/src/` is ~3,650 lines with its own rollout bridge, an
isolated venv, deterministic reward shaping over ~18 knobs (`env_rewards.py:17-41`,
`Trainers/grpo/configs/env_config.yaml:112-141`) and token-faithful loss masking — and it serves
exactly **one** environment domain, a markdown workspace, on a `local` or `e2b` backend
(`shared/environments/validator.py:44-50`).

EnvHarness's evidence is that *environment design* is where the gains are, which would argue for
investing there rather than in more reward knobs. But its gains were measured on frontier-model
in-context skills, not on small-model fine-tuning, and that transfer is unproven. Worth deciding
deliberately rather than by accretion.

---

## Sources

- https://github.com/google-research/envharness (clone inspected directly)
- https://arxiv.org/abs/2608.19880
- https://envharness.com/
- https://github.com/langfengQ/verl-agent (the RL branch's upstream)
