# Native MLX GRPO candidate

Status: CPU review candidate, 2026-10-27. No Apple execution or hardware
qualification is claimed. The existing qualified SFT environment stays intact.

## Source decision

Read-only source audit pins (public repositories cloned without executing code):

* [mlxrl 5ee8a6bd](https://github.com/inductiveML/mlxrl/tree/5ee8a6bd7764b0585094d78c530ac87d26364034),
  version 0.2.4, MIT, Copyright (c) 2026 mlxrl contributors. Its
  `mlxrl/algo/grpo.py` SHA256 is
  `e58a0b7002d3c398c0cf06cd9e5df74b65285b89c2731bba44149c113a7949bd`;
  `policy/logprobs.py` is
  `69503a9ea9847f10a2a17578623c2fc7cd4e760c8c770e003c4053adcb5dbca0`;
  `rollout/naive.py` is
  `40f582ca8a617115af6fd1dfffd1624c741ce9a46dfacb38ead8f4e0d737d7d2`.
  The package declares Python >=3.11 and platform-marked MLX >=0.32.2,
  MLX-LM >=0.32.0. Those lower bounds are not an admitted closure.
* [mlx-tune 9690fe1f](https://github.com/ARahim3/mlx-tune/tree/9690fe1fed19c72f0810902b9c188d9a2625eb5b),
  version 0.6.0, Apache-2.0. `mlx_tune/rl_trainers.py` SHA256 is
  `4e6eb860c16083d3c0f642a264b7efc0c671cf09d63e2fc65950d3e325e3978c`;
  `losses.py` is
  `31762d5fb71e0b2a94eef902ea4cdf18e4f4bedb8ad815aefb43547739739779`.
  The native `GRPOTrainer` calls `compute_log_probs_with_lengths` over full
  generated sequences without a prompt boundary in its gradient phase. It
  stores beta but that phase has no reference KL term. Zero-variance groups
  skip optimizer updates. Its subprocess fallback trains SFT on supplied
  answers and reuses training data for validation. These behaviors cannot
  satisfy our public controls or completion-only GRPO contract. Its wider
  dependency set also includes vision/embedding/dataset packages. No code
  from this implementation is selected.

Select a small attributed adaptation of mlxrl's group-normalization,
token-ratio and KL objective, rather than its whole trainer/CLI. Its original
tests include tiny hand-computed losses and masked-gradient controls; they are
source evidence, not locally executed acceptance. Our independent CPU checks
use scalar probability calculations and analytic gradients. Actual MLX
autograd checks remain a separate Mac gate.

The selected objective uses population variance plus 1e-8 inside the square
root, detached old-policy/reference probabilities and detached group advantages,
and a mean over generated tokens. It is explicitly **unclipped**, with exactly
one update per rollout batch. It is not advertised as TRL-equivalent PPO, GSPO,
GiGPO, Dr.GRPO or multi-turn environment RL. Unit temperature is required so
sampling and differentiated policy agree. The upstream naive sampler supports
filtered/temperature sampling while retaining raw model log probabilities; we
do not admit that mismatch. Upstream model-class fast paths, cache cloning,
reference cache forwards and implicit template fallback are not selected.

## Integrated boundaries

`compile_native_grpo` consumes the existing public GRPO request and prepared
`synaptic-grpo-prompts/v1` identity, recompiles all controls, and binds a distinct
`NativeMLXGRPOProfile`. The existing native planner, ProviderFamilyV1, command
catalog, grant, guardian and synchronous owner-thread stores remain the effect
boundary. No accepting qualification authority or new public verb is supplied.
The child remains `-I -S -B` with captured-source validation before imports.

The child reads exact retained publication bytes, selects only the configured
prompt column, renders messages with explicit template arguments, and refuses
overlength prompts rather than truncating. Other row fields, including answer
metadata, cannot reach generation or rewards. The raw-response regex reward
documents come from the public contract; no arbitrary Python callbacks or tools
are imported. The profile explicitly selects the pinned `regex` package's V0
engine and a per-search timeout, with a compiled 10-second aggregate worst-case
search budget for each complete evidence verification. This is a declared native
reward dialect, not a claim of all Python/TRL regex edge-case equivalence.
The package already exists in the unchanged 54-package closure; host verification
also needs it. Host review still owns semantic provenance and prompt leakage.

Fresh sequential KV caches retain sampled token IDs through EOS. Full causal
forwards gather only generated positions; prompts and padding have no loss.
Old and adapter-disabled base-reference log probabilities are materialized and
detached before differentiation. Dropout, accumulation, warmup, mixed precision,
intermediate checkpoints and non-unit temperature reject. Public grad clipping,
constant AdamW, seed, batch/group sizes, logging interval and finite generation,
reward-call and evidence-byte budgets are explicit. The profile binds model
quantization; no launcher model map or download is introduced.

Verification requires the actual optimizer count, complete grouped rollout
evidence, independently recomputed rewards/group variation and saved numeric
parameter delta, exact configured adapter inventory, and finite loss computed
after loading the final saved adapter into a fresh base. Constant rewards or
unchanged parameters cannot establish successful native qualification. Five
portable native roles retain input, reward, runtime and source bindings. Final
replay loss on sampled rollouts is a consistency check, not held-out quality.

## Qualification proposal and remaining gates

Use a new private GRPO runtime and named candidate; do not modify/reuse the
qualified SFT environment. The candidate closure keeps the reviewed 54 exact
package pins and adds only captured engine source (the attributed objective
needs no new installed distribution). Full wheel archive provenance, installed
bytes/native library/OS capture, final pushed source, independent review and
root-signed fresh one-shot admission are still required before ML imports.

First run independent tiny MLX forward/autograd/detach/mask checks under bounded
supervision, without model loading. Then propose a separately admitted small
Qwen3-0.6B 4-bit attempt, short prompts, group size >=2, a reviewed response
reward with no guaranteed variation, finite steps/token/reward/time/memory/file
bounds, and fresh output/state. Fit on the M4/24 GiB is unproven. Preserve any
constant-reward or consumed failure; no automatic retry. Root owns hardware,
model/runtime preparation, admissions, integration review and all Mac effects.
Public logs/cancel/reconcile remain unavailable. CPU doubles prove controls,
not MLX execution, hardware fit, training quality or general success.

## CPU review evidence at initial candidate 57245717

The native regression lane passed 266 tests with four explicit Apple numerical
skips. After tightening quantization inventory, index types and float32 reward
variation and zero-beta KL overflow, the complete GRPO and restricted-host lane
passed 63 tests. The
independent objective check compares probability-space scalar loss and analytic
derivatives against finite differences, including unequal generated lengths,
constant groups, and ignored positions. A separate real-MLX autograd test is
checked in but has not run on this Linux machine.

The existing baseline compiler at merged main
`3e357a2f0d1d85bdf06c67274946d7a9846ff742` and the candidate produce identical
canonical raw-SFT and prompt/completion-SFT workload bytes for the retained
CPU fixtures (2676 and 2744 bytes). Both original SFT dependency trees are
unchanged; the new 54-package lock copy is byte-identical (SHA256
`7c167018ee6bd61a3609971e3f834082c798e7bd2a3fa0c5c2564887da4fde4f`).
All six existing closure/lock maintainers report CURRENT without a write or
inventory/pin change. Ruff, diff checks and canonical skill-tree sync pass.

The sdist-to-wheel build and neutral base-only installed consumer pass without
ML/provider packages, including import of the optional native composition.
All 17 native-MLX/public source modules match both distribution archives,
including the complete embedded upstream MIT notice. Package equivalence is
source/package evidence, not a captured Mac execution runtime.
Private command records and XML/package/canonical-byte checks are retained at
`~/.cache/synaptic-tuner-release-proof/resume-20261027/grpo-integration`.

## R1 P2 response — bounded evidence publication

Independent review R1 reproduced an admitted 1,386,552-byte evidence document
that passed GRPO validation but exceeded the exclusive writer's fixed 1 MiB
write/readback limit. The fix adds an explicit validated `maximum` argument to
the generic writer, preserving its 1 MiB default and existing exclusive,
no-follow, 0600, fsync and same-tree readback behavior. The explicit limit may
not exceed the existing 512 MiB native per-file ceiling; the GRPO profile still
caps evidence at 16 MiB. Other control/SFT writes retain their original default.

The actual worker publication function uses the exact profile evidence limit
for writing and readback. Evidence is published before the other control
documents so an over-budget value creates none of them. This is not an atomic
multi-file transaction or replay authority after any failed consumed attempt.

CPU regression uses the exact reviewed 64-update, batch-four, 128-generated-token
fixture and full valid 1,386,552-byte canonical evidence. The real production
publication function and NativeTree round-trip it at both the admitted 16 MiB
ceiling and an exact-fit limit, then verify all five artifacts and the complete
large lineage. A one-byte-smaller admitted limit rejects before any document
write; default-limit, bounded-read, collision and invalid-limit negatives remain.
No MLX, model or training execution is needed for these artifact controls.
The fix passes 279 native CPU checks with four explicit Apple numerical skips,
including ten new writer/evidence boundary controls. All six existing closure
maintainers remain CURRENT without changing pins or inventories. Skills sync,
Ruff and diff checks pass; raw and prompt/completion SFT canonical hashes match
the independent R1 baseline exactly. Fresh source archives are checked without
installing packages; historical 57245717 installed-consumer evidence above is
preserved as historical rather than attributed to the new source.

## Closed native child diagnosis

The first separately admitted model attempt on source `19768026` failed with
worker exit 122, confirmed termination, no stdout and no initial output file.
Only the stderr byte count was retained. This locates the old `run_request`
boundary and establishes no tokenizer, import, model or adapter cause. A later
ordinary host tokenizer-only pass does not qualify the isolated GRPO path.
The pinned MLX-LM source contains both `LoRALinear` and `make_prompt_cache`;
missing-export speculation is not evidence. CPU recompilation and bound checks
did not establish a deterministic configuration defect.

Configuration cannot expose internal failure phases. The reusable child now
accepts an internal trusted milestone callback, passed only by child main for
GRPO and never selected by request/config data. Static worker exits 130–148
identify compilation through final custody; their exact boundary table is in
the canonical `native-local-training.md` reference. They contain no exception
text, traceback, path, argv, environment or prompt. They do not add success files,
change the public `native_worker_failed` result, or change SFT exits 120/121/122.
Bootstrap/capture-import failures retain 120/121 for both methods.

CPU controls inject actual operation failures throughout the existing pipeline,
check closed exits and empty diagnostic streams, and round-trip a synthetic
success through the unchanged seven output files and five-role public verifier.
The synthetic array/gradient backend is a control-flow double, not MLX training
or hardware qualification. Codes require the exact admitted source, GRPO method
and a confirmed worker-exited outcome; guardian codes belong to a different
boundary. A phase is not a cause, retry authority or retrospective diagnosis of
the consumed historical attempt. New source packaging/capture, independent
review and a fresh root admission remain prerequisites for any later execution.

## Explicit prompt-ID return contract

The exact pinned Transformers5.20.2 `apply_chat_template` source defaults to
`return_dict=True` and returns a mapping. Iterating that result yields keys,
which the original GRPO worker then rejects as noninteger IDs. Root's extracted
pinned-method CPU reproduction (rendering/encoding stubbed) confirms this API
mismatch; it does not establish the consumed historical122's cause.

GRPO now explicitly requests `return_dict=False` through `tokenize_prompts` in
the existing worker module. It preserves `tokenize=True`, the generation prompt,
and configured template kwargs; the existing public parser reserves these
renderer controls and the helper revalidates collisions. Only a nonempty list
of nonnegative exact integer IDs within the prompt bound is admitted; malformed
IDs fail in phase135 before ML imports. SFT and all dependency pins are unchanged.

The existing preparation example's separate config-first `profile-tokenizer`
action reuses that helper against the actual recompiled GRPO workload, verified
prepared publication and private tokenizer snapshot on Linux/Darwin. Its bounded
create-only private evidence records exact IDs and bindings, with a closed
counts/hash summary and no model/MLX/generation/training operation. CPU controls
use a realistic mapping-default tokenizer double, check unchanged scaffold/IDs,
reserved-kwargs negatives, malformed outputs before imports, private collision
and failure retention, source/snapshot custody and exact configured byte/version
bounds. This is ordinary tokenizer/control evidence; root still owns final
independent review, capture, admission and any actual model qualification.
