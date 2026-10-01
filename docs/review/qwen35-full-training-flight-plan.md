# Qwen 3.5 4B: full-training flight plan (not launch-ready)

Status: representative train/save/serve/chapter mechanics verified; human
writing-quality review remains pending. Full-epoch training remains held. The private
full-training recipe remains a candidate; no full run is submitted. The successful
two-step [same-job smoke](../../Trainers/recipes/qwen35_4b_modal_train_eval_smoke.yaml)
qualified the isolated SFT → saved LoRA → same verified base snapshot → vLLM →
evaluation → authenticated record/verified downloads workflow. Its three simple
cases did not qualify long-form writing quality.

## Representative rehearsal evidence (2026-09-30)

Pushed source `52d85db5a820cce43c00339c925cebc804b42a3e` completed the
two-step representative rehearsal on an NVIDIA L40S. Authenticated trainer
lineage reports 176 training and 44 evaluation examples, the pinned Qwen 3.5
4B revision, rank-32 LoRA and 32,768-token training context. Recorded training
time was 140.7 seconds, with `final_step: 2` and
`total_epochs: 0.09090909090909091`; this is not a full epoch. The public workflow
verified and downloaded all five required training artifacts, including the
saved adapter, and its authenticated same-job evaluation record.

The three representative chapter requests all failed before a valid response.
Their retained legacy `evaluation_error` labels and null response/latency do
not establish a timeout, server rejection, model-quality problem, or failed
content assertion. The original exception detail was discarded. A local
real-client/payload/assertion test with fake HTTP passed all three cases; that
does not qualify the live server. Keep the consumed attempt and its artifacts;
measure the live request boundary with reviewed closed diagnostics before
changing settings. At that checkpoint, writing quality and 32K concurrent chapter
completion were unproven. Full training, GGUF and publication remained held.

## Successful representative follow-through

The fresh ordinary rehearsal from pushed source
`9f8ed4302bedb3409e670eac78294c70833f72e7` passed end to end with the same
configuration/workload as the failed request attempt. Authenticated metrics
report two steps in 146.3 seconds. All five required training artifacts and
the signed evaluation record were verified and saved. The adapter archive is
169,922,560 bytes, below the existing 192 MiB per-artifact bound.

The three held-out chapter cases from different series passed their configured
nonempty-text and natural-stop checks, with no request errors. Saved replies
contain 730, 799 and 722 whitespace-delimited words; request latencies are
93.204771, 114.119370 and 92.746525 seconds. The authenticated workload confirms
32K context, three sequences, vLLM 0.26.0, bf16, thinking disabled, and no
request-level output-token ceiling. Actual decode settings are temperature 0.7
and top-p 0.9. The evaluator dispatches the cases concurrently; the retained
record does not measure their temporal overlap.

This is one successful representative rehearsal, not statistical reliability,
an explanation of the earlier discarded request errors, or a prose-quality
score. The new closed diagnostics did not change deadlines, retries, prompts,
model, data or concurrency; retain them to diagnose any recurrence. The author
still needs to read the saved drafts. No full epoch or GGUF was launched.

## Batch-capacity smoke: 8 examples at once

Joseph approved a more aggressive L40S smoke on 2026-09-30. The private
qualification recipe changed to batch 8 and accumulation 1, retaining two
optimizer steps, the complete 176/44 dataset, model/runtime pins, 32K context,
rank-32 LoRA, loss controls and same-job evaluation. The effective example
batch remains eight; padding and loss normalization can differ from batch 1
with accumulation 8. No full-epoch recipe was changed.

The earlier successful baseline's retained capacity profile reports 13.189 GiB
peak allocated and 14.498 GiB peak reserved out of 44.392 GiB visible to
PyTorch, leaving 29.894 GiB reserved headroom. These are allocator measurements
from a two-step run, not worst-case full-epoch or vLLM memory qualification.

Fresh attempt `modal-853d9958ca337b1ff90d3610`, from pushed source
`023e9ada8610b74163cb6978fddf38e1af0fc25a`, reached GPU submission after the
ordinary CPU gate but failed with `RUN_WORKER_SFT_TRAINER_CHILD_EXEC_OTHER`.
The exact bound read-only inspector independently returned
`WORKER_SFT_TRAINER_CHILD_EXEC_OTHER`. No new verified artifact set,
evaluation record, optimizer count, training timing or capacity profile was
obtained. The successful batch-1 run and all consumed attempts are preserved.

The label identifies an otherwise unclassified exception during child execution,
not a proven out-of-memory condition or a specific training milestone. The
classifier at that source/attempt recognized exact built-in RuntimeError/MemoryError
types; distinct library exception types could fall into OTHER. PyTorch's distinct
OutOfMemoryError is one compatible hypothesis, not evidence of this run's cause.
The original exception detail was not retained in a recoverable authenticated
diagnostic. Do not infer maximum safe batch size, reduce context, or replay
this attempt from that label.

The finite diagnostic improvement was implemented and independently reviewed:
876 provider-free tests passed; eight native sealed-descriptor tests remained
unexercised because the test interpreter lacks `os.memfd_create`. The child now
recognizes the already-loaded Torch OOM identity and preserves trusted finite
execution milestones for OOM and otherwise unknown library errors. Existing
codes remain accepted; no raw exception detail is transported.

The single approved fresh batch-8/accumulation-1 L40S smoke used pushed execution
source `f7ae5251325de28d8bd488d6f9f51ba82586f82a`, attempt
`modal-82ca1d46874b4eae914ff807`, submit
`becb752e9cc5f34e2941d279131b7576f80da264c7e7a203579bea333ac0f48e`, and call
`fc-01M3TEYN47EWK921PG3H0AP4P6`. It terminated with
`RUN_WORKER_SFT_TRAINER_CHILD_EXEC_TORCH_OOM_TRAIN_CALL`; the CLI reported
`retry_authorized: false`. The exact bound read-only inspector independently
matched `WORKER_SFT_TRAINER_CHILD_EXEC_TORCH_OOM_TRAIN_CALL` with
`authority: DIAGNOSTIC_ONLY`. This confirms the new diagnostic transport live:
Torch OOM identity and the trusted `TRAIN_CALL` milestone only. It establishes
neither device, allocation size/source, capacity limit, root cause nor completed
optimizer count. No new five-artifact verification, evaluation, metrics or
training timing was obtained. The earlier unknown-cause attempt and successful
batch-1/accumulation-8 baseline remain preserved.

Joseph separately approved one fresh batch-4/accumulation-2 L40S smoke with
`max_steps: 2`. It used pushed execution source
`cd97f2955d99830330f3501b0290a06088668504`, attempt
`modal-34879e001230749c84e9d714`, submit
`c6395199cddd456d03c4a17c86538c011f260d5671cfc4f9b09247b51c14fa1c`, and call
`fc-01M3TGBHTA9675SNG8MHM141MJ`. The ordinary CLI verified and saved all five
training artifacts plus the evaluation record, then exited 1 with
`evaluation_passed: false`. One exact bound inspector exited 0 with
`authority: DIAGNOSTIC_ONLY` and `result: PROVIDER_SUCCESS_UNKNOWN`; this is not
a training-failure diagnosis.

Saved training metrics and lineage agree on `final_step: 2`,
`total_epochs: 0.09090909090909091`, and 176/44 examples. Recorded training time
was 172.1 seconds, rounded to one decimal from `trainer.train()` wall time,
not whole-job elapsed time. The saved allocator profile reports peak allocated
27.093 GiB and peak reserved 29.922 GiB out of 47.374 GiB total, or 57.19% and
63.16% respectively. These `_gb` fields use a `1024**3` divisor; the separate
hardware figure of 50.9 decimal GB is not the same unit. The rounded values do
not establish raw-byte precision or capacity beyond this sampled two-step workload.

Evaluation passed two of three configured mechanics cases: case 0 in 80.649603
seconds and case 2 in 66.099006 seconds. Case 1 returned `request_timeout` at
120.007829 seconds with no response. This reached the then-existing 120-second
per-request runtime limit, which is not YAML-configurable. No OOM diagnostic
was reported. The two successful output texts remain inside the saved
evaluation JSON, not standalone chapter files; author quality review remains
pending. The failed evaluation gate does not undo training-artifact verification.
Attempt `modal-34879e001230749c84e9d714` has consumed its submission authority
and cannot be replayed. One separately approved fresh batch-4/accumulation-2
smoke remains pending fix review and release gates; it has not launched.
Full training, GGUF and publication remain held.

The narrow runtime correction removes that clamp and uses the remaining
existing evaluation deadline after startup and elapsed case work. It rejects
expired requests before I/O and late replies before assertions, retaining zero
retries, parallelism, request/response bounds and cleanup. No new timeout field
or recipe change is needed. The 66 provider-free evaluator/transport tests
passed; independent review and release qualification remain required.
Requests' timeout is connect/read inactivity, not a strict total HTTP timer;
slowly arriving bounded bodies can overrun the window. Requests drain before
normal runtime cleanup, which uses bounded TERM/KILL. Evaluation is synchronous
in the packaged worker; the configured 1800-second provider execution timeout
is the outer backstop, not proof that cleanup ran after forced termination.

## Proposed training shape

The private, gitignored candidate recipe is
`.tracking/evaluations/syntunia-fiction/chapter-review-training.yaml`;
the public smoke remains a reusable structural template. The existing
`train --job-config` handler accepts any resolved YAML path, and its Modal
planner constrains the *dataset locator*, not the recipe location. Copy the
exact pinned model revision, `qwen35-sft-v1`
profile, L40S × 1, `syntunia-sft-row/v2` dataset format, 32,768-token training
budget, non-4-bit loading, and LoRA
settings from the smoke. Keep `batch_size: 1`, `gradient_accumulation: 8`,
`learning_rate: 1.0e-4`, `r: 32`, `alpha: 64`, the seven named LoRA target
modules, and the prepared-message loss controls. The duration change is:

```yaml
training:
  batch_size: 1
  gradient_accumulation: 8
  learning_rate: 1.0e-4
  num_epochs: 1            # replaces max_steps: 2; never declare both
  packing: false
  completion_only_loss: true
  assistant_only_loss: false
  prompt_render: prompt_completion
  require_memory_efficient_loss: true
  chat_template_kwargs:
    enable_thinking: false
  save_steps: 10
  save_total_limit: 2
```

This expresses one pass through the pinned training split, not a quality
guarantee. At batch 1 and accumulation 8, 176 rows imply roughly 22 optimizer
updates, subject to the trainer's final-batch behavior; two smoke steps are not
an epoch. Preserve the prepared row contents and split identity. Do not add
YAML assistant targets, a system prompt, or a new context-ordering rule.

## Same-job evaluation decision

Retain `post_training.mode: same_job`, the exact vLLM 0.26.0 pin, LoRA rank 32,
`generation.max_tokens: null`, and `chat_template_kwargs.enable_thinking: false`
at serving. Null removes a request-level output-token ceiling, not the model
context, wall-clock, or response-size limits. Replace the smoke's greeting,
garden, and walking prompts with independently reviewed, private, representative
context-bundle → prose cases. Configure hard checks for nonempty final text
and natural `finish_reason: stop`; manually inspect visible thinking along with
the actual text for relevance, continuity, style, context fidelity, and unsupported claims.
The current evaluator has no semantic judge, so a 100% assertion pass is not a
chapter-quality score.

The Modal recipe accepts inline evaluation scenarios. A private gitignored
recipe can carry private cases without a new case-file loader: the compiler
includes and digests those exact bytes in the signed workload. They therefore
still travel to the authorized remote worker and retained workload artifact;
gitignored means **not committed to Git**, not secret from that execution path.
Do not copy private prose into a public recipe or use an unsupported scenario
path string.

The smoke served with `max_model_len: 4096` while training accepts 32,768 tokens.
Choose a serving context from measured representative prompt *and* desired
completion lengths, then qualify that context on the selected GPU before setting
the new recipe. Do not truncate the evaluation prompt to fit 4096 or silently
set 32768 on the strength of the short smoke. The same-job evaluator dispatches
independent cases concurrently through the single verified vLLM server and
adapter, bounded by the configured `vllm.max_num_seqs` and case count. This is
parallel serving after training, not another training job. Keep the authenticated
five training artifacts plus separate evaluation record.

## Measured and bounded capacity

- The retained successful attempt trained **two steps in 131.1 seconds**. It does
  not establish per-step throughput for the full 176-row epoch; preprocessing,
  startup, sequence lengths, and evaluation are not linear in step count.
- The smoke's whole-job timeout is **1800 seconds** and its operator maximum is
  **USD 2.00**. The Modal recipe loader permits a whole-job timeout of at most
  **3600 seconds**. The maximum is an operator GPU-estimate authorization, **not**
  a provider billing cap; build, CPU, memory, storage, and usage beyond timeout
  are excluded from that estimate. Requote the chosen duration and obtain an
  explicit cost/deadline decision before any paid submission.
- Current inline post-training config is at most **128 KiB**, with at most **16**
  messages per case and **32** cases. There is no separate per-message byte cap;
  the canonical config and the existing **1 MiB** HTTP request bound remain.
  Evaluation startup and total timeout are separately configurable; requests
  use the remaining evaluation window for HTTP inactivity and reply acceptance,
  subject to the deadline limitations above. Each retained response remains
  bounded at **64 KiB**. Large context bundles
  or chapters that exceed these limits cannot be truthfully tested by splitting,
  truncating, or calling `max_tokens: null` an unlimited-output mode.

Use three complete chapter requests from different series, with exact private
provenance and byte measurements in the evaluation README. Use validation rows
from the freshly rebuilt per-series split, not the earlier mixed train/validation
selection. This holds out later chapter targets in known series; it is not an
unseen-series generalization benchmark. Never relabel the old publication.
Retain complete original user contexts without assistant targets or a system
prompt. There is no judge, prompt-variation suite, or automatic prose-quality
verdict. The evaluation deadline and 64-KiB retained-response bound remain
measured risks; honoring the configured window does not qualify complete
generation beyond those limits.

## Bounded rehearsal before the epoch

The private `chapter-review-qualification.yaml` candidate changes only the
duration and job identity from the epoch recipe: `max_steps: 2` replaces
`num_epochs: 1`, with a 1800-second job timeout, 1200-second evaluation window,
and USD 2 operator maximum. It retains the verified 176/44 dataset, three
complete held-out prompts, 32K serving context, three concurrent sequences,
and null request output-token ceiling. Its provider-free configuration digest
is `25c5f88cfbd16acaf7721cf5cbc81794c95eb5df06632a5af22c5c429570834d`;
workload digest is
`58f39db517424d3eb437b9db00c342fdb19f44e49de228ee329f97473b96205f`.

The ordinary public `train` workflow qualifies the fresh packaged wheel on CPU
before GPU submission. A separate `--qualify` attempt would duplicate that gate.
The historical successful source eagerly tokenizes both complete dataset splits
before applying `max_steps`; its two-step run therefore admitted all 220 original
rows in the pinned training runtime. The rebuilt split preserves those messages
exactly. The next rehearsal naturally checks the new split in that same runtime;
no tokenizer-only cloud launcher or local training-image pull is required.

The current create-only training workflow provisions a fresh model-cache Volume
for each attempt. It downloads the pinned base for that attempt and reuses the
verified snapshot between training and vLLM evaluation. It does **not** adopt the
historical run's cache or guarantee no cross-run download. This limitation must
be disclosed before launching; do not silently add cache-adoption machinery.

A scoped read-only rate observation on 2026-09-30 returned L40S at USD 1.95/hour:
1800 seconds corresponds to about USD 0.98 for the GPU alone. Build, CPU, memory,
and storage can add cost. Neither this observation nor the recipe's operator
maximum is a provider billing cap or launch authorization. The epoch remains
held until the rehearsal and human review have been considered.

## Tokenizer evidence and release gate before a full run

The earlier tokenizer report binds the old dataset and an older packaged lock;
do not relabel that report as new admission evidence. The checked-in offline
profiler's environment uses Transformers 4.57.1, whereas the reviewed
`qwen35-sft-v1` training-image inventory pins Transformers 5.17.0. A refreshed
offline report must disclose that distinction even if all messages are unchanged.
Use actual pinned training-runtime admission before flight; an informational
profiler result is not proof of exact-image compatibility. Do not alter the
training-image pins to match an operator's local profiler.

1. Approve the one-epoch coverage and private evaluation cases, including the
   intended serving context and human writing-quality acceptance criteria.
2. Resolve a candidate recipe through the existing provider-free surface:
   `python tuner.py train --job-config .tracking/evaluations/syntunia-fiction/chapter-review-training.yaml --plan --json`.
   The private candidate is preparation, not launch approval. Verify exact
   model/dataset/profile pins and compiled workload digest.
3. Qualify the selected serving context and obtain a fresh scoped Modal rate
   observation with the existing `train --quote` surface; this is read-only but
   does contact the provider. Set the whole-job timeout and operator maximum
   only after reviewing that evidence and the intended evaluation duration.
4. Require clean pushed source, current source/lock checks, provider-free tests,
   a fresh host-owned grant, and explicit authorization for exactly one paid
   submission. An uncertain prior submission is never replay authority.

GGUF conversion and model publication remain out of scope.

## Representative split — approved local rebuild

Joseph approved representation from every series in both training and
validation. The original 181/39 whole-series split did not satisfy that: all
Shattered Crystal rows are validation and all other series are training.
The existing v2 `group_hash_rank` configuration cannot change this by weights
or seed alone. The existing preparation/publication surface now has an opt-in
`group_sequence_tail` policy; whole-group `group_hash_rank` behavior is unchanged.

The authoritative v2 declaration was recovered from
`.tracking/experiments/syntunia-context-production-v3/dataset-prep.json`.
Its target sequence metadata has ties, so whole equal-sequence cohorts stay
together rather than forcing a cut through them. The nearest 4:1 tail allocation
gives Endless Nights 41/10, Sandman Chronicles 63/16, Shattered Crystal 30/9,
and Symphony of Shadows 42/9 (train/validation), totaling **176/44**.
All 220 prompts, prose targets, row/context/target identities and document order
are preserved. The candidate has zero held-out target text or target-item
references in training contexts; seven validation contexts use earlier training
targets. Revision/derivative crossings fail explicitly. The artifact verifier
independently recomputes assignments from bounded ID-only lineage metadata.

This is late-chapter validation within known series, not a random sample of every
story phase. The existing `prepare-dataset` command with the private
`dataset-prep-80-20.json` declaration produced verified fresh publication
`dataset-0a4710dc910bbda61b06aa9fa1b9c45ff3b1fb5d7b535b747f4ada6e97facd50`.
The old 1320a280 publication is preserved. The private recipe and three
held-out review cases now bind the new verified identity, and provider-free
planning passes. Local offline profiling also fits all 220 rows under 32K
(maximum 22,236 tokens), subject to the disclosed tokenizer-runtime distinction.
The
32K serving context and three concurrent requests still require live qualification;
local rebuild approval does not authorize paid training.
