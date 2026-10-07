# Public API facade: preparation for the four new families (2026-09-17)

PREPARE pass for `EvaluationAPI`, `ChatAPI`, `DataAPI`, `PipelinesAPI`, scoped by
`docs/plans/api-facade-decisions-20260917.md`. Facts about the code on
`feat/modal-coordinator-adapter`; no design decisions. Every claim carries a
`file:line` that was read. Section 0 applies to all four families.

## 0. The v1 yardstick

### 0.1 Shape of a v1 contract

Two Protocol families exist with no engine implementation: `TrainingOperations` —
load, resolve, plan, preflight, start
(`synaptic_tuner/api/v1/training_facade.py:199-205`) — and `RunsOperations` —
list, show, outcome, logs, cancel, reconcile, verify, reverify, artifacts
(`api/v1/runs_facade.py:345-354`). The `TrainingAPI` wrapper re-validates types,
plan-fingerprint binding and preflight expiry (`training_facade.py:211-250`).
Value types are frozen slotted dataclasses validating in `__post_init__` and
round-tripping through `exact_fields`, so unknown or missing fields raise
(`_contract.py:52-70`); canonical JSON is sorted and `allow_nan=False`
(`:73-85`), digests domain-separated SHA-256 (`:88-90`).

### 0.2 Closed vocabularies

| Vocabulary | Location | Values |
|---|---|---|
| `TrainingRunState` | `api/v1/results.py:10-18` | planned, queued, running, succeeded, failed, cancel_requested, cancelled, reconcile_required |
| `LifecyclePhase` | `tuner/execution/contracts.py:136-148` | planned, ready, preparing, submitting, queued, running, verifying, succeeded, failed, cancelling, cancelled, reconcile_required |
| `VerificationStatus` | `contracts.py:151-157` | not_ready, pending, verifying, verified, invalid, inconclusive |
| `EffectKind` | `contracts.py:160-162` | **submit, cancel — only these two** |
| `EffectState` | `contracts.py:165-170` | claimed, attempted, found, definitely_absent, indeterminate |
| `ProviderRunPhase` | `contracts.py:179-185` | queued, running, succeeded, failed, cancelled, unknown |
| `EventCode` / `MessageCode` | `contracts.py:188-229` | 21 / 18 codes, all submission- or verification-shaped |
| `RunOperationCode` | `runs_facade.py:321-335` | 13 closed operation-error codes |
| `RunLogLevel` | `runs_facade.py:173-179` | trace, debug, info, warning, error, critical |

Public `TrainingRunState` and durable `LifecyclePhase` are already two
vocabularies: the public one drops ready/preparing/submitting/verifying/cancelling
and adds `cancel_requested`. Each new family inherits that mapping question.

### 0.3 Three structural limits every new family hits

These are the load-bearing findings of this pass.

1. **A lifecycle event carries no payload.** `LifecycleEvent`
   (`contracts.py:518-550`) accepts `code`, `occurred_at`, `message_code`, and
   optionally *either* an `EffectRecord` *or* a `GrantBinding`; anything else
   raises `"lifecycle event payload is invalid"` (`:550`), and `message_code` must
   be the single value `_EVENT_MESSAGE_CODES` maps the code to (`:232-254`). There
   is no slot for a progress counter, a per-case score, a token count, a chat turn
   or a stage name.
2. **The durable record is O(n²) in event count.** `LifecycleRecord` requires
   `revision == len(events)` and a non-empty event tuple (`:598-625`), and
   `canonical_bytes` re-serialises the whole event list per revision. A lifecycle
   emitting one event per generated row, eval case or chat turn rewrites its
   entire history each time.
3. **`EffectKind` is only `submit`/`cancel`.** `EffectRecord` also requires
   non-empty `canonical_command` bytes and, once `found`, both a
   `provider_job_ref` and a `receipt_digest` (`:367-405`); `ExecutionScope`
   requires provider, account_ref, namespace_ref (`:257-267`). Nothing here
   describes a local effect such as writing a dataset or spending at OpenRouter.

### 0.4 Already reusable

`ResultEnvelope`/`EventEnvelope` (`api/v1/events.py:14-81`) have published schemas,
and the event envelope is deliberately open — `event` any non-empty string, `data`
any object (`schemas/synaptic-event-v1.schema.json:9-14`) — with `sequence` and a
`final` flag that must bring a result (`events.py:63-66`). The JSONL writers
already redact by key name, URL userinfo and Bearer/Basic values
(`tuner/capabilities/events.py:19-26, 40-70`). `CapabilityDescriptor`
(`api/v1/capabilities.py:9-44`) and the declarative registry already declare
`evaluation.run` and `experiment.run` over an effects vocabulary of
filesystem_write / network / gpu / paid_compute / external_publish
(`tuner/capabilities/builtins.py:8-15, 46-91`). `AuthorizationRequirement` models a
paid effect with an optional `maximum_cost_minor_units` + ISO currency
(`training_facade.py:27-71`) — a preflight *cap*, not a recorded *spend*. Paging is
settled: limit 1..100, ASCII cursor ≤256 bytes, `truncated == (next_cursor is not
None)` (`runs_facade.py:77-87, 136-170`).

### 0.5 Import-boundary gate

The roadmap's P0 gate is "Public API imports no provider SDK or `tuner.*`
implementation". Today `api/v1/persistence.py:3-31`, `secrets.py:3`,
`context.py:3-4` and `sources.py:3-17` import from `tuner.*`, but only from
contract/value modules. `Evaluator/`, `SynthChat/` and `shared/experiment_tracking/`
are implementation, so that re-export shortcut is unavailable to the new families.

## 1. EvaluationAPI

**Unit of work.** One model × one backend × a set of YAML scenarios, run
in-process, writing one JSON file. Entry `Evaluator/cli.py:346` and
`Evaluator/__main__.py:8`; also `python tuner.py eval` via the router map
(`tuner/cli/router.py:292`). Inputs: `--model` required, `--backend` from a
closed list of 8 (`cli.py:357-363`), `--config-dir` defaulting to the
engine-relative `Evaluator/config` (`:365-369`), repeatable `--scenario`,
`--preset`, `--tags`, `--limit` and sampling flags (`:370-390`). Output is
written by `Evaluator/reporting.py:488-489` to the repo-relative default
`Evaluator/results/run_<ts>.json` (`Evaluator/cli_utils.py:158`). Neither GPU nor
payment is intrinsic: six backends are local, `openrouter` and
`openai_responses` are paid, and the descriptor already says `gpu="optional",
paid="possible"` (`tuner/capabilities/builtins.py:87`). Config-driven only in
part: scenarios, rubrics, behaviours and presets are YAML under
`Evaluator/config/`, everything else is argv. The cloud path is separate
(`tuner/handlers/cloud_eval_handler.py`, with in-job entrypoints
`Evaluator/cloud_hf_job.py` and `cloud_hf_job_vllm.py`).

**Schemas and vocabularies.** Input is YAML with no published JSON Schema;
validation is five hand-written validators totalling 2601 lines
(`Evaluator/config_validator.py`, `rubric_validator.py`, `judge_validator.py`,
`behavior_validator.py`, `schema_validator.py`). Output is
`{metadata, summary, records}` (`reporting.py:443-454`) with **no
`schema_version`** and no entry under `schemas/`; each record is a 22-key dict
(`:457-485`) inlining `raw_response` and `conversation_trace`.
`Evaluator/enums.py` holds the best-formed vocabulary in the four families:
`ResponseType` (`:14-35`), `BackendType` (`:38-70`), `ValidationLevel`
(error/warning/info, `:73-90`), `ToolCallFormat` (`:93-116`). Judge verdicts are
*not* an enum: `passed: bool` from `score >= pass_threshold`
(`shared/judge/models.py:79-82`); dimensioned rubrics compute a weighted
composite in our code, not the judge's (`:28-35`), and a `quality_gate` config
yields a second floor-gated composite (`:42-56`).

**Effects.** Filesystem: results JSON (`reporting.py:489`), mid-run partials
(`cli.py:159-160`), up to seven optional loss artifacts (`cli.py:265-343`). Network:
every backend client plus optional E2B sandbox execution (`cli.py:414-427`). Paid:
OpenRouter/OpenAI backends and every judge call.
External publish only on the cloud path (HF Jobs, `HfFileSystem` at
`cloud_eval_handler.py:457`). Ambient credentials: `Evaluator/config.py:156-159`
reads `VLLM_API_KEY` then falls back to `HF_TOKEN`;
`shared_llm_adapters.py:251,347` check `OPENROUTER_API_KEY`/`OPENAI_API_KEY`;
`cloud_eval_handler.py` calls `get_hf_token()` at eight sites
(`:172,202,207,216,457,524,544,663`), reading `HF_TOKEN` or `HF_API_KEY` from the
process environment (`shared/utilities/env.py:101-116`). No cancel path exists;
partial output on interrupt is the only mitigation (`cli.py:138-160`). Exit codes
are 0/1 via `SystemExit` (`Evaluator/__main__.py:8`).

**V1 gap.** Covered: `ResultEnvelope` fits the summary, the capability descriptor
exists, and paging and `RunLogEntry` transfer unchanged. Missing: **judge cost
and token accounting does not exist anywhere** — `shared/llm/base.py:16-37`
declares `chat(...) -> str` and `structured_output(...) -> Dict`, so no caller
can count tokens, and `JudgeResult` records only `latency_s`
(`shared/judge/models.py:111+`); the only token-bearing code is
`shared/flywheel/inference_logger.py`, which evaluation does not use.
Per-scenario progress events have no fitting `EventCode`
(`contracts.py:188-209`) and no payload slot (§0.3). Checkpoint selection exists
only as the bare string `Experiment.selected_run_id`
(`shared/experiment_tracking/experiment.py:119`). Resumability is genuinely
absent (`builtins.py:89` declares `resumable=False`).

**Legacy coupling not to inherit.**

| What | Where |
|---|---|
| Results default into the engine tree, not host state | `Evaluator/cli_utils.py:158` |
| Config dir defaults engine-relative | `Evaluator/cli.py:365-369` |
| Ambient `VLLM_API_KEY` → `HF_TOKEN` fallback in a settings getter | `Evaluator/config.py:156-159` |
| Ambient provider keys in adapters; ambient HF token at 8 call sites | `shared_llm_adapters.py:251,347`; `cloud_eval_handler.py:172-663` |
| `SystemExit` from library modules | `cli.py:1182`, `cloud_hf_job.py:462`, `cloud_hf_job_vllm.py:553`, `mlc_eval_handler.py:961`, `interactive_cli.py:789`, `lmstudio_cli.py:189` |
| Rich-table-only render with plain-print fallback, no structured return | `tuner/handlers/eval_handler.py:211-247` |
| Falls back to subprocess re-invocation when imports fail | `tuner/handlers/eval_handler.py:691-692, 820-822` |

**Questions.**
1. Does an evaluation run take a `run_id` in the same lifecycle repository as
   training, keyed by `(project_ref, run_id)` with no run-kind discriminator
   (`contracts.py:776-781`)?
2. Is a local eval a run with zero effects, or does `EffectKind` grow — and what
   is `ExecutionScope.provider` when there is none (`contracts.py:257-267`)?
3. Does `EvaluationAPI` require a usage-returning change to
   `shared/llm/base.py:16-37` so spend is recordable, and is that in scope?
4. Is the judge verdict closed as `passed: bool` + float or an enum, and does the
   result expose one score or two (`shared/judge/models.py:42-56`)?
5. Does the public result carry `raw_response` and `conversation_trace`
   (`reporting.py:483-484`), which are attacker-influenced text?
6. Is exact loss a stage of `EvaluationAPI` or a distinct capability?
   `EXPERIMENT_STAGES` treats `loss` as a peer of `evaluation`
   (`experiment_spec.py:13`) while `cli.py:219-343` bolts it onto the eval CLI.

## 2. ChatAPI

**Unit of work.** The family with the most v1-native code already written, in
two units. *Model-first*: `open_model_chat`
(`tuner/inference/model_chat.py:107-202`) is a context manager yielding a
`ChatSession`; it starts a local vLLM runtime bound to loopback (`:155-171`) and
builds a `VLLMClient` with `trust_environment=False, allow_redirects=False` and
byte bounds (`:172-180`), documented as never loading, submitting or resuming a
training run (`:121`). *Run-first*: `tuner/inference/run_chat.py:10-12` already
imports `RunsAPI`, `TrainingRunRef` and `VerifiedArtifact` from
`synaptic_tuner.api.v1`, and `PreparedRunChat` (`:39-70`) binds a session to a run
plus the exact canonical SFT artifact inventory, requiring the tuple to equal
`ROLES` in order and round-trip byte-identically (`:52-65`). *Modal*: ~50 modules
under `tuner/execution/providers/modal/`; the remote executable takes no argv and
is driven by a signed canonical JSON frame over Sandbox stdin
(`docs/review/modal-chat-stdio.md:9-15`). The session, not the completion, is the
unit: `ChatSessionPolicy` bounds request/idle/absolute timeouts, `max_turns`
≤10000 and `max_history_bytes` ≤64 MiB (`Evaluator/chat_session.py:37-61`).
Nothing is written to disk — a grep for `transcript` across `tuner/inference/`,
`chat_session.py`, `scripts/chat_model.py` and `modal/inference_run_chat.py`
returns nothing; history is in memory only.

**Schemas and vocabularies.** A wire contract already exists:
`_SCHEMA = "synaptic-modal-chat-channel/v1"` with five closed frame kinds —
ready, chat, stop, error, closed — and per-kind exact field sets
(`modal/inference_channel.py:16-62`). Every frame binds `session_id` and
`launch_digest`; chat/stop/error/closed carry a consecutive integer
`request_id`; error carries a `code`. Frames are bounded at 1 MiB (`:18`).
`ChatSessionState` is a six-field observation dataclass — closed,
cleanup_pending, request_inflight, turns, history_bytes, close_error
(`chat_session.py:64-71`) — with no `to_dict`. `PreparedModelIdentity`
(`run_chat.py:16-36`) is contract-shaped: model_ref, model_revision,
tokenizer_revision each exactly 40 or 64 hex chars, `model_kind` ∈ {full, lora};
the ready frame carries it (`modal-chat-stdio.md:38-41`). Errors are closed
strings, not enums: `"model_chat_failed"`, `"model_chat_deadline_expired"`
(`model_chat.py:47, 199`), `"modal_chat_channel_error"`
(`inference_channel.py:17`). OpenAI-compatible payloads are confined to
`VLLMSettings`/`VLLMClient` (`model_chat.py:160-180`) and surface nowhere public.

**Effects.** No filesystem writes; the model source must be a canonical local
directory or a Hub ref pinned to a 40-hex commit (`model_chat.py:58-104`), and
vLLM binds loopback only (`:125`). `OwnedProcessLease.spawn` uses
`start_new_session=True` and owns the process group
(`Evaluator/owned_process.py:45-67`); teardown polls `/proc` for group members,
distinguishes a still-tearing-down thread group from a foreign reap and marks
the lease *uncertain* rather than lying (`:150-185`), and `__exit__` raises
`OwnedProcessError("owned process family cleanup remains unresolved")` when
cleanup did not converge (`:194-202`). **This is Linux-only**: `_require_host`
demands `os.name == "posix"`, `/proc/self/stat` and `os.killpg` (`:207-216`). On
any exception `open_model_chat` attaches the pending runtime lease to the raised
error as `cleanup_lease`, including for `KeyboardInterrupt`/`SystemExit`
(`model_chat.py:186-202`). Credentials are already clean: the allowed environment
is a closed allowlist (`:29-36`), anything outside it is rejected (`:129-138`),
and `HF_HUB_DISABLE_IMPLICIT_TOKEN=1` is forced (`:140`). Cancellation is by
policy timeout plus an optional monotonic deadline (`:43-48, 152, 159, 184`).

**V1 gap.** Covered: `TrainingRunRef` and `VerifiedArtifact` already bind a
session to a verified run (`run_chat.py:39-70`), and the Modal frames give
sequence-numbered request identity and closed kinds. Missing: **a session is not
a run** — `LifecycleRecord` is keyed by `run_id` and needs at least one event
(`contracts.py:598-625`), while a session has only a `session_id`
(`inference_channel.py:24`). Streaming does not exist today at all: a turn is one
blocking request/response, so it is net-new rather than a migration. Turn-level
`EventEnvelope` emission is fine; one *lifecycle* event per turn is not (§0.3).
Readiness probing is internal to `Evaluator/vllm_runtime.py` with no public
shape. Reconnect/attach does not exist — `OwnedProcessLease` can only be created
by `spawn` (`owned_process.py:41-43`).

**Legacy coupling not to inherit.** The cleanest family; the specific items:

| What | Where |
|---|---|
| Linux-only process ownership, hard failure elsewhere | `Evaluator/owned_process.py:207-216` |
| Failure signalled by attaching a mutable attribute to the exception | `tuner/inference/model_chat.py:186-201` |
| `tuner/inference/` depends on `Evaluator.*` for session, client and runtime | `tuner/inference/model_chat.py:18-27` |

That last row matters: a `ChatAPI` in `synaptic_tuner/api/v1/` would transitively
pull `Evaluator/` — 15630 lines including Rich UI, argparse CLIs and eight
backend clients — into the public import closure, against the P0 gate (§0.5).

**Questions.**
1. Is a chat session a lifecycle entity with its own `run_id`, a child of a
   training run, or outside the lifecycle model? `PreparedRunChat` already binds
   `TrainingRunRef` (`run_chat.py:41`) with no lifecycle record.
2. With no push stream, how does a host observe an in-flight turn — cursor-paged
   polling of turn events, or only the completed turn?
3. Does the `synaptic-modal-chat-channel/v1` frame vocabulary
   (`inference_channel.py:16-62`) become the public `ChatAPI` vocabulary?
4. Must `ChatAPI` work on a non-Linux host? If yes, `owned_process.py` needs a
   second arm; if no, that is a stated engine constraint.
5. What is the public shape of "cleanup remains unresolved"
   (`owned_process.py:201`)? It is the chat analogue of `reconcile_required` and
   has no `RunOperationCode` (`runs_facade.py:321-335`).
6. Which of `chat_session.py`, `vllm_runtime.py`, `owned_process.py` move out of
   `Evaluator/` to keep the public closure clean, and is that move in scope?

## 3. DataAPI

**Unit of work.** Three units sharing config but not shape. *Generation*:
`python -m SynthChat.run generate` → `generate_mode`
(`SynthChat/modes/generate.py:30`); inputs are scenarios plus per-scenario target
counts, output one JSONL streamed row by row. Duration scales with target count ×
LLM latency; a pilot manifest such as
`SynthChat/config/targets_vault_shared_seed_10x10.json` implies ≥100 round-trips.
A paid provider is the practical default (`improve_handler.py:55` defaults to
`openrouter`) though settings default to `lmstudio` (`SynthChat/run.py:44`).
*Improvement*: `modes/improve.py` or `RubricRunner.run_on_file` taking
`start_line`/`end_line` and `max_iterations` (`synthchat_handler.py:335-370`).
*Validation/listing*: `modes/validate.py`, `modes/sanitize.py`, and `ListHandler`
for datasets/models/runs/rubrics/scenarios (`list_handler.py:1-17`). Pipeline
shape: modes → `SynthChat/generator.py` (1931 lines) → deterministic per-stage
gates (`stage_gates.py:29-47`) → optional rubric-driven improvement
(`engine.py`) → `StreamingResultWriter` (`result_writer.py:17-80`). Parallelism
is a thread pool of `--workers`, forced to 1 when prompt-optimisation overlays
are active (`modes/generate.py:229-234`).

**Schemas and vocabularies.** `SynthChat/config/settings.yaml` has top-level keys
llm, improvement, output, resilience, logging, generation, privacy_preprocess,
environment, cost_tracking, defaults (`:4,47,56,62,69,76,82,94,103,109`);
`validation.yaml` has scopes, scope_processing_order, llm, judge, validation,
output (`:5,76,83,97,121,134`); plus `tool_call_formats.yaml`,
`workspace_formats.yaml`, `label_mappings.yaml`, `privacy_profiles.yaml`,
`defaults.yaml` and 24 `targets_*.json` run manifests. `SynthChat/schemas/` is
Python, not JSON Schema. Output rows are JSONL, but **the first line is a `_meta`
header object mixed into the data stream** when `output.include_metadata` is set
(`result_writer.py:41-54`), so the file is not homogeneous and every reader must
special-case line 0. `StageGateResult{gate_type, passed, message, metadata}` has
a `to_dict` (`stage_gates.py:13-27`) but `gate_type` is an open string and an
unknown type yields `passed=False` rather than an error (`:37-46`). No output
carries a `schema_version`; nothing is under `schemas/`.

**Effects.** `StreamingResultWriter.__enter__` opens the output with mode `"w"`,
truncating (`result_writer.py:38`); handler-driven generation writes
`generated_<YYYYmmdd_HHMMSS>.jsonl` under a handler-chosen directory
(`synthchat_handler.py:238-241`). Every generation and improvement turn goes
through `shared.llm.create_client` (`SynthChat/run.py:39-77`).
**Cost accounting is declared and unimplemented**: `settings.yaml:103-105` sets
`cost_tracking: {enabled: true, include_usage: true}` and a repository-wide grep
for `cost_tracking` in Python returns zero consumers — consistent with
`shared/llm/base.py:16-22` returning a bare `str`. Ambient credentials:
`services/rubric_runner.py:108-109` reads `OPENROUTER_API_KEY` and
`OPENAI_API_KEY` directly and `:86-89` reads LMSTUDIO/OLLAMA host and port;
`services/privacy_preprocess.py:53-54` reads a config-named host/port var;
`services/pseudonymizer.py:99` reads `SYNTHCHAT_PRIVACY_SEED`;
`improve_handler.py:55-56,97` reads `IMPROVEMENT_BACKEND`/`IMPROVEMENT_MODEL` and
gates on `OPENROUTER_API_KEY`; `generate_handler.py:95` reads `LMSTUDIO_MODEL`.
Underneath all of them `shared/utilities/env.py:58-76` walks `cwd`, `cwd.parent`,
`cwd.parent.parent` and the engine root looking for a `.env` to load. Resume
exists only as the improvement line slice (`synthchat_handler.py:343-344`).
`sys.exit` appears in mode modules (`modes/improve.py:130,135`,
`sanitize.py:31,42`, `validate.py:52,57`, `run.py:243`) and inside a *service*
(`services/rubric_runner.py:615,620,630`); exit codes are 0/1.

**V1 gap.** Covered: `VerifiedArtifact{role, sha256, size_bytes}`
(`api/v1/results.py:39-60`) is the right shape for a produced dataset file, and
`ResultEnvelope` already carries an `artifacts` list (`events.py:20`). Missing:
**partial success has no representation** — rows stream to disk and the run may
then fail, but neither `TrainingRunState` (`results.py:10-18`) nor
`LifecyclePhase` (`contracts.py:136-148`) has a partial member, so the only
evidence today is "the file has N rows". Per-stage progress lives in
`SynthChat/utils/progress_tracker.py`, console-facing, with no envelope emission.
Cost accounting is structurally absent, resume is absent for generation and
actively prevented by the truncating open, and nothing hashes the produced JSONL,
so a `VerifiedArtifact` cannot be produced without new code.

**Legacy coupling not to inherit.**

| What | Where |
|---|---|
| `.env` discovery by walking up from `cwd` | `shared/utilities/env.py:58-76` |
| Ambient provider keys inside a service | `SynthChat/services/rubric_runner.py:108-109` |
| `sys.exit` inside a service, not a CLI | `services/rubric_runner.py:615, 620, 630` |
| `print()` as the only progress and result channel | `modes/generate.py:135-290` |
| Truncating output open, defeating resume | `SynthChat/result_writer.py:38` |
| `_meta` header row inside the data JSONL | `SynthChat/result_writer.py:41-54` |
| Hardcoded output filename, engine-relative `scenarios/` and `rubrics/` | `tuner/handlers/synthchat_handler.py:241, 247-249` |
| Rich import with silent `print` fallback, no structured return | `tuner/handlers/list_handler.py:37-52` |

**Questions.**
1. Is a generation run one lifecycle run producing one artifact, or one run per
   scenario? The `targets_*.json` manifests are per-scenario count maps, and
   `EXPERIMENT_STAGES` has no generation stage (`experiment_spec.py:13`).
2. How is partial success expressed — a new phase, a `succeeded` run declaring a
   shortfall, or a `failed` run that still lists artifacts? `RunOutcome` permits
   artifacts on any state (`runs_facade.py:91-112`).
3. Does `DataAPI` require the `_meta` header row to leave the data file
   (`result_writer.py:41-54`) so the artifact can be digested as homogeneous
   JSONL?
4. Is adding usage reporting to `shared/llm/base.py:16-37` in scope, or does
   `DataAPI` ship without spend despite `cost_tracking.enabled: true`?
5. Listing and validation are read-only with no run. Do they belong in the same
   facade, and do they get paging like `RunListRequest` (`runs_facade.py:136-146`)?
6. `SynthChat/` is 15763 lines and a handler imports Rich UI. What boundary keeps
   the public closure free of console code?

## 4. PipelinesAPI

**Unit of work.** One experiment = an ordered stage set over one dataset and one
base model, run **synchronously in one process**, with state on local disk.
Entry `python tuner.py run-experiment --experiment-spec <yaml>`
(`tuner/cli/parser.py:210, 266, 717`) → `ExperimentHandler`
(`tuner/cli/router.py:298`), which requires the spec
(`tuner/handlers/experiment_handler.py:39`). Stages are
`("training", "evaluation", "loss", "analysis", "recommendation")`
(`shared/experiment_tracking/experiment_spec.py:13`), selectable via
`--only-stage`, `--from-stage`, `--skip-stage` (`parser.py:737-748`).
`ExperimentOrchestrator` (`experiment_orchestrator.py:29-47`) holds three
`StageRunner` implementations (`:25-26`) and runs evaluation and loss
**concurrently** in a 2-worker thread pool after training (`:112-126`). State
lives in `.tracking/`: the registry is `{repo_root}/.tracking/registry.jsonl`
(`registry.py:264-271`), one serialized `RunRecord` per line (`schema.py:44-57`),
and experiments live at `.tracking/experiments/{id}/` (`experiment.py:100`). The
descriptor declares `gpu="required", paid="possible"` with confirmation required
(`tuner/capabilities/builtins.py:57-58`).

**Schemas and status divergence.** Input is `ExperimentSpec` YAML, a dataclass
tree with no JSON Schema and a hand-rolled `validate() -> list[str]`
(`experiment_spec.py:155-159`). The on-disk `Experiment` is a 40+ field
**mutable** dataclass (`experiment.py:98-140+`) covering status, stage_statuses,
stage_details, artifact_roots, derived_outputs, selected_run_id and fourteen HF
provenance URI/SHA pairs, written atomically as canonical JSON (`:60-96`) — the
one place legacy already matches v1 serialization discipline. The status
divergence is the headline for this family:

| Layer | Values | Source |
|---|---|---|
| Experiment status | `partial` (default), `completed`, `failed` | `experiment.py:111`; `service.py:3340` |
| Stage status | `running`, `completed`, `failed` | `service.py:3375-3385`; `experiment_orchestrator.py:116-119` |
| `RunRecord.status` | `completed`, `failed`, `running` | `schema.py:63` |
| v1 `TrainingRunState` | planned, queued, running, succeeded, failed, cancel_requested, cancelled, reconcile_required | `api/v1/results.py:10-18` |

Only `running` and `failed` are shared. Legacy says `completed`, v1 says
`succeeded`. Legacy has `partial`, which v1 has nowhere. v1 has five states
legacy cannot express, including every cancellation and reconciliation state.
`experiment.py:14-48` adds seven more SCREAMING_CASE tuples covering HF source
transport, submission, provisioning, cancellation, observation and result states;
`HF_TRAINING_RESULT_STATES` is `VerificationStatus` (`contracts.py:151-157`) in
different casing and missing two members. `.tracking` files are canonical JSON;
the CLI surface is not (see the coupling table below).

**Effects.** All of `.tracking/` — registry JSONL, per-experiment directories,
analysis bundles, a benchmark ledger whose directory is overridable by an env var
(`benchmark_ledger.py:28`). Paid compute is whatever the stage runners do;
`cloud_loss_job.py` is an in-HF-Job entrypoint configured by `SYNAPTIC_*` and
`HF_BUCKET_SYNC_*` env vars (`:57-59, 112, 125`) that copies the entire process
environment into the child (`:59`). Partial failure: `_finalize_stage_result`
marks each stage as it finishes (`experiment_orchestrator.py:62-84`), a restored
stage is skipped (`:49-60`), and resume reads `{stage}_run_id` off the experiment,
falls back to the registry, then to `stage_details`, accepting only `completed`
or `failed` (`:128-150`). Resume therefore exists and is per-stage, but keyed on
mutable fields of a mutable dataclass. No cancel path exists; exit codes are 0/1.

**V1 gap.** Covered: nothing structural. `LifecycleRecord` models exactly one run
(`contracts.py:598-625`) and `LifecycleRepository` is keyed by
`(project_ref, run_id)` (`:776-781`). Missing: **nested runs** — no parent/child
relation exists in the lifecycle model, while legacy expresses it with
`RunRecord.parent_run_id` (`schema.py:65`) plus a `relationship` of `"parent"` or
`"derived_from"` chosen by stage role (`experiment_orchestrator.py:72-80`).
Per-stage lifecycle records do not exist: stages are dict entries
(`experiment.py:122-123`) with no revision, event log or optimistic concurrency.
Cross-stage artifact handoff is by path string — `artifact_roots: dict[str,str]`
(`experiment.py:120`) and `StageResult.artifact_root`
(`experiment_orchestrator.py:22`) — not by digest, while roadmap Phase 9 requires
binding each stage to exact upstream artifact hashes rather than `latest`, so
`VerifiedArtifact` must replace the path string.

**Legacy coupling not to inherit.**

| What | Where |
|---|---|
| `.tracking` as a literal default argument, four sites | `experiment.py:349, 380, 396`; `analysis_bundle.py:81` |
| Registry path walks ancestors of the module file; `.tracking` compared as a string | `registry.py:264-271`; `service.py:138-148` |
| Mutable 40-field durable record with no revision field; stage state as free-form dicts | `experiment.py:98-140, 122-123` |
| `compare-runs` shells out to a repo-relative script; `create-experiment` prints and returns | `tuner/cli/router.py:254-273` |
| Whole process environment copied into an HF Job child; `sys.exit` in that entrypoint | `cloud_loss_job.py:59, 278` |
| `TrackingService`, 3523 lines, the orchestrator's sole persistence port | `shared/experiment_tracking/service.py` |

**Questions.**
1. Is a pipeline a run that owns child runs, or a separate entity with its own
   repository? `LifecycleRepository` has no parent concept
   (`contracts.py:776-781`) and `LifecycleRecord` no parent field (`:598-610`).
2. Does each stage get its own `LifecycleRecord`? If so, what does the pipeline's
   `phase` mean when evaluation and loss run concurrently
   (`experiment_orchestrator.py:112-126`) and disagree?
3. How does `partial` (`experiment.py:111`) map into v1? It is the legacy
   *default*, so most existing records carry it — migrate, or abandon `.tracking`
   per Phase 10?
4. Do the seven HF state tuples (`experiment.py:14-48`) collapse into the v1
   vocabularies, or stay a provider-local dialect behind the adapter?
5. Handoff by digest requires every stage to emit `VerifiedArtifact`. Does that
   force `EvaluationAPI` first, and force the loss stage to have a facade too?
6. What is the v1 resume key — the plan fingerprint, as `TrainingPreflight.binds`
   uses (`training_facade.py:124-127`), or a stage-level effect key?

## 5. Cross-family questions

1. **Event payloads.** All four families need progress reporting and
   `LifecycleEvent` structurally cannot carry it (§0.3). Does the lifecycle model
   grow a payload-bearing event, or do these families emit `EventEnvelope` as a
   separate, non-durable channel and keep the lifecycle record coarse?
2. **Record growth.** `revision == len(events)` with full re-serialisation
   (`contracts.py:625`) bounds durable events per run. What is the budget?
3. **Non-submission effects.** `EffectKind` is submit/cancel only
   (`contracts.py:160-162`) and `EffectRecord` demands a `canonical_command` and,
   once found, a `provider_job_ref` + `receipt_digest` (`:367-405`). Writing a
   dataset, spending at OpenRouter and opening a local vLLM server have none of
   those. Does `EffectKind` grow, or are local effects outside the ledger?
4. **Spend.** No engine path can count tokens, because `shared/llm/base.py:16-22`
   returns `str`, and `settings.yaml:103-105` promises `cost_tracking` nothing
   implements. Three of four families are paid. Is usage reporting a prerequisite?
5. **Import boundary.** `Evaluator/` (15630 lines), `SynthChat/` (15763) and
   `shared/experiment_tracking/` (8843) are implementation carrying Rich,
   argparse and provider SDKs, and the P0 gate forbids the public API importing
   `tuner.*` implementation (§0.5). What dependency direction is permitted, given
   `tuner/inference/run_chat.py:10-12` already imports *up* into the public API?
6. **Host-owned state.** Three families write to engine-relative paths today:
   `Evaluator/results/` (`cli_utils.py:158`), `.tracking/` (`experiment.py:349`)
   and handler-chosen SynthChat output dirs (`synthchat_handler.py:238-241`).
   Does each facade take a destination port from the host, as `ArtifactsAPI` does?
7. **Read-only queries.** Dataset, rubric and scenario listing and run comparison
   have no run and no effects. Do they live in the four facades, or in a separate
   query surface with its own paging contract?
