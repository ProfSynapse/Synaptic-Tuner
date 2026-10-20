# Durable Modal consumer: Syntunia integration baseline

This walkthrough uses the existing installed `synaptic train` and `synaptic runs`
commands, which compose public `TrainingAPI` and `RunsAPI`. It consolidates a
working consumer path; it does not introduce another launcher or persistence
implementation. Start Syntunia's integration with prepared data, Modal SFT,
known-run reads and verified artifacts. GRPO is also required on the release
and Syntunia readiness path; its genuine GPU qualification is still open. Add
evaluation with its own configured backend and authority.

The [support matrix](../../docs/plans/api-v1-support-matrix.md) distinguishes
implemented behavior from live qualification. Two bounded SFT runs on
`864a9bcc` passed completion and native artifact verification; the second passed
fresh-process observation/readback. Active cancellation, caller-loss recovery
and a joined training-to-evaluation product workflow remain qualification work.
HF Jobs is paused. Desktop local training on NVIDIA and Apple silicon is also
required: Docker/CUDA and a native macOS GPU runtime need separate qualification.
This Modal walkthrough does not qualify either local backend.

## 1. Prepare the consumer and installation

Use a real consuming repository with a clean pushed commit and the exact engine
gitlink. Start its manifest from
[`examples/host-project/synaptic.yaml`](../host-project/synaptic.yaml); replace
the example project identity, paths and configuration with the host's values.
The generic manifest alone is not a Modal recipe or execution grant.

Keep the engine checkout used for source verification separate from the
installed distribution used by the launcher. Record both identities. Examples
and source build declarations are checkout-owned and excluded from the wheel.
The accepted wheel's build source can differ from the tested checkout only when
the relevant excluded-file delta has been reviewed; do not relabel the wheel.

The effectful Modal launcher uses **POSIX CPython 3.11.14, Modal 1.5.4 and the
hash-pinned launcher closure**. Base package Python support is broader and does
not override those pins. Follow the existing
[isolated launcher installation](../modal_chat/README.md#launcher), using its
requirements lock and an exact reviewed wheel installed with `--no-deps`.
When using uv on the same filesystem as its cache, use `--link-mode copy`;
runtime source validation rejects shared hardlinks. Run the dependency check
in that environment. Do not install a local GPU stack.

Use the installed `synaptic` entrypoint from that environment. Confirm the
entrypoint and engine installation originate there, rather than an ambient
checkout. The version `1.1.0` is an intended release version, not evidence that
a stable release has been published.

## 2. Prepare and inspect a real recipe

The host supplies a reviewed Modal SFT recipe, real prepared dataset and exact
model/tokenizer/runtime profile. Existing recipe structure is illustrated by
[`qwen35_4b_32k_modal_prompt_completion.yaml`](../../Trainers/recipes/qwen35_4b_32k_modal_prompt_completion.yaml);
its dataset paths and resource choices are not a ready-to-run smoke. Do not
copy private data paths or treat a named candidate profile as qualification.
Use the existing dataset preparation service for the declared input contract.

Run from the consuming project with its manifest and `SYNAPTIC_ENGINE_ROOT`
set to the absolute path of its exact accepted engine checkout, as described in the
[Modal reference](../../.skills/fine-tuning/reference/modal-jobs.md).
In the commands below, replace the uppercase values with the actual reviewed
paths and named scope. The selected recipe must bind that real host and engine.

```sh
synaptic train --job-config /absolute/consumer/config/REVIEWED_RECIPE.yaml --plan --json
synaptic train --job-config /absolute/consumer/config/REVIEWED_RECIPE.yaml --prepare-local --json
```

`--plan` performs no provider effects. `--prepare-local` may fetch exact public
bootstrap artifacts and build the exact source wheel in private local storage;
it creates no Modal resources or submission claims. These are separate commands.
Normal execution also performs this preparation, so it is not an obligatory
manual prerequisite. Neither command authorizes paid execution.

The default private run root is
`$XDG_STATE_HOME/synaptic-training/<run-id>`, or
`~/.local/state/synaptic-training/<run-id>` when XDG state home is unset.
Choose a persistent, private local state location before execution and keep it
stable across processes. Do not put it in a disposable worker filesystem.

## 3. Submit one admitted run

After reviewing the exact source, input, profile, account/workspace, finite
resource/time/cost scope and current authority, the normal execution command is:

```sh
synaptic train --job-config /absolute/consumer/config/REVIEWED_RECIPE.yaml \
  --modal-profile NAMED_PROFILE --modal-environment EXISTING_ENVIRONMENT --json
```

There is no `--execute` flag. This command can create resources, qualify the CPU
runtime and submit paid training. Do not run it as a connectivity or HTTP health
check. Do not add `--fresh-attempt` to retry a timeout or unknown submission.
Retain the exact returned run identity and original attempt records. A reported
retained run after a failure is not proof that training completed.

The normal runner persists and reads back its original keys and composition
inputs before `TrainingAPI.start`. State includes:

| Retained material | Purpose |
| --- | --- |
| `workflow.sqlite3` | Workflow records and history |
| `authority-keys.json` | Original host signing/reader keys and runtime evidence key |
| `readback-inputs.json` | Exact request, workload, scope, runtime and planning bindings |
| Original attempt journal and catalogs | Consumed one-shot claims and exact known provider call |

Keep all of these together at their original configured locations. Directories
are private `0700`, retained key/input files are `0600` and create-only. They are
internal private state, not portable API exports. Provider credentials remain in
the named credential mechanisms; the host signing keys still need private custody.
Never send this state to the browser or commit it.

## 4. Read the same run from a new process

Use the exact run ID, the original state location, and the same named
profile/environment. `latest` is deliberately unsupported.

```sh
synaptic runs show --run EXACT_RUN_ID \
  --modal-profile NAMED_PROFILE --modal-environment EXISTING_ENVIRONMENT --json
synaptic runs observe --run EXACT_RUN_ID \
  --modal-profile NAMED_PROFILE --modal-environment EXISTING_ENVIRONMENT --json
```

`show` returns the stored outcome without polling the call. Reader composition
can still authenticate the named provider scope. `observe` checks the exact
known call and may persist authenticated evidence; a pending result does not
prove current GPU activity. Neither command prepares or submits new training.
Missing original keys, an unknown call or conflicting evidence must remain an
unavailable/uncertain read, never trigger automatic submission.

For an eligible `SUCCEEDED_UNVERIFIED` or `VERIFICATION_FAILED` run:

```sh
synaptic runs verify --run EXACT_RUN_ID \
  --modal-profile NAMED_PROFILE --modal-environment EXISTING_ENVIRONMENT --json
```

For an already `VERIFIED` run, use `show` and download instead of verifying again:

```sh
synaptic runs download --run EXACT_RUN_ID \
  --modal-profile NAMED_PROFILE --modal-environment EXISTING_ENVIRONMENT \
  --artifact-role final_model --dest /absolute/private/UNUSED_DOWNLOAD_DIRECTORY --json
```

The five native roles are `final_model`, `tokenizer`, `training_lineage`,
`training_metrics` and `workload_record`. Download verifies length/hash and creates
`<role>.artifact`; it refuses overwrites and preserves failed partial files.
These native files are not automatically an HF repository layout or a serving
deployment. Destination publication uses separate `ArtifactsAPI` composition.

Cancellation requires original recipe opt-in plus current caller permission;
it is not enabled by reopening the database. A cancellation acknowledgment is
not confirmation that compute stopped. The
[lifecycle supervisor](README.md#owned-modal-producer-interruption-linux-caller-example)
is qualification tooling, not a production job queue or recovery service.
General logs remain disabled without their own exact proof; the UI must handle
missing progress and use verified terminal outcomes to establish completion.

## 5. Build Syntunia around the public boundary

The product should call the typed facades rather than parse these CLI messages
as a new stable wire protocol:

| Product responsibility | Existing engine boundary |
| --- | --- |
| Admit a prepared request and selected provider | `TrainingAPI.load`, `resolve`, `plan`, `preflight` |
| Start after host authority is bound | `TrainingAPI.start(plan, preflight)` |
| Display retained state / observe a known call | `RunsAPI.show(run)` / `outcome(run)` |
| Verify and stream native files | `RunsAPI.verify(run)` / `artifacts(request)` |
| Publish to a configured destination | `ArtifactsAPI` with explicit source/store/destination ports |
| Evaluate and compare retained reports | `EvaluationAPI` with configured backend, scenario and authority ports |

`TrainingAPI.load` accepts the configured canonical request JSON, not arbitrary
recipe YAML. Use the existing preparation/compiler path to produce admitted
inputs. The CLI has closed diagnostics, but Training/Artifacts do not expose a
universal operation-code enum; do not promote exception messages to HTTP errors.

`synaptic_tuner.api.v1` is the public contract. Explicit composition modules
include `synaptic_tuner.api.v1.reference` and `synaptic_tuner.api.v1.modal`;
the host still supplies their configured ports and lifetimes.
`tuner.training.modal_host_composition`, `modal_standalone_runner` and
`modal_retained_reader` implement the normal installed CLI host. They are useful
implementation references, **not stable root API exports**. A future Syntunia
adapter must either compose the public ports or explicitly pin/isolate any use
of those internal helpers. This walkthrough does not claim a one-call public
constructor that restores every host component.

Pass the same caller-owned `SQLiteDurableStoreV1` instance to `records` and
`streams` for atomic supported head/history writes. Each process opens its own
connection on its owning thread; do not share it across HTTP worker threads.
The [storage guide](../../docs/architecture/reference-host-storage.md) specifies
other required catalogs, plan-index interruption limits and publication fencing.
Unknown database versions reject; no automatic schema migration is supplied.
Use SQLite-aware backup or close all writers, and retain original keys/catalogs;
copying an active database file alone is not recovery.

Syntunia supplies HTTP/authentication, project authorization, credential
selection, admission budgets, background job ownership, durable storage and
product identifiers. Provider scope alone is not tenant isolation. Keep
unknown submission, cancellation request and verified completion distinct.
Polling/SSE, if implemented by the host, must preserve capability limits and
cannot infer completion from missing metrics.

The older [`modal_chat` example](../modal_chat/README.md) remains a separate
same-process train/chat example. Its persistent attempt journal does not make
its in-memory coordinator/Foundation state restartable. Training qualification
does not authorize chat, model serving or a paid evaluation.

## Existing verification

The checked-in tests exercise the implementation used above:

- `tests/training/test_modal_run_retention_integration.py`: original inputs and
  keys retained before start; retention failure blocks submission.
- `tests/training/test_retained_runs_cli_integration.py`: fresh readers reopen
  stores, use exact known calls after local inputs are removed and never replay.
  These tests use a scripted provider, not a live fresh-process qualification.
- `tests/training/test_modal_retained_reader.py`: original custody/bindings,
  missing-state refusal and unsupported capabilities.
- `tests/contract/test_public_training_api_v1.py`: public signatures and
  uncomposed-family behavior.

The current live pair and its limits remain in the release plan. The next
integration acceptance is a real host-owned journey through preparation,
SFT and GRPO training, retained readback, artifact retrieval and separately
authorized evaluation, including restart and cancellation cases. No test fixture or this
walkthrough substitutes for that evidence.
