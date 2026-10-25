# Public SFT qualification consumer

This host-owned checkout example exercises the existing public TrainingAPI and
RunsAPI. It adds no engine provider capability, lifecycle database, grant
service or training launcher. The CPU tests use scripted provider boundaries
and synthetic trainer payloads with real compilers, SQLite, original host
signatures, native verification and public five-role downloads. They do not
qualify an image, model, GPU, account, bill or real SFT update.

Use an installed engine interpreter and the absolute consumer script from a
neutral working directory. The script exposes only its own directory for fixed
sibling adapters, never the checkout engine root:

```sh
python -I /absolute/checkout/examples/training_qualification/launch.py plan /private/consumer.json
python -I /absolute/checkout/examples/training_qualification/launch.py start /private/consumer.json
python -I /absolute/checkout/examples/training_qualification/launch.py show /private/attempt
python -I /absolute/checkout/examples/training_qualification/launch.py observe /private/attempt
python -I /absolute/checkout/examples/training_qualification/launch.py verify /private/attempt
python -I /absolute/checkout/examples/training_qualification/launch.py download /private/attempt --destination /private/downloads
```

Copy the provider template and replace every `REPLACE_*` value. Templates are
not qualifications or approvals. Configuration contains named SecretRefs only;
credential/signing-key values stay in the consuming host's credential store.
The default resolver reads explicitly named environment variables at composition,
without loading `.env` or choosing ambient credentials. Original authority/read
keys must survive restart unchanged. `host.allowed_requirements` selects existing
AuthorizationRequirement documents; the small host GrantPort refuses a different
requirement or run/project. It does not create an external approval service.

`plan` is a credential-free compiler/material/profile/basis preview. Full public
operational preflight is **deferred until actual host composition**. It does not
resolve signing keys, acquire an input lease, download models, provision resources
or contact a daemon. `start` creates an unused 0700 attempt, retains exact request
bytes/config/preview and then performs public load → resolve → plan → preflight
→ start. Its actual public basis/provider/run must match the preview. The existing
public plan and preflight are retained before the single start call. An existing
attempt refuses; unknown or failed submissions are inspected, never replayed.
Known-run routing never invokes shared preview or start.

Successful `observe` means the existing provider observation succeeded, not that
native training artifacts verified. `verify` uses the existing native verifier.
Only its verified five-role manifest can be downloaded: final_model, tokenizer,
training_lineage, training_metrics and workload_record. The existing downloader
checks exact streamed hash/length/EOF, preserves partial failures and refuses
completed-file collisions. Downloads default to 256 MiB per artifact; a local
failure cannot authorize training replay.

## Docker host boundary

Docker runs locally on the same host/filesystem as the selected Unix daemon.
Remote contexts/path mapping are not supported. The existing derived-image
verification profile/report and immutable image must already exist; the example
never builds or pulls an image. Its effect-stage identity checks use the same
explicit Unix endpoint as the actual CLI session. The session preserves its
existing daemon-ID/image/label/path checks and fixed source staging/claim order.
Configured command deadlines bound provider CLI work, not all synchronous host
configuration/store/signing time or the complete training job. Runtime resource
and job time limits come from the existing compiled material/profile. None of
these declarations enforce an invoice ceiling or prove physical termination.

Supply existing canonical CoordinatorResolvedMaterial and DockerProfile JSON
with exact hashes/sizes, plus the identical canonical request document. The
adapter rederives them through the existing recipe registry/bridge. Before
effects, material/profile/request bytes are copied privately using their existing
formats, and preview equality is checked again. Cold composition uses those
copies, existing-only SQLite and original keys. It still requires the original
derived-image profile/report (including their relative dependencies), actual
local source/artifact mount roots and unchanged installed engine closure. Missing
assets refuse; this is not self-contained source-free Docker reconstruction.
Source staging is disabled during read composition. It cannot restore a running
trainer, replay an uncertain submit or invent a lost job handle.

The first default Docker source adapter accepts independently verified published
syntunia-sft-row/v1 through 2 MiB. It opens a genuine existing one-use input lease,
but source publication/take occurs only through the existing claimed stage.
There is no one-off raw-row substitute. Actual model-cache preparation/placement
and image/model/GPU qualification remain operator prerequisites; source/artifact
mounts alone do not prove a runnable image or real training update.

`cancel --reason ...` and `reconcile` delegate existing public verbs, current
host policy and descriptor flags. The example never turns flags on. Ordinary
unqualified capability refusals remain refusals. Explicit current cancellation
authorization is default false. A stop acknowledgment is not causal CANCELLED:
natural success/failure remains its actual observed outcome. Consumed uncertain
attempts do not issue another stop or training submission.

## HF adapter boundary

The HF sibling adapter uses the same exercise/host ports. Its default capabilities
remain unpromoted. Pure preview is not SDK/image/account qualification. Warm start
requires genuine original compiled material, selected runtime profile, verified
prepared bytes, exact provider scope/bucket/hardware/secret references and a pinned
host SDK. Source-free known-job reads use the existing retained HF reader, original
keys and store, without restoring a source lease or submission authority. Explicit
`allow_cancel_reconcile: true` must be selected before the original plan;
configured cold cancellation/reconciliation then uses the original known-job
binding and current host permission. Defaults remain false and the separate
read-only opener still refuses these effects. SDK acknowledgment is distinct
from observed terminal state; lost replies stay uncertain without another cancel
or training submission.

Accepted CPU controls include an actual SIGKILL producer and first fresh SQLite
opener after original input/source deletion, with scripted provider state, native
verification and original authority retained. They do not establish live provider
behavior or active-worker continuation. Provider labels/scopes remain retained
configuration declarations, not fresh remote attestation. See the HF template
and [support matrix](../../docs/plans/api-v1-support-matrix.md) for exact limits.

## Preparing genuine HF consumer material

`prepare_hf.py` closes the host setup gap using the existing ingestion,
preparation, source authentication, compiler and HF preview APIs. Start with
`prepare-hf.config.example.json` and the separately declared raw-text fixture;
replace every host-owned path, bucket, scope, quote and named key reference with
actual reviewed inputs. Templates containing placeholders are not ready inputs.

```sh
python examples/training_qualification/prepare_hf.py declare /private/prepare.json
python examples/training_qualification/prepare_hf.py prepare /private/prepare.json
python examples/training_qualification/launch.py plan /private/output/consumer.json
```

`declare` is credential-free and exclusively writes the canonical runtime
member. It binds the selected profile/inventory/image, declared interpreter,
HF roots, configured reset environment and named requirements. The member is
retained in `CoordinatorResolvedMaterial.execution_context`; its SHA256 supplies
`ExecutionSourceV1.deployment_member_sha256`. This is a declared host
runtime configuration, not a staged member or observed provider deployment.
The remote worker still independently enforces its existing source, runtime,
inventory, roots and dispatch contracts.

`prepare` authenticates a genuine clean host manifest and engine Gitlink using
existing bounded public Git reads and a separately named local source-signing
key. That key must differ from both HF provider token references. Resolve it only
for an authorized preparation; never substitute `HF_TOKEN` or place key values
in config. The checked-in remote verifier bounds each Git read to 30 seconds and
source evidence freshness to 600 seconds. This command does not create Jobs,
buckets, mounts, models or provider resources.

Output is create-only. Exact prepared bytes, source lock, declaration, request
and parsed/rederived material are retained before actual `hf_host.preview`.
Only successful consumer-config validation and preview publish `consumer.json`;
closed failures leave retained partial evidence without an advertised ready
configuration. Full public preflight remains deferred until genuine warm host
composition. Existing capabilities and cold lifecycle restrictions are unchanged.

Ordinary HF staging verifies the exact source/input inventory and every member's
bytes through one bounded SDK download group. The original absolute staging
budget also covers upload and readback; batching does not reset it or prove why a
historical attempt failed. Each download keeps its own private directory and
checked file descriptors. Unconfirmed SDK work retains scratch and consumed
claims.

For a retained `stage_reconcile_required` attempt, the existing `show` command
can return authenticated **local** workflow phase and a closed stage diagnostic.
It uses the original authority key, workflow history, Foundation assessment and
material commitments, without a provider client or original input/source files.
A missing diagnostic remains unavailable; this is neither provider job evidence
nor cancellation, recovery, absence proof or permission to replay. Known-job
`show` and native verification keep their existing authenticated reader path.

## Preparing genuine Docker consumer material

`prepare_docker.py` uses the existing public source authentication, immutable
prepared-input verifier, SFT compiler and Docker bridge to generate the exact
request, CoordinatorResolvedMaterial, DockerProfile and consumer preview. Copy
`prepare-docker.config.example.json` and replace all placeholders with reviewed
host-owned inputs. The selected runtime release must admit the exact model,
revision and prepared v1 format; the image profile/report and immutable image
must still be qualified through the existing derived-image workflow. The template
is a candidate configuration, not a live qualification or approval. It does not
upgrade historical runtime profiles or enable logs/cancel/reconcile.

Run from a neutral directory with the installed engine interpreter:

```sh
python -I /absolute/checkout/examples/training_qualification/prepare_docker.py plan /private/prepare.json
python -I /absolute/checkout/examples/training_qualification/prepare_docker.py model-preflight /private/prepare.json
python -I /absolute/checkout/examples/training_qualification/prepare_docker.py prepare-model /private/prepare.json
python -I /absolute/checkout/examples/training_qualification/prepare_docker.py prepare /private/prepare.json
python -I /absolute/checkout/examples/training_qualification/launch.py start /private/prepared/consumer.json
```

`plan` reads declarations without keys, downloads or daemon calls. `prepare`
uses a separately named source-signing key and bounded public Git reads, verifies
an existing published v1 input (at most 2 MiB), recompiles actual material and
exclusively publishes a preview. Full public operational preflight occurs only
in the subsequent existing `start` composition. Preparation does not install,
pull, build, download a model or start a container. Original host manifests and
pushed source evidence must be genuine and fresh. Its source admission helper
is shared with the HF preparer without changing HF canonical outputs.

`prepare-model` is the explicit public-model network boundary. With the template's
`model_output_root`, it consumes a fresh private model claim and prepares the
pinned model before any short-lived execution-source proof is minted. It uses
anonymous upstream metadata, private persistent/SDK scratch, and a fresh artifact
mount's `run/cache`. Optional `model_seed_root` points to an independently held,
private read-only repository cache; each present member is copied into fresh roots
and checked against current pinned Git/LFS metadata. A corrupt, redirected or
changed seed member rejects; absent members use the normal cold download path.
An absent entire pinned revision under an otherwise private seed is also a
cold miss; an existing redirected revision or member rejects.
The seed's old claim and bytes stay untouched and never become new authority.
The native snapshot validator checks the resulting offline path and a complete
bounded file inventory is retained in `model-preparation.json`.

`prepare` first rechecks that receipt and actual model bytes against its declared
configuration, then authenticates fresh pushed source and compiles material. Its
consumer binds the exact receipt bytes; public `start` rechecks the model inventory
before Docker effects. Public start retains its own committed receipt copy.

For a separately reviewed import-only diagnosis, the same preparer exposes
`stage-diagnostic-source` after a fresh `prepare`. It validates the prepared
consumer/material, complete source lock, current clean host/gitlink, and the
still-valid signed source evidence, then uses the existing local source port to
stage the authenticated worker bundle and small prepared input into unused
private roots. A create-only diagnostic claim prevents replay. It makes no
Docker call or public Foundation stage and grants no training authority. The
diagnostic must separately pin the staged source, original and derived closure,
retained model receipt, and all container bounds before any live probe. An
expired source proof refuses staging; a later diagnostic uses only its
separately admitted immutable staged bytes, never the expired proof as training
authority.
Public `plan` authenticates receipt identity but does not attest current model
bytes; `start` performs that live check.
Later cold observe/cancel/reconcile/verify/download authenticate that retained
receipt, configuration and run identity without reopening weights or depending
on the preparation root. The source mount remains absent until the source stage.
Writable runtime directories are private and disjoint; the exact prepared-input
slot is never adopted. Each model and source root is one-use. A failed or uncertain
model preparation cannot be replayed under the same claim.
Vendor download behavior is not a hard job deadline or bill cap. No manual weight
upload, source inventory widening or restored training authority is involved.

Run `model-preflight` with the exact interpreter intended for `prepare-model`
before issuing short-lived source material. It checks the existing Hub SDK
signature contract without constructing a client, downloading, or creating
directories/claims. A base consumer installation can lack this optional SDK;
select an independently verified model-helper environment rather than upgrading
the pinned trainer or silently installing packages. `prepare-model` repeats
this check before its one-use claim. Compatibility is not package provenance,
an inventory lock, network readiness, or model/training qualification.

Create the selected source/artifact roots' necessary private parents before
preparation; the preparer validates them and creates only its own leaf roots.
An existing model-preparation claim is consumed even if no receipt was produced.
Do not replay it. Failures now retain the model preparer's existing closed
`model_preparation_stage` when available, without exception messages or paths;
historical generic failures cannot be retrospectively assigned a stage.

Docker staging creates only the authenticated empty project directory plus its
necessary parents, retaining their descriptors through submit checks. Extra files,
directories, symlinks, replacement and permission changes still refuse. This
allows the real source worker's root binding without manufacturing project data.
Native artifact role paths remain separate from the model cache; successful
preview or model preparation does not establish a real SFT update.

The same checked-in consumer has a stdlib-only `probe-filesystem` diagnostic for
a reviewed CPU container plan. It executes exact hash-verified native writer bytes
on fixed nonsecret data and reports actual modes/UID/GID and unchanged initial
umask. A fresh host-owned output and bounded host read are required to establish
accessibility. It does not create a TrainingAPI receipt or qualify training/GPU
behavior. The base image's root user alone does not prove output unreadability;
no UID API or permission guard has been relaxed.

### Owned Modal producer interruption (Linux caller example)

`modal_lifecycle.py` supervises the existing normal `tuner train` CLI. It does
not implement provider cancellation, grants, source authentication or a lifecycle
store. Review the genuine pushed host, normal plan/preparation and finite live
scope first. Configure absolute paths using `modal_lifecycle.example.json`, with
fresh empty private `state`, `tmp` and supervisor-output directories at mode0700.
The selected installed Python/launcher and explicit env file remain caller-owned;
no shell or `env` executable is launched, ambient credential/SDK overrides are
not forwarded. The existing train CLI resolves its explicitly selected env file.

Run the installed interpreter against this checkout-owned example:

```sh
/ABSOLUTE/PINNED-LAUNCHER/bin/python -I /ABSOLUTE/EXAMPLES/modal_lifecycle.py --config /ABSOLUTE/PRIVATE/supervisor.json
```

This command starts one effectful normal training attempt. A fsynced exclusive
intent is consumed before spawning. Never rerun after failure or uncertainty.
The supervisor owns a direct-Python child in a separate session, retains its
pidfd/start time and verifies the configured executable before any signal. Its
error, EOF or deadline closes local handles without killing a running producer;
losing the controlling TTY does not send that producer SIGHUP. Preserve the birth
record and original attempt rather than guessing or adopting a PID.

The control stream accepts one bounded JSON line at a time:

```json
{"operation":"status"}
{"operation":"interrupt","run":{"run_id":"ACTUAL_RETAINED_RUN","project_ref":"standalone-training"}}
```

Only an explicit interrupt request can signal the held child. It requires the
existing retained-input parser, matching configured recipe/request, engine and
scope, unchanged private state identity, and a stable bounded private progress
file for that exact run. Numeric metric records are distinguished before reading
`step`; partial writes wait, invalid complete records refuse the whole snapshot,
and nonmonotonic/exhausted/late-step records cannot reuse an earlier trigger.
Select bounds from the admitted recipe: allow the metric count plus the existing
capture-exhausted status record. No ordinary logs capability is promoted.

This is **host-origin decoded progress custody**, not an independently signed
file or pre-interruption Foundation/pending attestation. While the original
producer holds its exclusive journal lock, another public reader cannot open it.
After interruption and reaping, use the existing public `runs observe` with the
original keys and state to authenticate the original submit/call/current binding.
Only actual fresh pending permits a separately requested existing public
`runs cancel`; acknowledgment is not terminal cancellation. The supervisor never
calls cancel or an SDK. Natural completion, unknown state, lost custody or missing
progress is inconclusive and never authorizes replay. Existing provider/component
bounds still apply; the supervisor wait bound does not terminate remote work.

Status output also uses nonblocking bounded writes: a nonreading pipe/TTY cannot
hold the supervisor past its wait deadline (a final closed refusal has at most a
one-second output tail). An attempted signal is consumed even on error. Only a
returned owned signal followed by reaping the expected SIGKILL writes a completed
interruption receipt; an external SIGKILL or uncertain signal is not normalized
to successful interruption. Ordinary producer exit0 remains completion, not
cancellation evidence.
