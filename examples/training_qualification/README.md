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
