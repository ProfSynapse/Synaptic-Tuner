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
keys and store, without restoring a source lease or submission authority. Cold
cancel/reconcile remain unavailable. Provider labels and scopes are retained
configuration declarations, not fresh remote attestation. No fresh-process,
SIGKILL, active-worker recovery or live proof is claimed by this example's CPU
controls. See the HF template/adapter for its exact finite inputs.
