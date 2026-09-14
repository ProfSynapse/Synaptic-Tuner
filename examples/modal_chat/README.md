# Minimal consumer: train, retain evidence, chat once

This example belongs to the consuming-project layer, not the engine runtime.
It uses `APIHost.training`, `APIHost.runs`, and `open_run_chat` without changing
EHR, adding a provider-specific engine CLI, or staging model weights locally.

`launch.py` is the checked-in candidate launcher for the dedicated minimal
consumer. It composes the actual authenticated host and selected chat runtime;
its cloud path is not yet live qualified. No test fixtures, invented training
outputs, or arbitrary model directories may replace its inputs.

## Launcher

Effectful modes require CPython **3.11.14**, matching the packaged training
runtime pin, and the existing hash-pinned launcher dependencies. The deployment
uses a serialized function; a local Python 3.12 launcher is incompatible with
the measured Python 3.11.14 image. The launcher rejects a mismatch before
credentials, attempt storage or cloud resources. This checks host compatibility,
not the remote executable hash or successful deployment. `--mode check` remains
provider-free and usable on other supported project Python versions.

For Linux x86_64, prepare a separate launcher environment using the existing
Python/uv commands (never install over the training environment). Supply an
already installed CPython 3.11.14 executable and an absent private venv path:

```bash
/absolute/python3.11 -I -m venv --without-pip /absolute/new-launcher-venv
uv --no-config --no-cache pip install \
  --python /absolute/new-launcher-venv/bin/python \
  --no-deps --require-hashes --only-binary :all: \
  -r /absolute/consumer/synaptic-tuner/examples/modal_chat/requirements.lock
```

Use that venv's Python for the commands below. This installs the lightweight
launcher only, not model weights, a local GPU runtime or Docker. The consumer
lock includes the existing 37-package remote launcher lock and adds Requests'
three missing host dependencies; it does not change the remote runtime lock.
Do not use a
newer CLI/Python as a substitute for the reviewed pins. Modal documents the
serialized-function constraint in its [Python compatibility guidance](https://modal.com/docs/guide/jupyter-notebooks#known-issues).

Run the file from a clean, published consumer's exact engine submodule. The
default `--mode check` validates local source/configuration and the reviewed
inference image without credentials or cloud calls:

```bash
python -B synaptic-tuner/examples/modal_chat/launch.py \
  --project-root /absolute/consumer \
  --configuration configuration/smoke.json
```

The explicitly selected `--mode qualify-training --modal-profile NAME` creates
fresh named resources, deploys, submits once through the public TrainingAPI,
waits for the exact provider call, and retains native authenticated terminal and
five-artifact verification evidence. It stops before chat and does not claim
public RunsAPI qualification. This training-only mode does not depend on an
inference image. `--mode train-chat` refuses before credentials or cloud mutation
while the public observation/artifact capability flags remain disabled. Once
qualified, it requires successful public outcome/verification before one chat
turn and exact owned-Sandbox stopped readback.

Modal credentials come only from the explicitly named SDK 1.5.4 profile, with
environment overrides disabled. `HF_TOKEN` is inherited, or read from the one
existing file explicitly selected by `--hf-token-env-file /absolute/file`.
No credential value is accepted in argv or saved in state. The runtime Secret
contains only the model token and a newly generated in-process evidence key;
the model-serving child remains credential-free. Fresh workspace billing rates
are saved with operator authorization, not represented as a provider billing
cap. Training estimation is GPU-only and explicitly excludes CPU/memory/builds
and storage. All resources retain explicit timeouts and retries remain zero.

Set the training configuration's `runtime_environment.PATH` explicitly to
`/usr/bin:/bin` for this pinned Linux profile. Offline trainer children do not
inherit PATH: Triton's C compiler needs it to find the linker. Keep this value
in authenticated consumer configuration, not an ambient-environment fallback.

The CLI prints closed progress codes, not raw SDK/worker exceptions or model
responses. Responses and immutable evidence go under the manifest's private
state root. After failure, inspect the retained exact identities and evidence;
do not rename the attempt or delete its claims to resubmit.

Deployment failures retain non-authorizing phase, closed exception categories,
bounded allowlisted traceback locations, and known provider object IDs in the
existing private catalog. They never retain exception messages, source lines,
locals or credentials. A diagnostic is not deployment acceptance, permission to
retry, or proof that a provider object stopped. If diagnostic persistence fails,
the original error remains primary and any captured record remains available in
the owning process only.

## Saved state and ownership

The example owns its private SQLite file and immutable catalogs. A durable
attempt claim precedes the training or chat path. Reusing the same attempt is
refused, including after a crash or successful completion. An unknown outcome
is not permission to retry with another attempt identifier.

This is a single-process example. Its durable claim and saved records do not
implement restart recovery of the entire coordinator/Foundation state. Keep
the original process and composed stores alive through training, verification,
and chat. After interruption, inspect the exact retained provider identity and
evidence; do not infer shutdown or launch a replacement.

Keep the database under a consumer-owned `private/` directory outside committed
source. Never store credential values in configuration, argv, the database, or
captured exception text. The database is trusted consumer storage, not a hostile
shared-Volume security boundary. Credentials stay in the existing authority
ports and explicitly selected provider mechanisms.

## Required composition

Use `synaptic_tuner.api.v1.modal.compose_modal_coordinator` with the consumer's
configuration, authorities, stores, and explicit Modal client. Wrap its training
and runs operations in `APIHost`, using the same clock throughout. For Modal
chat, compose the existing `ModalRunChatRuntime` with its source/workload binders,
authenticated inference configuration and fresh quote, separate grants,
Foundation broker, SDK transport, and ready-lease handoff. The example supplies
neither fake authorities nor ambient provider fallback.

`ModalChatTrainingRequests` in `requests.py` bridges a configured rich
`TrainingService` into the coordinator's loader, resolver, and run-identity
ports. It uses the real input parser and recipe compiler, allocates identities
at load, and retains canonical request and resolved-material bytes in the
consumer catalogs. Cached resolution checks the original request, project,
and run bindings without resolving source provenance again. This permits
preparing the exact resolved request before constructing the request-scoped
Modal adapter. The host still supplies the actual source resolver and identity
allocator; the bridge does not manufacture provenance or authorize a job.

`ModalChatEvidenceReplay` in `replay.py` implements the source finalizer's
durable replay port using the same SQLite catalogs, without another table or
database. Each purpose/challenge pair admits only the identical issuer,
evidence reference, audience, payload digest, and expiry thereafter. The
existing finalizer remains responsible for authenticating and checking the
freshness of evidence before replay admission.

`deployment.py` owns a deployment made with the fixed engine builder and an
explicit Modal 1.5.4 client. Workspace lookup and exact existing-environment
lookup establish scope without listing other environments. A permanent claim
precedes deployment; a separate deterministic whole-app claim blocks a renamed
attempt or function from redeploying it. A returned deployment is recorded even
if subsequent definition metadata cannot be verified. Its acknowledged app,
function, locally acknowledged definition,
image, and Volume identifiers are saved before current-deployment readback. Each
later observation requires the same observed deployment generation; it cannot adopt an
unrelated deployment or infer shutdown after an ambiguous failure. This is
provider-observed ownership of the consumer's submitted static configuration,
not an independent server-side attestation of every function option.
Persisted deployment acknowledgements are audit evidence, not a restart/adoption
API. After an interrupted process, reconcile that exact deployment manually;
do not choose a new attempt identifier and deploy again. Failure to persist an
acknowledgement leaves only a non-authorizing `candidate_receipt` in memory,
and both the observer and facade remain unavailable.

Correction (2026-09-14): ordinary Modal name lookups and current app layouts
omit the immutable definition ID. This was measured after the Python-aligned
attempt successfully returned from deployment; the subsequent identity check
failed before training submission. Do not treat an omitted ID as verified, or
require a subscription upgrade just to obtain version-pinned invocation.
The consumer instead observes the exact app/environment generation before
deployment, requires exactly one generation increment afterward, and checks
the exact private single-function layout. Each later observation brackets its
layout read with the same deployed app/generation. Replacement, redeployment,
rollback or extra functions/classes fail closed. A locally returned definition
ID remains acknowledged evidence, not a claim that floating lookup returned it.
These current-state checks do not make later invocation version-pinned or
eliminate an external administrator's race after the last observation.
See [Modal's lookup semantics and plan constraints](https://modal.com/docs/guide/trigger-deployed-functions#version-pinned-lookups).

Training acceptance is not completion. Observe the exact returned run through
`RunsAPI`; successful verification and its authenticated five-artifact inventory
must precede chat. The selected adapter reverifies and prepares on its execution
machine. A chat turn uses one context-managed session and one explicit prompt.
Saved training artifacts are not deleted when the session ends.

Provider timeout, idle timeout, session deadline, and owned cleanup remain
mandatory. A local journal or context exit is not a hard billing guarantee.
Use a current quote and bounded resources within the operator's cumulative
budget; changing an attempt identifier does not renew budget or authority.

## Remaining live prerequisites

The current Modal source finalizer requires host/superproject provenance: a
clean, pushed consumer commit containing an exact engine gitlink, and evidence
binding both repositories. A folder inside a standalone engine checkout does
not satisfy this. Copy the example into a real consumer; do not pretend the
engine is its own host or disable source verification.

Live qualification also needs the final inference image with its reviewed
runtime/dependency/source resources, actual model artifacts, authenticated
configuration and grants, and confirmed exact-Sandbox termination. The previous
engine-installed CPU inspection is package evidence only, not proof of a
successful model response. No push, new remote repository, deployment, or paid
job happens merely by importing this example.

## Embedding the example

These are calls inside an already-composed consumer, not shell commands or a
complete host factory. `private_directory` is an existing canonical private
POSIX directory; this example targets Linux/WSL, not native Windows SQLite
locking. Open the storage explicitly and retain it for the consumer lifetime:

```python
from examples.modal_chat.storage import ModalChatStorage
from examples.modal_chat.consumer import submit_training_once, chat_once

with ModalChatStorage(private_directory / "smoke.sqlite3", "my-consumer") as storage:
    submitted = submit_training_once(
        host.training,
        storage,
        attempt_ref=training_attempt_ref,
        request_json=canonical_training_request,
        provider=selected_provider,
    )
    # Continue the host's RunsAPI observation/verification workflow here.
    # Do not call chat merely because submitted.start.accepted is true.
    # After that exact run is verified, and while this host remains composed:
    turn = chat_once(
        host.runs,
        storage,
        attempt_ref=chat_attempt_ref,
        run=submitted.start.run,
        runtime=authenticated_chat_runtime,
        prompt="Say hello in one short sentence.",
    )
```

The host supplies every named input above. The request digest is retained,
not raw training configuration. Chat response text is private consumer output;
do not put credentials into prompts. The helpers do not print raw provider
exceptions or automatically publish the model or conversation.

`chat-results` records that a response was received; it is saved before cleanup.
`chat-contexts` is written only after normal context exit and explicitly records
`provider_shutdown_proof: false`. A saved reply without a context record means
cleanup was not successfully recorded, not that the model call should be retried.
The original runtime retains any known cleanup lease while its process lives.

## Verification status

Provider-free tests cover actual public training start through the consumer
composition, one simulated spawn and convergent repeated start, real HMAC
verification, `open_run_chat` with simulated execution, permanent claims,
cleanup failures, canonical fractional SFT parameters, bounded replies, and
request/material identity substitution. Independent consumer/launcher and bundle
parser review passed. These tests do not establish a real GPU response.

Correction (2026-09-14): the bundle parser now accepts finite fractional values
only in the typed workload member; generic Foundation commands remain
integer-only. This changes locked source and invalidates the earlier inference
image for the updated source. Requalify the final image before chat. Historical
CPU records remain historical evidence, not current serving admission.
