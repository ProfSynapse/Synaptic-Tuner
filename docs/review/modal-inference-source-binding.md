# Modal inference source binding

Engineering checkpoint, 2026-09-10. Engine-only; provider-free qualification is
recorded below when complete. This is slice 2a of the bounded Modal chat plan,
not an operational inference service or permission to allocate a GPU.

## Boundary

Public run artifacts identify roles, sizes and hashes, not provider placement.
The native Modal reader additionally authenticates the exact client scope,
deployment, Foundation submit lineage, job, effect, Volumes and completion
inventory. `ModalCoordinatorRunReader.native_artifacts` preserves that native
inventory alongside its generic manifest using the existing authenticated read
path. It does not stream artifact bodies or add another reader implementation.
The binder validates both and retains reconstructed immutable projections,
canonical read-request/native evidence bytes and manifest/source digests, not
aliases to the nested request, manifest or native inventory objects.

`ModalInferenceSourceBinder` correlates that proof with current public
reverification and the consumer's retained verified workflow. Complete manifest
equality is required: matching public artifact hashes alone would not establish
the same provider attempt. A changed workflow during admission is a rejection,
not an automatic retry. Construction must use the same consumer's `RunsAPI`,
workflow store, Foundation ports, authenticators and reader; dependency injection
is a trusted composition boundary, not proof that arbitrary collaborators are
honest. An alternate `RunsAPI` instance is rejected before calls.

No new database, authority service, effect kind, command issuer, public CLI,
local weight staging, model downloader or publication step is introduced.
The existing generic `provider_run_read_request` derives the read request from
the retained submit record and assessment. Existing reader/transport checks
remain responsible for provider evidence authentication and bounded native reads.

## What this does not prove

Source binding is a current artifact-placement admission, not a durable serving
grant or proof that mounted bytes will remain unchanged. Remote preparation must
reverify the exact mounted bytes before use. The binding does not claim actual
full-model/LoRA kind, model usability, an inference image, a Sandbox identity,
model access credentials, a quote or a cloud billing deadline.

Remaining slice 2b must bind retained workload/model commitments and a separate
chat execution/resource policy to unchanged Foundation stage/submit/cancel
authority. The separate inference worker, authenticated client, bounded owned
lease and live qualification remain later slices. The training runtime lock is
not an inference lock.

## Qualification

The isolated acceptance lane passed 18 tests in 3.45 seconds against the lead
source using CPython 3.12.9 / pytest 8.4.2. Its real reducer fixture covers
verification, complete manifest correspondence, no body reads, workflow drift,
output ownership, exact role paths and Volume-entry identities, and sanitized
collaborator failures. A separate integration lane passes through real Foundation
lineage, the existing semantic launch bundle and signed control-evidence
transport with a test SDK facade. Its initial seven cases passed in 17.87 seconds;
an additional fresh-process import case excludes Modal and ML dependencies.
The integrated provider-free selection passed **2,323 tests in 312.23 seconds**.
This is the prior 2,297-test selection plus 18 acceptance and eight integration
cases. The final ownership assertion compares copied field values and separately
passed the 18-case module in 3.88 seconds. Independent source review passed;
its clean combined acceptance/integration run passed 26 tests in 22.70 seconds,
and its integration/existing-reader run passed 22 tests in 20.98 seconds.
Independent scope/document review also passed. The Modal 97-pin and offline
66-member lock checks remain `CURRENT`, with no lock changes, and the canonical
skill mirrors are synchronized. Exact-commit wheel qualification follows the
local source checkpoint.

These fixtures use consumer authority/verifier test doubles, not live provider
authentication. The shared launch fixture is a semantic bundle fixture, not a
real model-training run: its model/tokenizer revisions differ and the producer
fixture supplies synthetic artifact bodies. Those bytes are never downloaded by
the binder. Later workload/model admission tests must use equal-revision,
runtime-valid material and compare the authenticated workload-record descriptor
with the exact retained workload bytes before claiming that additional proof.

No provider access,
credential access, cloud object creation, paid run, push or merge was performed
for this checkpoint.
