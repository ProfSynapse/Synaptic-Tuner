# Modal inference workload binding

Engineering checkpoint, 2026-09-10. This extends native source admission with
the intended pinned model identity. It is not chat mutation authority, a model
load test, a serving deployment or live Modal qualification.

## Existing byte chain

The resolved workload bytes become the bundle's `workload.json` member. Bundle
validation replays the existing recipe compilation and requires exact equality
with the retained resolved material. The worker passes those bytes as trainer
stdin. Runtime validation requires the same canonical compiled payload, and a
successful runtime writes that payload unchanged as `workload_record`, recording
its size and SHA-256. There is no newline or archive envelope in that semantic
artifact. Model and tokenizer archives remain separate artifacts.

`bind_modal_inference_workload` takes a fresh same-composition native source
binding. It reauthenticates the retained Foundation/stage/launch proof through
`prepare_modal_submit_dispatch` and reuses the semantic bundle parser. It must
match the workload bytes to the verified native workload-record entry and bind
the exact command, deployment, client scope, Volumes and evidence-key reference.
Snapshots guard against collaborator changes during admission. These immutable
projections are evidence, not a new authority service, database or durable grant.

The intended model fields come from the matching configuration and identity
records in that workload. Model and tokenizer revisions must be immutable and
equal, as the existing runtime requires. `load_in_4bit` records the training
configuration; it does not choose inference precision. Actual full-model/LoRA
kind and model usability still require remote artifact admission and loading.
Neither operator-side artifact streams nor model preparation are part of this
function. The future remote adapter must compose it after current source
admission, not accept arbitrary serialized source claims as authentication.

## Qualification

Implementation and independent source review passed locally. Combined
provider-free qualification passed **2,339 tests in 329.86 seconds**, using
clean CPython 3.12.9 / pytest 8.4.2 without Modal or PyTorch. This is the defined
adapter/coordinator/Foundation/training/runtime/inference selection, not every
test in the repository. The initial three workload tests and eight
source-integration tests passed independently (11 tests in 23.82 seconds).
The acceptance lane subsequently passed a 13-case workload selection in 32.12
seconds, then added a separately passing artifact-Volume denial. The lead added
two post-authentication mutation regressions for the retained and owned launch
copies, bringing the integrated workload module to 16 cases; all are included
in the combined result. Both source-lock checks remain CURRENT (97 Modal
bootstrap pins and 66 offline trainer members), formatting passes, and canonical
skill mirrors are synchronized. The wheel check remains pending at this source
checkpoint and is recorded separately after the immutable commit is packaged.
The old shared launch fixture's differing model/tokenizer revisions and
synthetic workload-record body are not a valid positive for this additional
check. Workload-binding positives must derive matching material before bundle
compilation/signing and obtain their source through the real native binder.

No provider access, credentials, cloud objects, paid execution, push, merge or
EHR changes are part of this slice. Separate chat resource/lifetime policy,
mutation binding, locked remote inference execution and live qualification
remain open.
