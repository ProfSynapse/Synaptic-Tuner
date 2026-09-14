# Dedicated Modal chat smoke consumer

This is the small **host project** used to exercise Synaptic Tuner as a submodule.
It lives on `smoke/modal-chat-consumer` in the same GitHub repository as the
engine, but has an independent root history. It must not be merged into main
or an engine feature branch.

The `synaptic-tuner` gitlink selects engine commit
`6ab0e7262c8f97b38aaca7143bf05273832af70f`, advertised by
`smoke/modal-chat-engine`. Both origins are
`https://github.com/ProfSynapse/Synaptic-Tuner.git`.
Different commits and an explicit gitlink keep host and engine provenance
distinct; the Modal source verifier checks both. Nested submodules are disabled.

## Obtain the source

```bash
git clone --branch smoke/modal-chat-consumer --single-branch https://github.com/ProfSynapse/Synaptic-Tuner.git modal-chat-consumer
cd modal-chat-consumer
git submodule update --init -- synaptic-tuner
git ls-tree HEAD synaptic-tuner
git -C synaptic-tuner rev-parse HEAD
```

The last two commands must agree on the exact engine commit. Do not use
`submodule update --remote`: the tracked branch advertises provenance, while the
gitlink selects the reviewed bytes.

## What this branch provides

`synaptic.yaml` is a strict host manifest with private consumer output roots,
HTTPS/GitHub-only source policy, and no source or engine writes. No deployment
credentials, signed grants, model weights, or fabricated training evidence are
checked in. `configuration/smoke.json` selects dedicated object names and a
two-step, pinned SmolLM2 LoRA workload; `data/smoke.jsonl` contains eight tiny
training examples. Fractional training values belong in this configuration.

Use the engine's [minimal consumer example](synaptic-tuner/examples/modal_chat/README.md)
for `submit_training_once`, `chat_once`, and the private attempt/result store.
The checked-in consumer launcher composes the actual public training API and
authenticated provider collaborators. Its native qualification mode exists to
measure training/artifact evidence before enabling public read capabilities.
It does not claim to have completed public chat.
Importing the example or cloning this branch starts no cloud job.

From the checkout, run the provider-free source checks in an existing compatible
Python environment with the engine's manifest dependencies installed:

```bash
python -B -m unittest discover -s tests -v
```

These tests use the real local source inspector. They deliberately do not attest
remote publication, contact Modal, load credentials, or authorize a job.

## Run the bounded smoke

Use an isolated CPython 3.11.14 environment with Modal 1.5.4 and the engine's
`examples/modal_chat/requirements.lock` installed with hash verification and
no dependency resolution, as documented in the engine example README. A Python
3.12 launcher cannot serialize this worker for the pinned Python 3.11 image.
No local Docker or model-weight download is involved.
The explicit named Modal profile supplies credentials internally; an optional
`--hf-token-env-file /absolute/path/to/existing/.env` reads only its `HF_TOKEN`.
Otherwise `HF_TOKEN` is inherited. Never put credential values in arguments.

```bash
python -B synaptic-tuner/examples/modal_chat/launch.py \
  --project-root /absolute/path/to/modal-chat-consumer \
  --configuration configuration/smoke.json \
  --mode train-chat --modal-profile synaptic-labs
```

This provisions the configured fresh Volumes and runtime Secret, deploys the
fixed worker, submits once, verifies the same run through the public outcome,
verification and artifact paths, then saves one chat reply and confirms the exact
owned Sandbox stopped. A launch claim is permanent, including after failure. Do not rerun an
ambiguous attempt or rename it to evade reconciliation. After a known terminal
failure, review its evidence and deliberately commit a new isolated attempt.

The default `--mode check` makes no cloud calls. `--mode train-chat` requires
current inference-image evidence and qualified public observe/artifact
capabilities, then verifies that same run, saves one reply and confirms the exact
owned Sandbox stopped. The current configuration selects the refreshed ea66c4c
CPU capture and stopped-Sandbox readback, reviewed against this pin's unchanged
runtime closure. That check is not proof of GPU serving or a chat reply. Operator
budgets and timeouts do not guarantee a provider-side billing cap.

## Boundaries

Use only `synaptic-smoke-v1`. Model preparation happens on Modal; operators
do not stage weights locally. Training outputs stay in provider-native storage,
and the consumer keeps its one-shot claims and result records in `.synaptic/state/`.
The example is single-process: reopening its database refuses automatic replay
but does not reconstruct all Foundation/coordinator state.

This branch is a dedicated smoke fixture, not general release qualification.
Attempt `modal-chat-20260914-a` created its three Volumes and runtime Secret,
then exited during deployment without reaching training submission. Scoped
readback found the prior app deployment unchanged and the exact new function
absent; app logs recorded an image build but did not establish the failure cause.
Its private journal and resources are preserved. Attempt `modal-chat-20260914-b`
also failed before training submission. Its closed diagnostic records an
`InvalidError` at FunctionCreate, a precreated function ID and the built image;
it does not prove a published deployment or provider shutdown. Image metadata
reported CPython 3.11.14 while the launcher used 3.12.9, a documented serialized
function incompatibility. The exact server rejection reason was not retained.
Attempt `modal-chat-20260914-c` used CPython 3.11.14 and successfully returned
from deployment at app version 7, then failed the local identity readback
before training submission. Ordinary named-function and layout metadata omit
the immutable definition ID; the version-pinned lookup was unavailable.
The corrected consumer verifies the current app generation and exact private
layout, without claiming version-pinned invocation. Attempt
`modal-chat-20260914-d` passed deployment verification and submitted one training
call. That exact call terminated with failure; its Modal logs identified an
integer-only workload reparse before source preparation. The corrected engine
uses the bounded finite-number workload parser throughout worker and inference
admission, preserving strict command/evidence checks. Attempt
`modal-chat-20260914-e` reached source preparation and returned a terminal
worker failure (code 124). Its saved operation log identifies model preparation;
the runner incorrectly required the cache's run ID to start with `run-`.
The next engine binds the cache to the exact authenticated source run ID.
It also hydrates lazy Modal call handles before checking their IDs, so the
launcher can wait for remote work rather than prematurely reading evidence.
Attempt `modal-chat-20260914-f` loaded the model, then failed in Triton's C
helper compilation: GCC could not find its linker. The offline child intentionally
does not inherit its parent environment; the consumer now explicitly configures
`PATH=/usr/bin:/bin`. A credential-free local compiler probe reproduced the
failure without PATH and succeeded with that value. F also exposed a delayed-read
bug: generating a new timestamped Foundation assessment mismatched the retained
submit binding. The engine now authenticates and reuses the exact retained
assessment for qualification, public reads and chat artifact admission.
F's diagnostic readback does not replace authenticated qualification, which its
launcher did not retain. Attempt G failed at FunctionCreate before training:
its image metadata reported no Python version after the PATH override omitted
the pinned interpreter directory. The dedicated app's generation/layout remained
unchanged and G's selected function was absent. The configured PATH is now
`/opt/conda/bin:/usr/bin:/bin`, retaining Python/pip discovery and system tools.
This is a configuration correction, not a runtime/image pin change or success
claim. Attempt `modal-chat-20260914-h` subsequently passed native authenticated
training qualification and verified all five roles: final_model, tokenizer,
training_lineage, training_metrics and workload_record. Its aggregate report
SHA-256 is `a4f3e19b17a89e20010c5cfa495ebfe8adc2c64a14cf73e71e69b0ba2612d30b`;
the engine retains its bounded summary in
`docs/review/evidence/modal-training-qualified-20260914h.json`. H did not run chat.
Attempt I passed public training outcome and artifact verification, then failed
during chat before a serving SUBMIT binding or Sandbox ownership was retained.
No reply/context was saved; its empty cleanup record is not shutdown proof.
The original failure location was not retained. This engine pin adds bounded,
closed host diagnostics and a verified workflow snapshot without changing the
qualified runtime closure or its CPU image evidence. No root cause is claimed yet.
The current configuration selects a separate `modal-chat-20260914-j` instrumented
train-chat attempt with new resource/function names. Earlier attempts' claims or resources
may not be adopted or erased.
EHR is not used or changed. No publication, live endpoint, or teardown of an
older app is implicit in this fixture.
