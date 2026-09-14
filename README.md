# Dedicated Modal chat smoke consumer

This is the small **host project** used to exercise Synaptic Tuner as a submodule.
It lives on `smoke/modal-chat-consumer` in the same GitHub repository as the
engine, but has an independent root history. It must not be merged into main
or an engine feature branch.

The `synaptic-tuner` gitlink selects engine commit
`9dca62b2a2faf18d9164fa820cf0348f83e663c3`, advertised by
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

Use an existing compatible Python environment with Modal 1.5.4 and the engine's
host dependencies. No local Docker or model-weight download is involved.
The explicit named Modal profile supplies credentials internally; an optional
`--hf-token-env-file /absolute/path/to/existing/.env` reads only its `HF_TOKEN`.
Otherwise `HF_TOKEN` is inherited. Never put credential values in arguments.

```bash
python -B synaptic-tuner/examples/modal_chat/launch.py \
  --project-root /absolute/path/to/modal-chat-consumer \
  --configuration configuration/smoke.json \
  --mode qualify-training --modal-profile synaptic-labs
```

This provisions the configured fresh Volumes and runtime Secret, deploys the
fixed worker, submits once and saves authenticated native training/artifact
evidence. A launch claim is permanent, including after failure. Do not rerun an
ambiguous attempt or rename it to evade reconciliation. After a known terminal
failure, review its evidence and deliberately commit a new isolated attempt.

The default `--mode check` makes no cloud calls. `--mode train-chat` requires
current inference-image evidence and qualified public observe/artifact
capabilities, then verifies that same run, saves one reply and confirms the exact
owned Sandbox stopped. The current configuration's historical CPU capture is
not qualification of updated runtime source; refresh it before chat. Operator
budgets and timeouts do not guarantee a provider-side billing cap.

## Boundaries

Use only `synaptic-smoke-v1`. Model preparation happens on Modal; operators
do not stage weights locally. Training outputs stay in provider-native storage,
and the consumer keeps its one-shot claims and result records in `.synaptic/state/`.
The example is single-process: reopening its database refuses automatic replay
but does not reconstruct all Foundation/coordinator state.

This branch is executable smoke preparation, not a claim of a successful GPU smoke.
Attempt `modal-chat-20260914-a` created its three Volumes and runtime Secret,
then exited during deployment without reaching training submission. Scoped
readback found the prior app deployment unchanged and the exact new function
absent; app logs recorded an image build but did not establish the failure cause.
Its private journal and resources are preserved. The current configuration
selects a separate `modal-chat-20260914-b` attempt with new resource/function
names and closed deployment diagnostics; it must not adopt or erase attempt A.
EHR is not used or changed. No publication, live endpoint, or teardown of an
older app is implicit in this fixture.
