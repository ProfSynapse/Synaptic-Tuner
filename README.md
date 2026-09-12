# Dedicated Modal chat smoke consumer

This is the small **host project** used to exercise Synaptic Tuner as a submodule.
It lives on `smoke/modal-chat-consumer` in the same GitHub repository as the
engine, but has an independent root history. It must not be merged into main
or an engine feature branch.

The `synaptic-tuner` gitlink selects engine commit
`b3d916b16a110f884ae394fe3a25a138aac7c863`, advertised by
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
IDs, credentials, signed grants, placeholder runtime hashes, model weights, or
fabricated training evidence are checked in.

Use the engine's [minimal consumer example](synaptic-tuner/examples/modal_chat/README.md)
for `submit_training_once`, `chat_once`, and the private attempt/result store.
This source fixture is not yet a standalone live launcher: authenticated host
composition, training configuration/data, final inference image commitments,
a verified trained run, and actual chat/cleanup evidence remain required.
Importing the example or cloning this branch starts no cloud job.

From the checkout, run the provider-free source checks in an existing compatible
Python environment with the engine's manifest dependencies installed:

```bash
python -B -m unittest discover -s tests -v
```

These tests use the real local source inspector. They deliberately do not attest
remote publication, contact Modal, load credentials, or authorize a job.

## Boundaries

Use only `synaptic-smoke-v1`. Model preparation happens on Modal; operators
do not stage weights locally. Training outputs stay in provider-native storage,
and the consumer keeps its one-shot claims and result records in `.synaptic/state/`.
The example is single-process: reopening its database refuses automatic replay
but does not reconstruct all Foundation/coordinator state.

This branch is source preparation, not a claim of a successful GPU smoke.
EHR is not used or changed. No publication, live endpoint, or teardown of an
older app is implicit in this fixture.
