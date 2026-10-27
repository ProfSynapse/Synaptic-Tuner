# Native GRPO input preparation on the Linux host

This example wraps the existing `TrainingInputPreparationServiceV1`,
`GRPOTrainingInputNormalizerV1` and `GRPOPreparedInputFormatVerifierV1`. It needs
only the engine's CPU host dependencies. It selects no runtime/provider,
downloads nothing, imports no ML library and never constructs an admission.
The existing `prepare-dataset` CLI accepts SFT dataset-prep schemas; the `train`
GRPO preparation join requires a Modal profile. Neither is a native-only input
preparation command. This example supplies that missing host composition.

The checked-in two short color-choice rows are a **controlled synthetic
mechanics fixture**, not quality evaluation or research data. Configuration
declares batch four/group four, three updates, 64 prompt/8 completion/128 total
token bounds, unit temperature, thinking disabled, rank-four q/v LoRA, and a
weight-one raw-response blue-word reward. The separate controls proposal declares
last-two-layer selection and constant AdamW at 1e-4. Only config contains these
choices. Root must review/revise the real controls and bounds before any ML use.
There is no guarantee of grouped reward variation: retain constant-reward or
failed/uncertain attempts without automatic retry or replay.

## Actions

Run on Linux, using an existing CPU environment. Copy the configuration and seed
into a **fresh private directory directly under a safe owner-controlled parent**.
All ancestors must pass `NativeTree` checks; a group-writable cache ancestor is
refused. Do not chmod shared parents to make it pass. Paths in the config may be
absolute, or relative to the configuration file. The output parent must already
be owner-private; the output directory itself must not exist.

For root's proposed host staging location (choose a fresh name if it exists):

```bash
mkdir -m 700 /home/profsynapse/native-grpo-preparation-20261027-r1
cp examples/training_qualification/native_grpo_preparation/configuration.example.json /home/profsynapse/native-grpo-preparation-20261027-r1/configuration.json
cp examples/training_qualification/native_grpo_preparation/seed.jsonl /home/profsynapse/native-grpo-preparation-20261027-r1/seed.jsonl
cp examples/training_qualification/native_grpo_preparation/qualification-controls.example.json /home/profsynapse/native-grpo-preparation-20261027-r1/qualification-controls.example.json
chmod 600 /home/profsynapse/native-grpo-preparation-20261027-r1/configuration.json /home/profsynapse/native-grpo-preparation-20261027-r1/seed.jsonl /home/profsynapse/native-grpo-preparation-20261027-r1/qualification-controls.example.json
python -B examples/training_qualification/native_grpo_preparation/prepare.py inspect /home/profsynapse/native-grpo-preparation-20261027-r1/configuration.json
python -B examples/training_qualification/native_grpo_preparation/prepare.py prepare /home/profsynapse/native-grpo-preparation-20261027-r1/configuration.json
```

`inspect` is read-only: it validates the public GRPO declaration, selected prompt
column, original seed hash/size, safe private output parent and unused output
name. It does not normalize rows or promise preparation success. `prepare`
revalidates those inputs, claims the fresh root with create-only mkdir 0700,
copies the exact approved seed to `source/seed.jsonl` at 0600, then invokes the
existing preparation service with explicit source and publication roots.
The explicit 0600 step seals only these newly copied private input files; `cp`
can otherwise preserve group-writable source modes. Never apply it to shared
parents, existing runtime trees or an earlier failed preparation root.

Success retains:

- `configuration.json`: reviewed declaration, canonically serialized.
- `source/seed.jsonl`: actual approved original bytes; no prompt/metadata edits.
- `publications/dataset-<actual semantic digest>/`: existing engine manifest and
  immutable original-byte JSONL publication, independently reverified.
- `identity.json`: actual public prepared identity from the service.
- `training-input.json`: canonical public request with the **actual prepared
  ref**, replacing the declaration's logical unprepared-seed ref.
- `prepared-request.json` and `result.json`: actual public request/identity and a
  bounded host summary with the real publication path and counts.

Stdout is one JSON summary <=8 KiB; failures expose a fixed phase only, without
exception messages, tracebacks, prompts or environment values. A claimed failure
retains its configuration, source and any normalizer stage/publication plus a
best-effort create-only `failure.json`. Reusing any existing output rejects
without modifying it. Investigate retained failures; preparation does not grant
retry/replay authority for a training attempt. Root custody loss cannot be
repaired by cleanup or adoption through this example.

The normalizer preserves extra row fields in the original publication. It does
not pass them to generation or reward: those boundaries continue selecting only
the configured prompt and raw response in the existing engine. Host review owns
semantic provenance and any literal target leakage already inside a prompt.

Root alone supplies the actual captured runtime/profile/source/model bindings,
keys, fresh bounded admissions and later native-host execution. The controls
proposal is data for that review; the preparation CLI does not consume it or
declare a model/hardware qualification. Linux retained-descriptor preparation is
separate from any later reviewed private copy to the Mac.
