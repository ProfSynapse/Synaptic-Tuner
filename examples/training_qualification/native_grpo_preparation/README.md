# Native GRPO input preparation and tokenizer-only diagnosis

The Linux preparation actions wrap the existing `TrainingInputPreparationServiceV1`,
`GRPOTrainingInputNormalizerV1` and `GRPOPreparedInputFormatVerifierV1`. It needs
only the engine's CPU host dependencies. It selects no runtime/provider,
downloads nothing, imports no ML library and never constructs an admission.
The existing `prepare-dataset` CLI accepts SFT dataset-prep schemas; the `train`
GRPO preparation join requires a Modal profile. Neither is a native-only input
preparation command. This example supplies that missing host composition.

The separate `profile-tokenizer` action supports Linux/Darwin host diagnosis
using an already verified publication, reviewed compiled GRPO workload and
private tokenizer snapshot. It imports the declared Transformers package for
local tokenizer use only; it imports no MLX and loads no model. The ordinary SFT
profiler does not execute GRPO's exact shared prompt-tokenization boundary.
Preparation's existing Linux-only behavior and configuration remain unchanged.

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

## Tokenizer-only GRPO profile

Root must review the final pushed source, actual workload, copied publication,
snapshot inventory and existing environment before running this action on the
Mac. This command supplies no source/runtime/admission authority and performs no
model/rollout/training operation. Do not install packages, contact the Mac, or
replay an earlier training attempt to obtain this evidence.

Copy `tokenizer-profile.example.json` to a fresh private host configuration file
at0600. Replace its placeholder paths with the **actual** reviewed canonical
`workload.json`, immutable `dataset-<semantic-digest>` publication and existing
tokenizer snapshot JSON (`repository`, `revision`, `root`, complete `files`
inventory). Paths may be absolute or relative to the configuration. The example
declares Transformers5.20.2 from the selected GRPO lock; the command checks the
installed metadata version exactly without resolving or changing dependencies.
Review its explicit tokenizer/evidence byte bounds and fresh output root.
The snapshot source/revision must equal the compiled workload's tokenizer pins;
no model-specific mapping or substitute profile is constructed here.

This read-only action retains the publication through `NativeTree` on both
Linux and Darwin, then uses the existing pure publication, identity and input
verifiers. The preparation service's Linux-only filesystem reader stays on the
preparation path. Publication members must be single-link owner files, sealed
0400 originals or separate byte-identical 0600 native copies under a 0700 root.
Exact member identities and bytes are rechecked after tokenizer use; replacement
by an identical file still rejects.

After root approval, using the already selected interpreter:

```bash
python -B examples/training_qualification/native_grpo_preparation/prepare.py profile-tokenizer /absolute/private/tokenizer-profile.json
```

The action recompiles the exact public workload and verifies its GRPO prepared
publication with the existing format verifier. Only selected prompt messages
reach the worker's shared `tokenize_prompts` helper. It loads the exact verified
private snapshot with `local_files_only=True`, `trust_remote_code=False` and the
compiled `use_fast` setting. It verifies source/snapshot custody again afterward.
The helper explicitly requests `tokenize=True`, `return_dict=False` and
`add_generation_prompt=True`, forwarding the same configured template kwargs
without changing the scaffold. Public validation reserves those renderer
controls, including `return_dict`; conflicting kwargs reject before tokenizer
calls. IDs must be a nonempty list of nonnegative exact integers within each
prompt budget, with a valid integer EOS; no truncation, key iteration or repair.

A create-only0700 output contains0600 `configuration.json`, bounded private
`token-evidence.json` (actual IDs, EOS, source/workload/snapshot bindings and
tokenization controls), and `result.json`. Stdout is a summary<=8KiB with token
counts and the evidence hash, without prompts/IDs; tokenizer Python output is
suppressed. Failed claimed stages retain a closed `failure.json` and partial
bytes. Existing output roots reject without adoption, overwrite or auto retry.

This can prove the ordinary host's exact GRPO prompt-tokenization boundary. It
does not prove isolated child import admission, MLX, model/adapter loading,
generation, reward variation or training. The pinned API mapping-default defect
is statically reproducible, but historical worker exit122 still has no retained
phase/cause evidence and must not be relabeled or replayed. Fresh final-source
review/capture and separate root admission remain necessary for any later model
attempt.
