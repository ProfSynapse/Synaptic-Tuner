# Native MLX host consumer — restricted qualification candidate

This example composes the existing public `TrainingAPI` and `RunsAPI`, reference
host record/stream stores, Foundation grants, and the native provider family.
It has no engine database, global provider registration, background training
job, replay, install, or model download. `inspect` and `profile-candidate` are
read-only. `start` is a one-shot effect and requires a separate root-signed
admission for its exact compiled workload and host selection: source, runtime,
hardware, model snapshots, prepared publication, project, request, run, attempt,
private state root, process limits and artifact budget. An ordinary two-run qualification receipt is not
created by this workflow.

## Inputs and private layout

Use an isolated **clean pushed** engine checkout on the M4 Mac. The proposed
runtime is CPython 3.11.14, MLX/MLX-LM 0.32.2, original 50-dependency lock SHA
`f9a6298260cd17f2c3d567ae3674108a71efa68e99369906487a4d8c0feb7117`,
and model/tokenizer `mlx-community/Qwen3-0.6B-4bit` revision
`73e3e38d981303bc594367cd910ea6eb48349da8`. The proposal file under
`Trainers/mlx_sft_mac/runtime-candidate/` contains the exact 3-update,
5-minute, 3-train/2-validation, LoRA and artifact bounds. It is reviewable
parameters, not execution approval.

Keep the existing private candidate at
`/Users/jrosenbaum/.local/share/synaptic-qualification/native-mlx-candidate-20261020`
intact. Place each new prepared publication, complete pinned model snapshot,
capture, reviewed profiles, admission and attempt state under a separate
owner-only 0700 directory there. The runtime, source checkout and model roots
must have no group/other write permission. The model and tokenizer snapshot
files list every regular member, size and SHA-256. They are reviewed inputs;
the consumer verifies complete current inventories and never downloads them.
Retain original private authority keys outside source control. The three
`host` secret references in the config resolve at start, not during `inspect`.
After a separately approved private snapshot acquisition, inventory its exact
bytes with the read-only `snapshot-candidate` action. The repository/revision
arguments are claims for independent review; this command does not fetch or
attest them.

The checked-in [config template](native_host.example.json) is deliberately
incomplete. Replace **every** placeholder with the actual reviewed paths and
identities. Use a fresh `state_root`, `run_id`, `request_id`, and `attempt` for
each separately approved run. Select `attempt_kind=raw_full_sequence` for the
first publication and `prompt_completion_assistant` for the second. The second
public input must explicitly set `chat_template_kwargs` to
`{"enable_thinking":false}`; the shared preprocessor supplies exact labels.
The consumer checks declared split coverage and compilation, while the root
review checks representative complete rows and token/label semantics.

Use the existing `python tuner.py prepare-dataset --config <reviewed-config>
--json` workflow to create the immutable synthetic publication. Supply its
actual `PreparedTrainingInputIdentity` and publication path; never construct
an identity from the filename. The existing dataset snapshot adapter verifies
the publication when the consumer reads it. For the message attempt, review
the actual shared preprocessor output; a `prompt_completion` label does not by
itself prove assistant-only labels.

## R5 Mac preparation and remaining boundary

The authorized preparation used a new 0700 directory at
`/Users/jrosenbaum/.local/share/synaptic-qualification/native-mlx-candidate-20261020/prep-r5-aba3c576-20261020`.
Its `source/` is a clean isolated clone of pushed commit
`aba3c57699f6617878440b3126d15e475e295651`; the user's dirty checkout
was untouched. The existing private venv is CPython 3.11.14 with the original
50-dependency lock (SHA-256 `f9a6298260cd17f2c3d567ae3674108a71efa68e99369906487a4d8c0feb7117`).
`uv pip check` passes. Retained install logs and unpacked wheel-cache metadata
are observations, not selected archive attestation: the original wheel archives
are absent. The host reports M4, 24 GiB and macOS 27.0.1 (26A434).

The checked-in static capture refused the actual Homebrew stdlib root because
`/opt/homebrew/Cellar` has mode 0775, while `NativeTree` requires every runtime
ancestor to be free of group/other write access. No capture was published. Do
not change shared Homebrew permissions or weaken the capture check to pass it.
The candidate venv also lacks `jsonschema`, imported by the current host
composition through `tuner.project`; the existing venv was not modified.
Resolve these exact closure/composition blockers and recapture on the final
pushed source before any signed execution admission.

The r7 proposal uses a single new owner-private interpreter and one 54-package
environment for host and child. The original 50 lock remains unchanged. The
four-wheel additive lock and exact combined lock are under
`Trainers/mlx_sft_mac/runtime-candidate/`; the combined SHA-256 is
`7c167018ee6bd61a3609971e3f834082c798e7bd2a3fa0c5c2564887da4fde4f`.
The checked-in `native-private-runtime.example.json` binds 54 selected wheel
URLs, filenames, sizes and hashes (146,218,037 bytes), the 20,051,753-byte
Astral interpreter archive, and a new unused private root. These are upstream
metadata selections awaiting actual archive and install readback.

From the final clean pushed source, the reusable preparation CLI provides:

```sh
# Future effects, held until separate root review. Copy the exact reviewed
# template to a NEW 0600 config file in this existing 0700 parent first.
private_config=/Users/jrosenbaum/.local/share/synaptic-qualification/native-runtime-r7-reviewed.json
python3 -B examples/training_qualification/native_runtime_preparation.py plan "$private_config"
python3 -B examples/training_qualification/native_runtime_preparation.py acquire "$private_config"
python3 -B examples/training_qualification/native_runtime_preparation.py install "$private_config"
python3 -B examples/training_qualification/native_runtime_preparation.py verify "$private_config"
```

Select the CPU host Python used for `python3` explicitly before those commands.
`plan` is local-only. `acquire` hard-bounds archive bytes before each write and
retains them. `install` uses only the exact retained archives, a hashed private
copy of uv, an offline local Python mirror and wheelhouse; both effectful phases
create consumed private claims. Every owned process has a 300-second deadline
and 4 KiB combined output cap. Temporary/cache paths remain under the private
root. Archive expansion is checked before install, but uv's temporary disk use
is not represented as a hard aggregate disk quota; the retained private tree
has a 2 GiB postphase guard. No phase signs admission.
Afterward review observed interpreter/framework/stdlib/packages, dylibs and
source trees. The final capture must include narrow `source/shared` alongside
`source/tuner` and `source/synaptic_tuner`.

Two synthetic publications were made through the existing `ingest` and
`prepare-dataset` CLIs, then copied into private Mac staging. Raw/full-sequence
publication digest: `0e4951bb2e54ed0985e6ca0d4c9940f0f5d75ccb78821c14544ef44897580cbe`.
Prompt/completion publication digest:
`d22b95afbf5e415fa6a363381fda7714cfd0d7817d3b98a5e44e3149250ee9a0`.
Each has three training and two validation rows, with declared source IDs and
split identities. Their private project is at `prep-r5-aba3c576-20261020/prepared-project/`.
Mac content hashes matched the local publications; Mac API readback is blocked
by the missing `jsonschema`. The complete rows were reviewed for their source
shape, but exact tokenizer IDs, assistant-only label masks, and
`enable_thinking=false` still require the pinned tokenizer and shared
preprocessor before training. These data hashes do not grant training authority.

The model workspace is empty. Its adjacent `model-acquisition-draft.json` is
incomplete by design: `members` and `tokenizer_members` are empty. No model
metadata request, weights download or ML import has run. The reusable
`native_model_preparation.py` action supports a later separately reviewed
acquisition using exact complete upstream metadata, a 2 MiB streaming response
cap before Hub `ModelInfo` parsing, and exact prewrite bounded HTTPS transfers
to explicitly precreated 0700 nested destinations. The SDK's unbounded
`snapshot_download` cache/transfer path is not used. An unexpected redirect
host rejects. Its observations never sign admission. The next
reviewed sequence is:

```sh
candidate_python=/Users/jrosenbaum/.local/share/synaptic-qualification/native-mlx-private-runtime-r6-20261020/venv/bin/python
stage=/Users/jrosenbaum/.local/share/synaptic-qualification/native-mlx-candidate-20261020/prep-r5-aba3c576-20261020
# After separate approval for one bounded Hub metadata request, from final clean source:
"$candidate_python" -B examples/training_qualification/native_model_preparation.py \
  metadata-candidate "$stage/model-acquisition-draft.json"
# Independently review every candidate member and complete tokenizer selection;
# copy the reviewed selection into a fresh private config and workspace. Inspect
# is local-only. Acquire is a later bounded model download, not approved here.
"$candidate_python" -B examples/training_qualification/native_model_preparation.py \
  inspect /ABSOLUTE/REVIEWED/model-acquisition.json
"$candidate_python" -B examples/training_qualification/native_model_preparation.py \
  acquire /ABSOLUTE/REVIEWED/model-acquisition.json
```

The exact acquisition schema and pins are in
[`native-model-acquisition.example.json`](native-model-acquisition.example.json).
The template paths are placeholders, never execution authority. Review the
complete metadata candidate, selected bytes, both snapshot manifests and
tokenizer loadability before wiring them into the host config.

## Reusable preparation and review sequence

The following capture/admission/start commands are for a **later reviewed
stage**. The current source and runtime closure cannot pass capture yet.

The r11 private 54-package runtime repair passed checked-in `verify`, but
static capture remained blocked by the pinned MLX wheel's nested runpath:
`mlx/core…so` declares `@loader_path/lib` and loads `libmlx.dylib`, which
requests `@rpath/libjaccl.dylib` without declaring its own rpath. The observed
file is inside the private package tree. The r12 resolver change models
declaring-loader-relative inherited paths and still requires a unique captured
target in every load context. Recheck the exact final pushed source before
running capture; the r11 unresolved selection is review material only.

The separately admitted r12 model acquisition consumed its one attempt and
stopped at a zero-byte `scratch/repository/model.safetensors` after several
small members had been verified. The retained workspace has no published
model/tokenizer snapshots or acquisition result. This locates the transfer
boundary, not its cause or retry authority. Preserve scratch and inspect the
exact transport behavior before any new independently reviewed attempt.

The r9 offline install and metadata-only `verify` succeeded with a private
Python 3.11.14 executable and 54 locked distributions. Full custody review
found only two empty uv lock files, `uv-python/.lock` and `venv/.lock`, at mode
0666. The prior install claim/result remain consumed and unchanged. After
independent review of the final pushed source and the exact retained r9
records, a single `native_runtime_preparation.py seal-uv-locks
<absolute-r7-config>` may seal those two held files to 0600 and retain a
create-only repair claim/result. A failed or interrupted repair cannot be
replayed automatically. Run checked-in `verify` and inspect exact custody
afterward. Do not run capture from the stale r5 config: it selects Homebrew,
omits `shared`, and binds the original 50-package lock. Prepare a fresh
capture config against the final pushed source and combined 54-package lock.

```sh
# From final clean pushed source with the reviewed new private 3.11.14 venv.
umask 077
set -C  # Every shell redirection below creates a new file or fails.
candidate_python=/Users/jrosenbaum/.local/share/synaptic-qualification/native-mlx-private-runtime-r6-20261020/venv/bin/python
private_root=/Users/jrosenbaum/.local/share/synaptic-qualification/native-mlx-candidate-20261020/reviewed-attempt-1
# Create and inspect this fresh owner-only directory before writing any files.
mkdir -m 700 "$private_root"

# Prepare capture-config.json with observed exact stdlib, private package tree,
# narrow source/tuner + source/synaptic_tuner + source/shared trees, and
# executable/framework/system-library roots.
# The capture command creates capture.json once and reports OBSERVED_NOT_ADMITTED.
"$candidate_python" -B scripts/capture_native_mlx_runtime.py \
  --config "$private_root/capture-config.json" \
  --dependency-lock Trainers/mlx_sft_mac/runtime-candidate/requirements.runtime.lock \
  --output "$private_root/capture.json"

# Run separately for complete, already reviewed private model and tokenizer
# roots. Keep both manifests as root-review candidates, not authority.
"$candidate_python" -B examples/training_qualification/native_host.py \
  snapshot-candidate --snapshot-root "$private_root/model-snapshot" \
  --repository mlx-community/Qwen3-0.6B-4bit \
  --revision 73e3e38d981303bc594367cd910ea6eb48349da8 \
  > "$private_root/model-snapshot.json"
"$candidate_python" -B examples/training_qualification/native_host.py \
  snapshot-candidate --snapshot-root "$private_root/tokenizer-snapshot" \
  --repository mlx-community/Qwen3-0.6B-4bit \
  --revision 73e3e38d981303bc594367cd910ea6eb48349da8 \
  > "$private_root/tokenizer-snapshot.json"

# Fill the template with exact paths, pushed ref and commit, scope, original
# private secret references, fresh attempt IDs, snapshots and publication.
# The render action only canonicalizes JSON; it grants no authority.
"$candidate_python" -B examples/training_qualification/native_host.py \
  canonicalize "$private_root/host-config-draft.json" > "$private_root/host-config.json"

# The profile candidate uses the actual pushed Git ref, Mac hardware and the
# full runtime recapture. Root reviews this output before it is selected.
"$candidate_python" -B examples/training_qualification/native_host.py \
  profile-candidate "$private_root/host-config.json" > "$private_root/profile.json"
"$candidate_python" -B examples/training_qualification/native_host.py \
  inspect "$private_root/host-config.json" > "$private_root/inspection.json"

# Root supplies the exact expiry and authority/key references. Review the
# inspection, complete source/capture/build provenance, wheel selections,
# tokenizer rows and admission content independently before signing.
"$candidate_python" -B examples/training_qualification/native_host.py \
  admission-candidate "$private_root/host-config.json" \
  --expires-at REVIEWED_EXPIRY_WITHIN_ONE_HOUR_UTC \
  --authority-ref ROOT_REVIEW --key-ref REVIEWED_KEY \
  > "$private_root/admission-content.json"

# Root signing occurs outside the consumer, with a private key that is never
# placed on the execution Mac. The reviewer returns admission.sig and a pinned
# root public key. The host only verifies their exact content.
"$candidate_python" -B examples/training_qualification/native_host.py \
  seal-admission "$private_root/admission-content.json" \
  --signature "$private_root/admission.sig" > "$private_root/admission.json"

# Only after root's bounded live authorization: one start, no retry on error.
"$candidate_python" -B examples/training_qualification/native_host.py \
  start "$private_root/host-config.json" \
  --trusted-root-key-sha256 REVIEWED_ROOT_PUBLIC_KEY_SHA256

# New process, same known run and preserved state. These are public API reads;
# logs, cancel and reconcile are intentionally unavailable.
"$candidate_python" -B examples/training_qualification/native_host.py \
  observe "$private_root/host-config.json" \
  --trusted-root-key-sha256 REVIEWED_ROOT_PUBLIC_KEY_SHA256
"$candidate_python" -B examples/training_qualification/native_host.py \
  verify "$private_root/host-config.json" \
  --trusted-root-key-sha256 REVIEWED_ROOT_PUBLIC_KEY_SHA256
"$candidate_python" -B examples/training_qualification/native_host.py \
  download "$private_root/host-config.json" \
  --trusted-root-key-sha256 REVIEWED_ROOT_PUBLIC_KEY_SHA256 \
  --destination "$private_root/downloaded-artifacts"
```

The `admission-content.json` file is the exact canonical message to sign with
`openssl dgst -sha256 -sign <offline-root-private-key> -out admission.sig`.
`seal-admission` merely records the signature hash; `start` verifies the
signature against the separately passed public-key fingerprint. A hash in
the config or inspection is not independent approval. Root must review the
source commit, selected wheels/install provenance, capture/build closure,
hardware, model snapshot and both distinct prepared datasets before signing.

An admitted restricted run can only use its exact signed selection. Its
one-hour admission is a start deadline. Cold show, observe, verify and download
recompose the same signed selection and retained state after expiry, without
renewing dispatch authority; they do not require the remote branch to remain
at its former head. A new start or replay still requires fresh reviewed admission.
Host metadata commands use an owned subprocess with a combined output byte
bound and deadline; exceeding either ends its process group.
The normal capability receipt still needs two real accepted runs, native
artifact verification, Darwin deadline/owner-loss evidence, and a real
Syntunia bridge readback. Public logs/cancel/reconcile, GRPO, and asynchronous
desktop operation remain unsupported.
