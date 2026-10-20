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

## Reusable preparation and review sequence

These are commands for the **later approved Mac stage**. They have not been
run against the Mac for this increment.

```sh
# From the clean pushed engine checkout, with the existing private 3.11.14 venv.
umask 077
set -C  # Every shell redirection below creates a new file or fails.
candidate_python=/Users/jrosenbaum/.local/share/synaptic-qualification/native-mlx-candidate-20261020/venv/bin/python
private_root=/Users/jrosenbaum/.local/share/synaptic-qualification/native-mlx-candidate-20261020/reviewed-attempt-1
# Create and inspect this fresh owner-only directory before writing any files.
mkdir -m 700 "$private_root"

# Prepare capture-config.json with exact stdlib, private package tree, narrow
# engine/tuner source tree, executable/framework and system-library roots.
# The capture command creates capture.json once and reports OBSERVED_NOT_ADMITTED.
"$candidate_python" -B scripts/capture_native_mlx_runtime.py \
  --config "$private_root/capture-config.json" \
  --dependency-lock Trainers/mlx_sft_mac/runtime-candidate/requirements.lock \
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
