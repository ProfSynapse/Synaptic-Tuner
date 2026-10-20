# Native local training preparation

Run `python tuner.py doctor native-training --json` on the target machine to
record a bounded, versioned `DIAGNOSTIC_ONLY` report. The command observes OS,
architecture, macOS version when applicable, CPython implementation/version,
and installed distribution versions for `mlx`, `mlx-lm`, `numpy`, `transformers`,
and `torch`. `absent`, `malformed`, and `observation_failed` remain distinct.

`native_apple_candidate` is true only for observed Darwin `arm64`. The report
does not establish Metal access, GPU capacity, package provenance, runtime
compatibility, or SFT/GRPO training support. It does not import the ML packages,
load a model, access credentials, initialize a GPU, install packages, or make
network calls. `--fix` and `--env-file` are rejected. Existing legacy MLX SFT
is separate from an API v1 native Mac adapter and training qualification.

Run the same command on Windows and inside WSL separately: each report describes
the OS running that Python process. Windows AMD64/ARM64 and Linux/WSL metadata
do not establish NVIDIA hardware, CUDA, WSL GPU passthrough or Docker readiness.
The Windows desktop qualification path remains WSL2 plus the pinned NVIDIA
Docker runtime. Installed-base CI exercises this diagnostic on Windows, macOS
and Linux; those checks remain separate from real GPU training evidence.

For the next Apple dependency-preparation step, use the checked-in
`Trainers/mlx_sft_mac/runtime-candidate/README.md` and its hash-pinned lock.
Resolve with the documented platform/interpreter/index and install only in a
fresh private environment with hashes required and source builds disabled.
This candidate does not select a model, enable a provider, or qualify Metal,
SFT or GRPO. Keep CUDA/Modal environments and the legacy Mac checkout unchanged.

## Prepared SFT worker candidate

`tuner/runtime/native_mlx_sft.py` is an internal CPU-tested boundary. It consumes
the existing shared `PreparedSFTExample` objects after host verification and
tokenization; it does not introduce a dataset format or public execution API.
Pass the exact prepared-input identity, tokenizer revision, retained row IDs,
declared train/validation tuples and explicit `MLXSFTConfig` controls. The
projection copies all tokens/labels immutably and records a deterministic digest.
The host must still bind those rows to the authenticated source and tokenizer.

No MLX text/chat loader, extra EOS, resplit, truncation, packing, tail dropping,
or single-offset masking is allowed. Failed masking diagnostics, already
truncated rows, attention masks other than all-ones unpadded causal rows, and
rows without shifted target supervision reject. Arbitrary ignored token spans
remain ignored. Training shuffles deterministically within its own split;
validation retains every row in declared order, including partial batches.

The custom hooks target pinned MLX-LM 0.32.2. One batch is one optimizer update;
gradient accumulation and activation checkpointing are deliberately unavailable in this
increment. Upstream accumulation averages microbatch means, which needs a
separate objective review for unequal supervised-token counts. Model, LoRA,
optimizer and learning-rate policy are prepared explicitly by the isolated
child composition; this worker does not infer those controls.

`tuner/runtime/native_mlx_runtime.py` compares host-supplied exact interpreter
and complete distribution pins against the running candidate before worker ML
imports. `CANDIDATE_MATCH_ONLY` means metadata and executable bytes match.
It does not authenticate wheel files, imported module origins, Homebrew's linked
Python framework, other native libraries, Metal, or launch authority. Retain the
original hash lock and use the existing `uv pip freeze`, `uv pip check`,
`shasum -a 256` and `otool -L` tools for review evidence; do not install packages
into the shared interpreter or treat a newly observed inventory as approved.

The internal worker may run only in an independently admitted, host-owned
bounded child. Use the checked-in restricted host consumer described below;
do not call the worker directly as an ad hoc substitute.

## Execution-boundary continuation

The internal runtime modules now include a read-only native material capture,
retained-source/tokenizer join, explicit LoRA/constant-AdamW composition,
independent safetensors delta verifier, and owned subprocess guardian. These are
reviewable runtime boundaries. The gated family described below composes them
through the existing public coordinator; no global provider registration is
supplied.

`scripts/capture_native_mlx_runtime.py` takes a JSON object with `capture` and
`expectation`. `expectation` uses the existing `MLXRuntimeExpectation` fields.
`capture` contains absolute `stdlib`/`packages` roots, sorted `sources` trees,
sorted explicit external `files`, sorted `system_libraries`, and sorted
`aliases` pairs of absolute alias path and exact symlink spelling. Every tree
file, including startup `.pth` and bytecode, is captured. Unlisted aliases reject.
Homebrew's framework aliases must resolve to explicit captured bytes. Its
stdlib `site-packages` alias is recorded but excluded: the isolated child uses
`-I -S -B` and explicitly binds only its captured package search path.

Run the selected interpreter with `-B`, from an independently reviewed source
checkout, to avoid creating bytecode while capturing. Output requires an
existing owner-private parent and is exclusive; no environment repair or
installation occurs. `--check <capture> --expected-digest <reviewed-sha256>`
revalidates the exact original configuration and capture. Unknown/ambiguous
Mach-O dependencies, uncaptured external files and extra declared OS libraries
reject. OS shared-cache dependencies are explicitly trusted through the recorded
OS/kernel build, not falsely described as hashed ordinary files.

This captures installed bytes and static native dependencies. It does **not**
prove selected wheel archive provenance, attest dynamic `dlopen` behavior,
freeze the filesystem against a malicious same-uid process, or independently
grant execution. Root review must reconcile the original hash-required install
evidence and selected archives; public capability remains gated.

`native_mlx_input.py` consumes the existing retained stream lease, verifies
identity/hash/size, and feeds the established JSONL publication rows to shared
`materialize_sft_example`. It retains row IDs and splits without filtering or
repair. The tokenizer loads only from a verified private local snapshot with
`trust_remote_code=False` and `local_files_only=True`. Model and tokenizer
snapshots, exact source, rendering arguments and worker controls are committed
in the bound input. The host still owns publication semantic/lineage validation
and binding to an actual compiled workload.

`native_mlx_composition.py` loads a configured local MLX snapshot, selects every
LoRA target/layer explicitly, checks the realized trainable names/shapes and uses
explicit AdamW parameters with constant schedule. Unsupported schedules reject.
It checks the optimizer's actual step count. The native adapter verifier checks
tensor headers, bounds, names/shapes/dtypes, finite values and a nonzero numeric
parameter delta. Final validation loads a second clean pinned base, attaches the
same configured LoRA, loads the saved final adapter and compares every trainable
tensor against the saved bytes before evaluating that second model. A no-op or
partial reload rejects. Upstream in-loop validation occurs before the last
optimizer update.
It saves native `adapter_config.json`, `input_binding.json` and
`native_result.json` exclusively. These worker files become authenticated public
artifact roles only through the family verifier described below.

`native_process.py` records claim/spawn/outcome through the existing
`DurableRecordStorePort` in the effects partition. There is no engine database.
The stdlib guardian owns the worker group, drains bounded output without
exposing its contents, enforces time/file bounds, and terminates on owner-pipe
loss or cancellation. Darwin uses kqueue process notifications because the
qualification CPython does not expose `os.waitid`. Linux tests use waitid without
reaping. Persisted PIDs are diagnostic only; another process cannot acquire
signal authority by reading them. Interrupted submission/publication retains
uncertainty and never auto-replays. This is trusted-worker supervision, not an
OS sandbox; escaped hostile descendants and guardian failure remain outside the
current evidence. Real Darwin lifecycle tests remain required.

Before ML imports/training, independently review the isolated child bootstrap,
authenticated compiled workload/native material, public provider-family
composition and native artifact roles. The bounded next proposal is
`Trainers/mlx_sft_mac/runtime-candidate/sft-qualification-proposal.json`; its
status is review-required and it is not executable admission.

## Public coordinator continuation

`native_mlx_compilation.py` recompiles `TrainingInputV1` plus the retained
`PreparedTrainingInputIdentity` and explicit `NativeMLXProfile` into a normal
`CompiledWorkload`. Parsing re-derives the complete document. Model/tokenizer
pins, loss semantics, chat arguments, batch/sequence/update/seed/checkpoint
controls, LoRA and constant AdamW are explicit. Accumulation, epochs, DoRA,
RSLoRA, implicit splits, packing and unsupported checkpoint-pruning policies
reject. Profiles declare the alpha/rank scale conversion; no model switch is
embedded in a launcher.

`native_mlx_child.py` is an internal `-I -S -B` entrypoint. It restricts import
roots, checks imported source/extension origins against capture members and
compiles pinned Python source rather than ambient bytecode. It joins the exact
staged dataset, capture, lock and snapshots to the recompiled workload. Before
optimizer work, it builds the exact complete tokenizer role archive, extracts
those bytes into a private temporary directory and loads the tokenizer locally
with remote code disabled. The host later publishes that same deterministic
archive. It is
not a public invocation or an independent source/qualification authority.
`native_runtime_spec` represents the capture with the existing `RuntimeSpec`
`native_environment` discriminator; it invents no OCI image. Legacy OCI
canonical bytes are unchanged.

`native_mlx_planning.py` supplies immutable planning/preparation adapters.
`native_mlx_family.py` supplies the existing `ProviderFamilyV1`: stage, bounded
synchronous submit, authenticated known-run observation and artifact streaming.
Use it with `compose_reference_host`; Syntunia's existing loader/resolver,
grant, record, stream, secret and clock ports remain authoritative. The consumer
must authenticate actual source evidence and current hardware before signing
command material. `NativeCommandCatalogV1` retains create-only signed material
through the existing record port. Signed terminal evidence must also match the
retained process outcome on every read. It is not an engine database or replay grant.
Immediately before a submit process claim, the family recaptures the admitted
runtime/source closure and verifies complete model/tokenizer inventories. Drift
at this boundary rejects before `run_once`; later dispatch remains uncertain.
The CPU artifact verifier requires every configured target in the exact final
model layer set and complete tokenizer snapshot selection.

No accepting **ordinary** hardware authority is shipped. An unadmitted
candidate has observe/artifact capabilities disabled and preflight is not
ready. A separately verified `NativeRestrictedAdmissionV1` enables only its
exact compiled workload and host selection, including project, request, run,
attempt, private state root, process limits, artifact budget, source, runtime,
prepared bytes and model/tokenizer snapshots, subject to a finite start expiry;
it is not qualification. Retained public reads authenticate that signed selection
after expiry and remote branch advancement without new dispatch authority. The host
consumer in `examples/training_qualification/native_host.py` verifies the
external signature using a separately supplied reviewed public-key SHA-256.
It has no signing key or acceptance switch. The same host checks the actual
clean pushed Git ref, full runtime recapture, private prepared publication,
model snapshot inventories and M4 hardware before issuing command material.
The proposal remains parameters, not an admission. See
`examples/training_qualification/native-host.md` for exact config fields,
preparation/readback commands, and the root signing boundary. No Mac training,
ML import, model download or host qualification has been performed for this
consumer yet.

`NativeQualificationV1` contains exact selected profile/source/runtime/hardware
and two distinct retained attempt-evidence references; its independent host
authority must authenticate the supporting real runs, selected wheel/install
provenance and final-adapter/public-readback evidence. Reference hashes alone
are not qualification. Public logs, cancellation and reconciliation remain
disabled even with this narrow qualification. The guardian's internal stop
path does not promote public cancellation.

`native_mlx_publication.py` independently checks the exact output inventory,
workload/input binding, optimizer count, finite saved tensor delta, retained
checkpoint tensors and final-adapter validation identity. It emits the five
existing roles. `final_model` is a deterministic POSIX tar containing the MLX
adapter/config/checkpoints and pinned base manifest; `tokenizer` is an explicit
configured member archive from the verified snapshot. Lineage/metrics retain
the exact worker bindings. Finite file/aggregate budgets and public 1 MiB
stream chunks apply. CPU verification does not rerun the final evaluation.

See `docs/plans/native-desktop-training-api-v1.md` for remaining review and
real-Mac acceptance. The CPU facade tests substitute the ML child;
they are not a hardware result and must never unlock the candidate.
