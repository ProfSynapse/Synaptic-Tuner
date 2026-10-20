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
gradient accumulation and checkpointing are deliberately unavailable in this
increment. Upstream accumulation averages microbatch means, which needs a
separate objective review for unequal supervised-token counts. Model, LoRA,
optimizer and learning-rate policy must be prepared explicitly by the future
child composition; this worker does not infer or silently apply those controls.

`tuner/runtime/native_mlx_runtime.py` compares host-supplied exact interpreter
and complete distribution pins against the running candidate before worker ML
imports. `CANDIDATE_MATCH_ONLY` means metadata and executable bytes match.
It does not authenticate wheel files, imported module origins, Homebrew's linked
Python framework, other native libraries, Metal, or launch authority. Retain the
original hash lock and use the existing `uv pip freeze`, `uv pip check`,
`shasum -a 256` and `otool -L` tools for review evidence; do not install packages
into the shared interpreter or treat a newly observed inventory as approved.

There is no native launch CLI yet. The internal worker may run only in an
independently admitted, host-owned bounded child. The desktop plan lists the
missing native runtime material, child lifecycle, artifact roles and API joins.
Do not call the worker as an ad hoc substitute for that composition.

## Execution-boundary continuation

The internal runtime modules now include a read-only native material capture,
retained-source/tokenizer join, explicit LoRA/constant-AdamW composition,
independent safetensors delta verifier, and owned subprocess guardian. These are
reviewable internal boundaries, not a registered provider or native launch CLI.

`scripts/capture_native_mlx_runtime.py` takes a JSON object with `capture` and
`expectation`. `expectation` uses the existing `MLXRuntimeExpectation` fields.
`capture` contains absolute `stdlib`/`packages` roots, sorted `sources` trees,
sorted explicit external `files`, sorted `system_libraries`, and sorted
`aliases` pairs of absolute alias path and exact symlink spelling. Every tree
file, including startup `.pth` and bytecode, is captured. Unlisted aliases reject.
Homebrew's framework aliases must resolve to explicit captured bytes. Its
stdlib `site-packages` alias is recorded but excluded: the future isolated child
must use `-I -S` and explicitly bind only its candidate package search path.

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
parameter delta. Final validation is a distinct pass **after reloading the saved
adapter**; upstream in-loop validation occurs before the last optimizer update.
It saves native `adapter_config.json` and `native_result.json` exclusively.
These are MLX artifacts, not PEFT or authenticated public artifact roles.

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

Before ML imports/training, complete and review the isolated child bootstrap,
authenticated compiled workload/native material, public provider-family
composition and native artifact roles. The bounded next proposal is
`Trainers/mlx_sft_mac/runtime-candidate/sft-qualification-proposal.json`; its
status is review-required and it is not executable admission.
