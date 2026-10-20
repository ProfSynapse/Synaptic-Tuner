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
