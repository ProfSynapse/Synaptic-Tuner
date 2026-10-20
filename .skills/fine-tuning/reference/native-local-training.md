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
