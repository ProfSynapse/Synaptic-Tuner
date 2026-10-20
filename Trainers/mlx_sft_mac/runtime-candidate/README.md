# Candidate native MLX dependency closure

This directory prepares an isolated Apple runtime for later qualification. It
does not change the legacy trainer, admit a public API runtime, select a model,
or establish SFT/GRPO support. Never install it into the shared host interpreter.

The starting candidate is CPython **3.11.14**, macOS **15 or newer**, **arm64**,
MLX **0.32.2** and MLX-LM **0.32.2**. The interpreter version matches the observed
qualification Mac; its executable and native libraries still need independent
capture. macOS 15 is a dependency-resolution target, not a tested minimum OS.

The reviewed MLX-LM source is tag `v0.32.2`, commit
`93e6d4a6447b3080861985947f0820e76fcb957d`. Its
[dependency declaration](https://github.com/ml-explore/mlx-lm/blob/93e6d4a6447b3080861985947f0820e76fcb957d/pyproject.toml)
requires Python 3.11 or newer and Transformers 5.7 or newer. Keep this closure
separate from the existing CUDA/Modal training environments.

## Reproduce the candidate

From the repository root with **uv 0.12.22**, resolve wheel-compatible packages
without installing or building code:

```sh
MACOSX_DEPLOYMENT_TARGET=15.0 uv --no-config pip compile \
  Trainers/mlx_sft_mac/runtime-candidate/requirements.in \
  --python-version 3.11.14 --python-platform aarch64-apple-darwin \
  --default-index https://pypi.org/simple --only-binary :all: \
  --generate-hashes \
  --output-file Trainers/mlx_sft_mac/runtime-candidate/requirements.lock
```

Review every changed version/hash. Existing output pins constrain re-resolution;
an intentional dependency upgrade needs separate review. The lock records
accepted package archive hashes, not the selected installed wheel inventory,
Metal availability, or package provenance beyond the selected index.
uv's generated header omits the deployment-target environment value and default
index; the command above records both. Run it without ambient index overrides.

Later installation must use a fresh private virtual environment on the exact
target interpreter, the complete lock with `--require-hashes --only-binary
:all:`, and an explicit public index. Retain the selected wheel filenames and
hashes, installed inventory, interpreter identity and native library inventory.
Do not use a successful installation as runtime or GPU qualification.

The existing uv CLI provides the preparation commands; no custom installer is
needed. On the target Mac, first verify the selected interpreter reports exactly
3.11.14 and the candidate parent is owner-controlled. For a previously unused
absolute directory:

```sh
(
set -eu
candidate_root=/absolute/unused/private/native-mlx-candidate
candidate_python=/absolute/path/to/python3.11
umask 077
mkdir "$candidate_root"
uv --no-config venv --python "$candidate_python" --no-python-downloads \
  "$candidate_root/venv"
UV_CACHE_DIR="$candidate_root/uv-cache" uv --no-config pip sync \
  --python "$candidate_root/venv/bin/python" --require-hashes \
  --only-binary :all: --default-index https://pypi.org/simple \
  Trainers/mlx_sft_mac/runtime-candidate/requirements.lock
uv --no-config pip check --python "$candidate_root/venv/bin/python"
uv --no-config pip freeze --python "$candidate_root/venv/bin/python"
)
```

Stop on any failed command. Do not rerun setup into an existing environment
without diagnosing the retained outcome. Invoke the tools without credential
variables or ambient uv configuration. `pip check` and `pip freeze` examine
installation metadata; they do not load MLX or test the GPU.

## Candidate installation evidence — 2026-10-20

The M4/24 GiB qualification Mac accepted all 50 packages from this lock in a
fresh private CPython 3.11.14 virtual environment. Installation used uv 0.12.23,
an empty inherited environment except executable search path and private uv
cache, an explicit public PyPI index, required hashes and wheels only.
`uv pip check` passed, and the complete frozen inventory matched the 50 lock
versions. The shared interpreter and existing Mac checkout were unchanged.

Lock SHA-256:
`f9a6298260cd17f2c3d567ae3674108a71efa68e99369906487a4d8c0feb7117`.
Frozen inventory SHA-256:
`ed46e2fdf76e99040351501a4f0a05a04ae1589a45300641b2414afe88f4f877`.
This is dependency-installation evidence only; no ML framework was imported
and no Metal probe, model load or training step was executed.

## Prepared worker boundary

The pinned
[dataset code](https://github.com/ml-explore/mlx-lm/blob/93e6d4a6447b3080861985947f0820e76fcb957d/mlx_lm/tuner/datasets.py)
adds EOS for plain text. Its prompt/completion loader constructs a user and
assistant conversation, and chat masking calculates one prompt offset. Those
operations are not automatically equivalent to the engine's admitted prepared
rows, chat-template options, or per-token target masks.

The pinned
[trainer](https://github.com/ml-explore/mlx-lm/blob/93e6d4a6447b3080861985947f0820e76fcb957d/mlx_lm/tuner/trainer.py)
accepts custom loss and batch iteration functions. The internal
`tuner/runtime/native_mlx_sft.py` candidate now implements those hooks over
shared `PreparedSFTExample` values, preserving split/row identities, arbitrary
ignored-label spans and partial batches without using the upstream loaders.
CPU contracts check causal shifting, safe ignored targets, token-weighted loss,
and complete validation traversal. Real tokenizer comparisons, MLX autograd and
parameter updates are still required.

The associated `native_mlx_runtime.py` comparison accepts an exact host-supplied
inventory, interpreter hash and original dependency-lock hash. It rejects
metadata/interpreter drift before worker imports and returns only
`CANDIDATE_MATCH_ONLY`. It is neither an inventory issuer nor a native runtime
material. The complete expectation must be reviewed against this 50-package
lock; the two-package CPU fixtures are only unit-test inputs.

Read-only source inspection on the installed Mac candidate found these hashes:

- `mlx_lm/tuner/trainer.py`:
  `0b6027fdb11803e951ceee52c83a81f82daa922ea8c6b7ca6888953461064e8e`
- `mlx_lm/tuner/datasets.py`:
  `dcd675376d68d1813261c8389236b586b16b9a58a5ea80be8eb535bb9d0e9689`
- Candidate Python executable:
  `b41507ef245eac19f76b6ed5a53fbc5eb49bf7b83b95b78414a50fccfe0bf0e7`
- Linked Homebrew Python framework:
  `c5042528a85834df4ead67663e949dc938f7354f98c0c7b70c0f4a17830234e8`

`otool -L` confirms that executable loads a separate Homebrew Python framework
and the system library. Its executable hash alone is not a native-library
closure. These observations involved no ML imports or GPU activity.

See the [native desktop plan](../../../docs/plans/native-desktop-training-api-v1.md)
for runtime material, provider lifecycle and artifact acceptance requirements.
