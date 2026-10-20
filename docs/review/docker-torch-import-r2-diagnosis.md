# Docker r2 Torch import boundary — CPU diagnosis, 2026-10-20

## Retained failure

The one authorized r2 public start on engine `0d515ae1ae235d0858802228ce4af8dba8fccd03`
and host `85ea6b56af89f6dcb7c0a2c7644e95c511762486` was accepted at
13:47:50Z and failed at 13:47:52Z. Container
`5256d2446d3751e423a6ff535c3ed9a13a2bc5df92765a61db5da700bf7e601d`
remains retained with exit 23. Its 81-byte trainer stderr, SHA-256
`95d34d74eae9cb55a223eeda8428ee4387ee79095e5c0f39a0ceb60da9f45513`,
contains only `OFFLINE_SFT_WORKER_REJECTED phase=TRAINER category=IMPORT
milestone=TORCH_IMPORT`. This locates the first Torch import boundary in
`Trainers/sft/train_sft.py`; it does not identify an import dependency or a
remedy. The earlier bare marker on container
`07ecde6d5b798abf51eb85595be165039cdc1043295cbd0a5b0d3d526064bfd7` remains historical
and uninterpreted.

The release file SHA-256 is
`5f65824ceb5e250a58314da0233b9c07f1d63f1b0a39892d9b36af694938bfe6`;
its immutable image ID is
`sha256:e737a0a091474c4892690690e712bc1f932f108856a3d5df60383e1cd9436e1e`.
The retained final runtime capture records CPython 3.12.3 and Torch
`2.11.0+cu128`. Source, image, release and installed wheel were not changed.

## Targeted CPU experiments

The checked-in `scripts/diagnose_docker_torch_import.py` binds that release
file hash, daemon ID and exact local image ID. It starts a fresh retained
container with at most 2 CPUs, 4 GiB and 120 seconds; network and GPU devices
are disabled. It has no credentials, model mount or training entrypoint, and
emits only closed categories and identity fields.

The bare image probe returned `PASS` in container
`33af62c4fdda5d855b3b154c2937912b5266a88a48f2c0757b3664fe42a45917`.
The final guarded probe mounted the r2 staged source read-only, validated the
offline worker closure, installed its owned-module guard, ran
`init_trainer_env()`, and imported Torch successfully. Its retained container
is `2e5b6d06a4cc946541a475da72b7aa7e851b5840f86fd675c2099540b6fc1045`;
`evidence-guarded-final2/result.json` has SHA-256
`326b59ebf3896a737732661895ef67b3efcf2527d65ce912b02ff20d9437ee33`.
The exact final checked-in helper source has SHA-256
`c214fc8b9d617fe7171e7eac987d93187169b8a637fac9ef24b4323a70bf6c53`.
These artifacts are under the private Windows WSL root
`/home/profsynapse/.local/share/synaptic-qualification/torch-import-r4-20261020/`.

Earlier guarded diagnostic containers returned `READ_MANIFEST/OS` because
the probe dropped all capabilities while running as root against mode-0600
staged files owned by the WSL user. Running the diagnostic as that owner
resolved this probe setup failure. Those containers and reports are retained;
their failures do not reinterpret r2.

## Interpretation and next boundary

Torch is importable in the exact image both bare and after the retained worker
closure, guard and trainer bootstrap under the finite CPU diagnostic. The r2
container requested GPU 0 and used root, default capabilities and a writable
root filesystem; the CPU probes deliberately did not reproduce that full
runtime. The cause of r2's Torch import failure remains unresolved. In
particular, the evidence does not support changing packages, image, recipe,
model, GPU settings or SFT runtime code yet.

The r2 claim, container and all older failure evidence remain consumed and
preserved. No training start, image build, pull, model load, package install or
HF action occurred during this diagnosis. A further comparison of the
original GPU/runtime environment needs a separately reviewed exact scope;
this diagnostic does not authorize another training attempt. No runtime
closure or Modal lock refresh is needed for the diagnostic script and skill
changes, because no locked runtime member changed.
