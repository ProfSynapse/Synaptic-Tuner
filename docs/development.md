# Developer guide

## Choose an integration boundary

Use `synaptic_tuner.api.v1` for application integration. Its
[compatibility policy](architecture/api-v1-compatibility.md) defines the public
contract; the [support matrix](plans/api-v1-support-matrix.md) defines which
compositions and operations have qualification evidence. This is an embedded
Python library. Your application owns configuration, credentials, authorization,
durable stores, UI and transport.

Start with the [reference storage composition](architecture/reference-host-storage.md),
[Modal consumer](../examples/training_qualification/modal-consumer.md), or
[native MLX host contract](architecture/native-mlx-host.md). The Mac path now has
real bounded SFT/GRPO and progress/cancel/caller-loss evidence; it still requires
independent host qualification or a reviewed restricted admission. Installing
the package or matching a report hash never grants execution authority.

## Install a release or work from source

The first publication is [1.1.0rc1](releases/1.1.0rc1.md), a GitHub engine preview.
Download and verify its wheel using the attached `SHA256SUMS`, then install it
in a new Python 3.10+ virtual environment. There is no PyPI release in this
workflow. Base imports/assets/help are tested on Ubuntu Python 3.10–3.14 and
Windows/macOS Python 3.12; that matrix does not qualify GPU execution.

For examples, scripts and runtime declarations, obtain the matching source:

```sh
git clone --branch v1.1.0rc1 --depth 1 https://github.com/ProfSynapse/Synaptic-Tuner.git
cd Synaptic-Tuner
python -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/synaptic --help
```

Use the tag for reproducing a release and a new branch from current `main` for
development. The built sdist provides library build inputs; use the Git checkout
for host examples, scripts, skills and runtime declarations.
An installed wheel alone does not provide a captured training runtime or the
full source-tree CLI workflow. Optional Modal/ML dependencies belong in the
separately documented pinned environments; do not add them to a base consumer
just to make imports pass. On Windows use `.venv\Scripts\python.exe` and the
corresponding console executable.

Native MLX executes on supported Apple silicon; NVIDIA Docker training requires
a usable NVIDIA GPU and the pinned CUDA runtime. A Linux CPU installation is
enough for provider-free contract checks, not GPU training or inference.

Host-project manifests must explicitly opt into this prerelease, for example
`engine: {requires: '>=1.1.0rc1,<2', api: v1}`. A stable-only range such as
`>=1,<2` intentionally rejects prerelease engines; do not bypass that check.

## Validate changes

Read `AGENTS.md` and the relevant canonical `.skills/` guide before changing a
workflow. Skills are guidance, not authorization for provider effects. Run
tests appropriate to the change; GitHub CI runs the broader CPU/contract and
installed-platform suites. A small base environment does not include every
optional dependency needed by the complete repository test suite.

```sh
.venv/bin/python -m pytest tests/contract/test_runtime_assets.py -q
python3 .skills/scripts/sync_skill_trees.py --check
git diff --check
```

When canonical skills change, run the sync script without `--check`, then check
again. Do not edit `.agents/skills` or `.claude/skills` independently.
Hashed source changes require the existing fixed-inventory maintainers:

```sh
python scripts/regenerate_offline_sft_worker_closure.py
python scripts/regenerate_packaged_training_worker_closure.py
python scripts/regenerate_grpo_worker_closures.py --kind offline
python scripts/regenerate_grpo_worker_closures.py --kind outer
python scripts/regenerate_modal_runtime_lock.py
python scripts/regenerate_modal_inference_lock.py
python scripts/regenerate_hf_parent_closure.py
```

These are read-only checks by default. After reviewing the actual source delta,
use their explicit `--write` mode in dependency order to refresh hashes, then
rerun the checks. Do not widen fixed inventories or change dependency pins as a
hash repair. `CURRENT` means source/lock agreement, not hardware qualification.

## Cutting a release

1. Choose the channel and scope. Use a prerelease while the broad stable
   workflow gates remain open; do not relabel bounded evidence as universal
   support. Update `_version.py`, classifier, changelog, release notes and the
   current support docs. Refresh only affected source hashes and synced skills.
   Update positive host/CI fixture manifests to explicitly admit a prerelease;
   keep stable-only rejection tests intact. Diagnostic derivative fixtures can
   also pin hashes of a version-bearing source closure: reproduce the old pins
   first, then refresh their exact expected hashes without weakening admission
   or relabeling historical live evidence.
2. Run the applicable contract/installed checks and independent release review.
   Open a release PR and require all current checks on its exact head before
   merging. Preserve historical execution identities when only metadata/docs
   change; do not retrain to obtain a new package version.
3. Build from the clean, merged source in a dedicated Python environment with
   `build>=1,<2`. Use a fresh absolute output directory:

   ```sh
   python -m build --sdist --outdir /absolute/release/dist
   python -m pip wheel --no-deps --wheel-dir /absolute/release/dist \
     /absolute/release/dist/synaptic_tuner-1.1.0rc1.tar.gz
   ```

4. Validate metadata/resources and install that wheel in a neutral base-only
   environment. Reuse `tests/contract/test_installed_base_compatibility.py` and
   the resource expectations/build pattern in
   `.github/workflows/base-python-compatibility.yml`. Run asset checks with
   `SYNAPTIC_WHEEL_UNDER_TEST` pointing to the built wheel. This release does
   not claim self-containment; keep that separate publication gate closed.
5. Hash the exact two archives with SHA-256 into `SHA256SUMS`. Annotate and push
   the version tag on the validated merged commit. Publish a GitHub prerelease
   with the wheel, sdist, checksums and exact scope/evidence notes. Download the
   published assets and verify the hashes and tag target before handoff.

The current process uses an annotated Git tag and archive checksums; it does
not claim a cryptographic signing identity or build attestation. PyPI upload,
provider execution and application deployment are separate actions. GitHub's
automatically generated source archives are distinct from the attached built
sdist and have their own bytes.
