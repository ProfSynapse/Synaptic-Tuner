# Offline SFT worker diagnostic boundary — 2026-10-20

The Docker attempt on engine `9a36aaaac30877c8a21da6f6275047eefbc773b1`
and host `2e774416f8b70910824ba9c4fdf91f2f2c505218` failed with no artifacts.
Its pinned image was
`sha256:e737a0a091474c4892690690e712bc1f932f108856a3d5df60383e1cd9436e1e`.
Container `07ecde6d5b798abf51eb85595be165039cdc1043295cbd0a5b0d3d526064bfd7`
exited 23 at 12:28:41Z with `OOMKilled=false`. The retained child stderr was
only `OFFLINE_SFT_WORKER_REJECTED`; stdout was empty. These facts establish
worker invocation, not trainer entrypoint execution, a missing dependency,
OOM, or a cause.

The retained 6,349-byte workload was read without running a container or
trainer and matched SHA-256
`f33b68e492571e9ea73263519c8885c45900961e93465e91744144939688fcc8`.
Its raw-text, three-step, non-4-bit SFT configuration projected through the
existing `_append_sft_arguments` into 34 arguments accepted by the worker
validator. This checks the configuration-derived argument tail; the original
child argv was not retained. The earlier exact 68-member staged closure check
also passed. Worker and argument-builder source were unchanged between the
attempted commit and the implementation base `c14fc325`.

The existing two-generation CPU test exercises runtime decoding, invocation
construction, closure validation, and guarded source import with a substituted
trainer. It passed before this change. Neither that test nor argument validation
reproduced the original rejection. No ML import, model load or training was
performed during this diagnosis. A read-only Docker inspection over WSL failed
with `Failed to initialize: protocol not available`; ordinary retained-file
reads worked. No Docker state was changed.

## Change and scope

The worker now emits finite phase/category labels on its existing stderr path,
plus an optional last admitted trainer milestone through the already-existing
`__packaged_runtime_phase__` hook. It retains failure exit code 2, exact closure
validation, argument validation and import-origin checks. Exception messages,
arbitrary exception attributes, paths, arguments and traceback frames do not
enter the diagnostic. Unknown milestones clear the previous marker. A normal
zero `SystemExit` still passes the final owned-module origin check; nonzero or
text exits fail closed. Unclassified failures retain the generic marker.

This change adds diagnosis, not a demonstrated repair of the historical
failure. It does not classify Torch OOM or identify missing imports. Library
output is separate from the worker's own bounded diagnostic line. No profile,
model, image, runtime, dependency or hardware pin is changed.

CPU regressions cover every worker phase, exception categories, all existing
finite trainer hooks, hostile exception formatting/attribute methods, unknown
hooks, silent success, postcheck after zero exit, guard refusal of an unlisted
owned import, and raw-text/message argument-builder agreement. Subprocess
fixtures execute the actual isolated bootstrap and staged closure, substitute
only the trainer boundary, and assert that Torch, Unsloth, Transformers and
TRL were never imported.

Validation on the implementation branch:

- Worker and SFT runtime suites: **182 passed**.
- SFT/GRPO/packaged/HF closure maintenance, Modal runtime/inference lock
  contracts and packaged-child suites: **248 passed, 1 skipped**. The skip
  requires Torch, intentionally absent from the supplied nontraining venv.
- All seven checked-in manifest/lock maintenance checks: **CURRENT**.
  Recursive before/after comparison confirmed unchanged keys, inventories,
  ordering and non-hash pins; only hashes and byte counts changed.
- Canonical skill mirror check, Ruff 0.16.10 on changed Python files and
  `git diff --check`: passed.

Tests used the supplied nontraining CPython 3.12 interpreter with
`-B -m pytest --noconftest -q -o addopts=''` and explicit test files, one
process at a time. These suites define their own fixtures. The repository
root conftest imports unrelated NumPy/Pandas data fixtures, absent from that
venv; the initial unfiltered command stopped at the NumPy import before
collection. No packages were installed into the shared test environment.
The system Python also lacked `packaging` for the inference-lock maintainer;
its successful check/write/check used the supplied test interpreter.

## Fresh qualification prerequisite

Root must independently review and integrate the change, verify every affected
fixed closure and Modal training/inference lock, and build/install the exact
new engine wheel. The fresh host must pin that pushed engine gitlink, bind its
new closure digests, and retain a fresh source admission and consumer material
before any effect. Use a genuinely new run/request/grant and private roots,
with the normal checked-in preparation and public consumer commands. Preserve
all old consumed claims, containers and cache evidence. HF remains paused.

A fresh bounded qualification can establish its own result or retain the new
phase/category/milestone for a targeted follow-up. It cannot retroactively
localize the old failure, prove writing quality, or authorize replay.
