# Provider-free coordinator conformance slice

Status: checked-in qualification candidate; CI execution not yet observed.

## Claim and non-claims

The `Provider-free coordinator conformance` workflow is a bounded regression
lane for the provider-neutral coordinator and Foundation contracts, the generic
fake-provider family, and the Modal adapter and runtime. It runs the existing
pytest surface, including the bounded Modal effect, reader and staging adapters, without
installing the Modal SDK and with common provider credential variables
explicitly empty.

Passing this lane means those synthetic and local contracts conform in the
tested source revision. Fake-provider success is **not** evidence of a live
Modal API call, authenticated provider observation, cloud execution, training,
artifact retrieval, model quality, release publication, or spending approval.
It cannot activate the currently non-operational Modal coordinator adapter.

## Exact test surface

The workflow executes:

- `tests/contract/test_provider_neutral_foundation_v1.py`
- `tests/contract/test_modal_optional_dependency.py`
- `tests/contract/test_modal_runtime_lock.py`
- `tests/contract/test_modal_runtime_lock_regeneration.py`
- `tests/contract/test_public_runs_api_v1.py`
- `tests/contract/test_public_training_api_v1.py`
- `tests/execution/coordinator_v1/`
- `tests/execution/foundation_v2/`
- `tests/execution/test_fake_provider_v1_conformance.py`
- `tests/execution/test_mutation_broker.py`
- `tests/execution/providers/test_modal_*.py`

The Modal adapter test drives the real generic coordinator against its
synthetic Foundation executor, verifies refusal by production preflight, and
imports the adapter in a fresh process without loading `modal`, consumer code,
SQLite, or the old Modal training lifecycle. The optional-dependency contract
also checks the SDK-free public import boundary and statically rejects
module-scope SDK imports from the provider package.

Correction (2026-09-09): integration expands the lane to the complete
`test_modal_*.py` provider-free set, including the existing runtime regressions
and the new staging boundary. Public API and mutation-broker contracts and
the hash-only runtime-lock maintenance tests are also included. This does not
add ML execution, GPU, cloud, or live-provider tests.

## Environment and dependencies

CI uses Ubuntu 24.04 and CPython 3.12. The install is deliberately limited to:

- a wheel built from the local project, installed with its four declared
  runtime dependencies;
- the declared supported test range, `pytest>=8,<9`;
- `numpy>=1.24,<3` and `pandas>=2,<3`, because the repository-wide
  `tests/conftest.py` imports both during collection even though this slice does
  not use its DataFrame fixtures;
- the declared build requirements, `setuptools>=68` and `wheel`, to build the
  candidate without build isolation.

The Modal extra is not installed. Pytest plug-in autoload is disabled, the
cache provider is disabled, and no provider credentials are supplied. Tests in
this surface use in-memory fakes and local subprocesses; the workflow neither
invokes a provider CLI nor performs an authenticated cloud operation.

Before pytest, CI changes to the runner's temporary directory and imports the
installed wheel's preparation, common binding, effect, reader, staging and launch modules. It
also reads and parses the packaged `modal-runtime-v1.lock.json` and
`offline-sft-worker-v1.json` through `importlib.resources`. This checks that the
provider-free imports and required package data do not succeed merely because
the repository root is the current directory.

## Repeatability and qualification

The checked-in command is:

```text
python -B -m pytest -q -p no:cacheprovider \
  tests/contract/test_provider_neutral_foundation_v1.py \
  tests/contract/test_modal_optional_dependency.py \
  tests/contract/test_modal_runtime_lock.py \
  tests/contract/test_modal_runtime_lock_regeneration.py \
  tests/contract/test_public_runs_api_v1.py \
  tests/contract/test_public_training_api_v1.py \
  tests/execution/coordinator_v1 \
  tests/execution/foundation_v2 \
  tests/execution/test_fake_provider_v1_conformance.py \
  tests/execution/test_mutation_broker.py \
  tests/execution/providers/test_modal_*.py
```

An earlier 896-test comparison used pytest 9.0.2 while the project declares
pytest below 9. It remains useful regression evidence, but it is not a release
qualification under the declared dependency contract. This workflow enforces
the supported major-version range before collecting tests.

Local authoring verification collected and passed 644 tests in the subset that
predated the effect and reader additions, in 155.84 seconds, using CPython
3.12.9 and pytest 9.0.2 with an empty inherited environment, plug-in autoload
disabled, and explicit candidate
imports from a separate clean checkout. That is same-environment regression
evidence only: because pytest 9 is outside `pytest>=8,<9`, it is not the
supported-dependency CI qualification that this workflow is designed to
provide. The effect and reader additions and the wheel-installed neutral-CWD
smoke have not been executed in this worktree. No workflow was dispatched while
authoring this slice.

Correction (2026-09-09): the integrated lead candidate subsequently passed
1,070 tests in 166.20 seconds in a clean temporary CPython 3.12.9 environment
with pytest 8.4.2, no system-site packages, and no Modal SDK. This included the
final reader, effects and staging changes, plus all existing Modal tests, but
preceded launch and lock-regenerator integration. The lock-regenerator module
then independently passed 13 tests in 0.29 seconds. These are distinct runs,
not a claim of one combined 1,083-test run. The installed non-editable wheel
also passed the neutral-working-directory import and both package-resource
checks. The source lock reports `CURRENT` for all eight members, and skill
mirrors pass their synchronization check. No GitHub workflow has run yet.

Later combined checkpoint (2026-09-09): **1,091 passed in 164.73 seconds**
after launch and lock-regenerator integration. A subsequent test-only fixture
change binds its deterministic authentication tag to the complete launch
payload; all eight launch tests passed again in 5.57 seconds. The rebuilt
launch-integrated wheel passed SDK-free imports and both resource checks.
Wheel SHA-256:
`deab14b0c31e7841737dcf4276dc6d72d7d707e49f083052c385f4584a0e1fee`.
Bundle, resolved-material and remote-wire additions remain outside this
checkpoint. The workflow test wildcard will include their provider tests as
they are integrated; the new generic material test requires an explicit entry.

The workflow pins external actions to immutable commits. On 2026-09-09 the lead
verified the official upstream release refs with read-only `git ls-remote`:
`actions/checkout` tag `v4.3.0` resolves to
`08eba0b27e820071cde6df949e0beb9ba4906955`, and `actions/setup-python` tag
`v5.6.0` resolves to `a26af69be951a213d495a4c3e4e4022e16d87065`.
The mutable `checkout@v4` ref differed and is not used. Until a run is observed,
this document claims only local test evidence, not a green CI or release gate.
