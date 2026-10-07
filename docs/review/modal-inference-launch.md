# Modal chat launch admission

Status: locally qualified, 2026-09-10.
Engine-only; no provider calls, executable remote worker or live qualification.

## Boundary

The host launch builder authenticates both exact retained chat command bindings,
uses the existing Foundation evidence reduction to establish a completed STAGE,
and requires its actual authenticated receipt and record to match all SUBMIT
predecessor fields. The STAGE provider reference commits to the exact stage
binding. A single signed launch claim commits to both commands and their shared
preparation snapshot, without repeating configuration, source, workload or
artifact inventory projections as separately editable runtime data.

Worker admission accepts one bounded canonical argument. It verifies the launch
signature and admission window, reconstructs the exact commands and preparation,
and compares them with independently supplied configuration, expected SUBMIT and
executor identity, issuer/audience/key/challenge and mount paths. The expectation
must come from trusted consumer/deployment composition, not the submitted claim.

This is a pure evidence boundary, not a grant or a runtime readiness check.
Foundation still owns actual grant consumption, predecessor admission, one-shot
dispatch and recovery. The builder can be called directly and does not itself
prove that SUBMIT authority was consumed; future SDK transport must compose it
inside the existing broker-authorized executor call. The worker trusts the
host-signed assertion of checked stage evidence; it does not receive host grant
or assessment authorities or independently replay their services.

Admission expiry bounds when the claim may be accepted, not GPU billing or the
duration of an eventual chat session. Matching configured mount roots and Volume
IDs does not prove that a live mount belongs to that Volume. Configuration/image
and runtime commitments remain unqualified until independently inspected and
matched to an actual inference lock. No mounted byte read, model download,
process startup, credential resolution or provider call occurs in these helpers.

## Qualification

Source commit: `fcd2cf163473b00b440a353e7cd07763a7376d88`. Independent source
review passed the frozen launch and wire modules, SHA-256 respectively:

- `e1ffb55ab8843900fa998e92f771d5cd115c421bb83387731195462ff3abe016`
- `0492c10c6a384596bba04b3b409dc34f1f1a4afa88b7321d0d1a6fac5e29a1fb`

Review corrections covered exact input types before callbacks, preservation of
original and owned inputs across authority/signing callbacks, and independent
worker expectation guards around verification and clock callbacks. Host signing
now requires an actual clock, rejects future issuance (including one second),
and checks freshness before signing. Worker admission retains the existing
30-second clock-skew policy; neither window replaces a bounded serving lease.

The integrated selection contains 41 independently authored launch/admission
cases and 17 real Foundation integration cases. The source owner independently
ran the latter: **17 passed in 55.10 seconds**. They exercise actual STAGE
receipts and records, forged predecessors, independent assessment authority,
time/configuration mismatches and callback mutations. The broker integration
observes consumed SUBMIT authority before transport, then proves a duplicate
dispatch does not sign or admit again. The pure builder test separately confirms
that preparing/admitting a launch does not create a Foundation SUBMIT record.
The combined regression selection passed **2,613 tests in 938.20 seconds** under
isolated CPython 3.12.9 / pytest 8.4.2, without system site packages, Modal or
Torch, and with automatic pytest plug-ins disabled. It includes all 58 final
launch/admission cases, coordinator and Foundation, provider-neutral/public API,
training/runtime, Docker provider, inference, selected Evaluator/client callers
and all selected Modal provider tests. This is not the entire repository suite;
the independent 17-case result overlaps it and is not an additional test count.

The exact Git archive produced `synaptic_tuner-1.1.0-py3-none-any.whl`,
**2,070,369 bytes**, SHA-256
`90c67d968e3906bb05b9dc11bb46878bb215c89f8db156c28bc61fd1778d2ef8`.
Independent audit verified 724 unique ZIP/RECORD entries, every recorded hash
and size, and all 716 Python members byte-for-byte against the source archive.
The packaged launch and wire modules match the reviewed hashes above.

A fresh install containing only the wheel and declared dependencies passed the
checked-in CI probe's **43 engine/Evaluator imports**, both packaged-resource
checks and four credential-free `inspect.Signature.bind` checks for the host
builder, worker admission, expectation and launch envelope. The two new modules
were imported from the installed environment, from a neutral directory under
isolated Python. Modal, Torch, pytest, NumPy and pandas were absent. The signature
probes call no entry point and assert none hides required parameters in `**kwargs`.

The unchanged 97-file training runtime lock and separate 66-member offline
worker closure (679,487 payload bytes) report `CURRENT`; both are schema-valid
in the wheel. The mixed Windows/WSL checkout's offline closure check used scoped
local `GIT_DIR`/`GIT_WORK_TREE`, never `PYTHONPATH`. Canonical skill mirrors match.
A filename-only package scan found no test paths or private artifact filenames;
it is not a content-level credential audit. No cloud resources, model downloads,
GPU jobs, push, merge or EHR changes were part of this slice.

## Remaining work

Implement explicit-client SDK transport with trusted exact Volume-to-mount
composition, then reuse the existing artifact materializer, pinned-model preparer
and vLLM controller on the worker. Capture the separate inference runtime/image
and bootstrap lock. Durable owned leases, readiness and request bounds,
exact-target cleanup and ChatSession integration remain required before live
serving. No operator-side weight upload or local Docker cloud launcher is added.
