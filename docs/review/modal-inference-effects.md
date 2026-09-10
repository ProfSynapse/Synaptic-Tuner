# Modal chat Foundation effects

Status: locally qualified, 2026-09-10.
Engine-only; no Modal SDK transport or live serving qualification.

## Boundary

`ModalChatEffectExecutor` supports retained chat STAGE/SUBMIT commands only.
`ModalChatReconciliationAdapter` provides lookup for those exact commands.
Their chat-specific descriptors and resolvers use the existing Foundation
minting helpers, broker and reconciliation service. The consumer supplies its
catalog, complete-content authority and transport. No training grant, executor
or provider transport is reused. `ModalEffectOutcome` is reused only as the
existing result value; it does not supply authentication or finality authority.

Admission reconstructs the retained chat binding, compares exact command,
preparation, payload, provider/account/environment scope and descriptor, and
guards caller and owned inputs across callbacks. Lookup derives its resolution
identity from the exact command and configured adapter. A direct call to this
private executor is not a grant check: consumers must compose it through the
existing Foundation broker. The broker owns one-shot admission, verifies actual
authenticated predecessor evidence and retains authenticated result receipts.

Transport is called at most once per admitted dispatch or lookup. Ordinary
transport exceptions cross to Foundation, which exposes closed ambiguity or
interruption diagnostics. An exception is never definite absence, and a lost
create response never authorizes a second create. A returned absence claim
still requires the consumer's existing Foundation finality verifier. Process
control exceptions are not converted into provider evidence.

The provider-free tests use an in-memory Foundation repository and synthetic
content/grant authorities. They prove adapter composition, not production
persistence, live account identity, remote launch authorization or a mounted
Volume's authenticity. CANCEL is still excluded until the owned remote target
and cleanup authority are implemented.

## Qualification

Source commit: `c6ef69d8d0a2fb299bfb6816aaea0636cccec996`. Independent source
review passed the frozen effects module, SHA-256
`deb6c50931e9ac26d7005e964948d850551cf48995a7ce36c62f604b563f15f0`.
Review corrections covered input checks before callbacks, collaborator/method
and capability snapshots, exact lookup resolution/version, and constructor
state validation before resolver minting. No Foundation runtime code changed.

The final focused run passed **59 tests in 170.02 seconds**: 51 independently
authored adapter cases and eight lead-authored real Foundation broker cases.
These include malformed input rejection, callback mutation, process-control
exceptions, invalid grants before catalog/transport access, actual stage receipt
and record validation, duplicate dispatch, orphaned lost responses, interrupted
lookup and rejection of unauthenticated finality proof. A draft catalog-swap
test was corrected to assert the callback actually ran before detecting the
transport swap; an earlier version only tested preexisting constructor mutation.

The broader regression selection passed **2,503 tests in 575.17 seconds** under
isolated CPython 3.12.9 / pytest 8.4.2, without system site packages, Modal or
Torch, and with automatic pytest plug-ins disabled. It covers coordinator and
Foundation, provider-neutral/public API, training/runtime, Docker provider,
inference, selected Evaluator/client callers and existing Modal provider tests.
That run was collected before the independent 51-case module and the additional
finality test were integrated; it includes the first seven broker cases. The
59-case run covers the final complete new test files. These are overlapping
selections, not an aggregate whole-repository test count.

The exact Git archive produced `synaptic_tuner-1.1.0-py3-none-any.whl`,
**2,062,001 bytes**, SHA-256
`359f9fb3e910ef75be120e0d540edb4d0958bad9c11f39989e6e835c5a01b7e5`.
Independent audit verified 722 unique ZIP/RECORD entries, every recorded hash
and size, and all 714 Python members byte-for-byte against the source archive.
The packaged effects module matches the independently reviewed source hash.
Metadata retains the declared `requests>=2.32,<3` dependency.

A fresh install containing only the wheel and declared dependencies passed the
checked-in CI probe's **41 engine/Evaluator imports**, both packaged-resource
checks and four additional credential-free `inspect.Signature.bind` checks for
the chat executor/lookup constructors and methods. Imports came from that
environment, from a neutral directory under isolated Python. Modal, Torch,
pytest, NumPy and pandas were absent. No provider action was invoked.

The unchanged 97-file training runtime lock and separate 66-member offline
worker closure (679,487 payload bytes) report `CURRENT`; both resources are
also present and schema-valid in the wheel. The mixed Windows/WSL checkout's
offline closure check used explicit local `GIT_DIR`/`GIT_WORK_TREE`, never
`PYTHONPATH`. Canonical skill mirrors match. A filename-only package scan found
no tests or private credential/artifact files; it is not a content-level secrecy
claim. No cloud resources, model downloads, GPU jobs, push, merge or EHR changes
were part of this slice.

## Remaining work

Implement authenticated remote launch and exact Volume-to-mount admission,
reuse the shared materializer and pinned-base model preparer on the worker,
capture a separately reviewed inference runtime/image/bootstrap lock, and add
the explicit-client SDK transport. Durable owned leases, bounded readiness and
requests, exact-target cleanup and the existing ChatSession integration remain
required before live serving. No operator weight upload or local Docker
orchestration step is introduced by this adapter.
