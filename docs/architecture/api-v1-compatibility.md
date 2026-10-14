# API v1 compatibility policy

This policy defines compatibility commitments for published stable API v1 releases.
The package compatibility version is `1.1.0`; metadata alone does not establish
publication or backend qualification. [GitHub release records](https://github.com/ProfSynapse/Synaptic-Tuner/releases)
bind the published source, artifacts and accepted scope. The
[support matrix](../plans/api-v1-support-matrix.md) identifies operational support
and dated qualification evidence; this document defines public compatibility.

## Scope of the stable contract

The reviewed `synaptic_tuner.api.v1.__all__` inventory, its documented public request/result types and operation signatures, and the corresponding versioned public schemas define API v1. Contract availability and operational support are separate: the support matrix identifies each supported composition, method/provider/profile combination and operation. A facade, exported type, packaged profile or passing CPU fixture does not qualify a backend, account, model, image or hardware combination. Optional families remain contract-only or host-supplied where the matrix says so.

The host supplies configuration, credentials, authorization, stores and product state. Documented reference compositions demonstrate use of existing ports; their implementation modules are not an additional blanket stable API. Checkout consumers, maintenance tools, legacy CLIs and `tuner.*`, `shared.*`, `SynthChat.*`, `Evaluator.*` or trainer internals do not acquire the public v1 compatibility promise merely because they ship in the distribution. Their scope and pinned dependencies are documented separately. This policy does not promise an HTTP service, remote worker restoration, universal provider parity or a permanent support lifetime.

## Compatibility and deprecation

For stable v1 releases, preserve existing documented public signatures, required fields, enum meanings and canonical serialization of unchanged admitted documents. Do not silently change the interpretation of an existing versioned document, replace a retained identity, or reinterpret unknown effect state as retry authority. A breaking public shape or meaning requires an explicitly versioned contract and documented adoption; a new schema tag alone does not supply a migration.

Additive public types/operations and newly supported backend combinations require review, documentation and appropriate contract/installed evidence. They must preserve existing accepted paths. Each schema/parser determines its closed fields, enum values and deliberately open metadata/configuration maps. Preserve those explicit open fields and extension points; callers must not assume an older reader accepts arbitrary extensions outside them. Security/correctness fixes may reject inputs outside the documented contract, with material compatibility effects described in release notes.

Deprecation must identify the affected public symbol or document, its supported replacement if one exists, and the intended transition in release notes. Notice alone does not permit removal or a breaking signature/interpretation change within stable v1: that requires a new versioned public contract and an explicit transition. Do not promise a fixed grace period or automatic conversion that is not implemented. Exact provider SDK, launcher, runtime image and workload releases remain independently pinned; API v1 stability does not make those releases interchangeable.

## Documents, identities and bounds

Package release versions, the public API namespace version and document schema versions are distinct. A package version bump does not reinterpret retained v1 records or supply migration authority. Use the exact schema/tag and parser for the admitted document. Retain its original canonical bytes, declared digest domains, model/source/configuration bindings and applicable runtime release. Unknown fields are refused where the declared schema/parser closes them; explicitly open metadata/configuration maps retain their declared acceptance rules. Invalid numbers, wrong types and mismatched bindings remain refusals under the applicable contract. Do not repair, relabel or round-trip a historical record through a different schema to make it admissible. Any migration must be an explicit reviewed operation that preserves provenance; none is promised generically.

Limits are part of each operation's contract, not one universal budget. Preserve declared row, page, stream chunk, artifact, request/response and execution bounds. An explicit null output-token cap removes only that request-level token ceiling. It does not remove context, byte, time or spend controls. Pagination cursors are opaque and scoped to their documented request/store semantics, not durable authorization or a cross-store migration format.

## Errors and unknown effects

Where a facade exposes a typed operation error and code enum (for example Data, Evaluation and Runs), callers should branch on that public code. Exact enum meanings are contractual; messages, traceback text and internal/provider exception classes are not. Public type/shape checks can still raise `TypeError` or `ValueError`. Training and Artifacts do not currently provide one universal typed operation-code enum; do not infer one or parse their messages as a stable protocol. Host adapters may add documented closed diagnostics without exposing raw provider data or credentials.

Distinguish rejected input, unsupported capability, conflicts/integrity failures, known terminal outcomes and unresolved effects. A timeout, missing receipt/catalog row or failed local readback does not prove that an upstream request or submission was absent. Preserve its consumed claims and uncertain spend/effect state. Repeated public invocation is not general retry authority: only an operation's existing authenticated idempotency/lookup behavior may reuse a known result. Unknown submission/publication/cancellation must not be blindly reissued. Supported metered paid compositions stop later paid dispatch after request/usage uncertainty; this is not a provider-side abort or an invoice ceiling.

Cancellation acknowledgment, requested cancellation and observed terminal state are distinct. Terminal success/failure can remain its actual outcome despite cancellation. A missing acknowledgment cannot be fabricated from terminal status. Reconciliation is limited to each supported adapter's authenticated known identity/receipt evidence; it does not discover or adopt an unknown submission. Retained reads require original keys, bindings and supported host dependencies. They do not restore one-use input leases, active model calls, arbitrary interrupted rows or trainer checkpoints.

## Qualification and release evidence

A supported installation claim names its Python/platform and tested paths. Base imports/assets/help on Linux Python 3.10–3.14 and Windows/macOS Python 3.12 are separate from POSIX stores/workers, portable SFT requirements and exact effectful Modal CPython 3.11.14/SDK pins. Wheel metadata is not execution qualification. Checkout-owned examples and source build inputs remain explicit prerequisites.

Publish stable support only after the selected release's contract/installed gates and independently scoped live qualifications pass. Evidence names its exact source, wheel, configuration and limits; historical or scripted success cannot promote current capabilities. Keep experimental/unqualified combinations and disabled operations visible in the support matrix rather than describing them as universal v1 support.
