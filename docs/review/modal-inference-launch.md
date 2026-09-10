# Modal chat launch admission

Status: implementation and local qualification in progress, 2026-09-10.
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

Pending final source review, unit and real Foundation integration tests,
regression checks and exact-commit installed-wheel verification. No completed
qualification is claimed yet.

## Remaining work

Implement explicit-client SDK transport and authenticated physical mount/source
admission, then reuse the existing artifact materializer, pinned-model preparer
and vLLM controller on the worker. Capture the separate inference runtime/image
and bootstrap lock. Durable owned leases, readiness and request bounds,
exact-target cleanup and ChatSession integration remain required before live
serving. No operator-side weight upload or local Docker cloud launcher is added.
