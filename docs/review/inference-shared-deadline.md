# Shared inference deadline

Status: implementation under final local qualification, 2026-09-11.
Engine-only; no cloud/server/GPU execution or EHR changes.

## Boundary

The existing startup, verified-chat composition and session controller now
accept the same optional absolute monotonic `deadline`. This is process-local
timing data, not UTC evidence, runtime authentication or a new grant. Omission
retains separate startup/session behavior. An injected `ChatSession` clock must
use the same domain as its deadline; the verified composition uses the process
monotonic clock throughout.

Startup validates exact finite deadline numbers before filesystem
projection, checks expiry again after projection/port probing, and caps startup
and readiness polling at the earlier local timeout/deadline. Clock bools,
numeric subclasses, non-finite values and conversion overflow are rejected when
a deadline is supplied. Expiry after process acquisition closes only that owner;
unresolved cleanup retains its exact handle. Control-flow interruptions survive.

The verified composition checks the unchanged deadline after readiness and before
yield. Session absolute lifetime is the earlier configured lifetime/deadline;
idle and request limits still apply. Watchdog and request waits use remaining
time. Responses crossing expiry during transport or validation cannot append
history or renew activity. This bounds acceptance and cleanup initiation, not
arbitrary syscall duration, thread scheduling, cleanup completion or billing.

## Qualification

Pending final package and documentation audit. The final selected runtime,
HTTP, chat and local materialization regression passed 387 tests in 1.84 seconds
under CPython 3.12.9 / pytest 8.4.2 with plugins disabled. It includes full/LoRA
real-target/argv/client composition with fake OS/HTTP effects, both with and
without a deadline, across success, error, interruption and timeout. The earlier
370-case run and Sol's 27-case subset are not additive qualification totals.

## Remaining work

The Modal bootstrap must freshly authenticate/rederive the original claim and
serving projections and conservatively convert its remaining lifetime into this
clock domain without renewal. Distinct inspected inference image, Python,
dependency and complete bootstrap/source pins are still missing; the current
training lock, SFT closure and source staging cannot be relabeled for inference.
Runtime verification must gate serving and SDK activation. Authenticated remote
access, exact Sandbox ownership/cleanup and live qualification remain unfinished.
