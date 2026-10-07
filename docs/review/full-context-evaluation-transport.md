# Full-context evaluation transport

The same-job evaluation API already accepts inline scenarios. Complete context
bundles exposed a size-policy mismatch: three measured prompts serialized to
roughly 365 kB for a smallest-case set and 787 kB for a representative set,
before adding workload and coordination metadata. The old 128 KiB evaluation
limit could not carry those cases. This is a generic text-transport capability,
not a corpus-specific recipe, provider workaround, or output-token budget.

## Coordinated finite bounds

| Boundary | Allowance |
| --- | ---: |
| Post-training evaluation configuration | 1 MiB |
| Canonical training configuration | 1 MiB |
| Compiled workload, including isolated-trainer admission | 1 MiB |
| Coordinator material containing configuration and workload | 2 MiB |
| Modal ordinary bundle member | 2 MiB |
| Modal signed packaged dispatch including encoded workload | 2 MiB |

Coordinator bundle aggregate-member, canonical-bundle and transport bounds stay
4 MiB, 6 MiB and 8 MiB respectively. The generic bounded-JSON parser retains its
512 KiB default; only explicit training document and coordinator callers request
the larger allowances. JSON structure, duplicate-key, finite-number, digest,
signature, exact-source and model/runtime-pin checks remain in force.

Each allowance includes the serialized container, not merely prompt characters.
Escaping and encoding add overhead, and the coordinator includes the resolved
configuration and compiled workload. Admission at an inner boundary does not
promise admission at every enclosing boundary. Reject oversize input before
submission; never truncate prompts or silently select smaller examples.

## Serving and retained-response limits

The same-job client has a 1 MiB HTTP request bound and a 1 MiB HTTP response bound.
The response-retention correction aligns the per-case UTF-8 text bound with that
same 1 MiB response budget. HTTP JSON overhead still counts; a text payload at
the bound is not a promise that its larger HTTP envelope fits. The signed
evaluation document keeps its separate 16 MiB aggregate bound. Publication and
authenticated readback must use that evaluation-specific canonical serializer,
not the 1 MiB workload-document serializer; workload/configuration limits are
unchanged. No text is truncated to satisfy these boundaries.

`max_tokens: null` omits the request-level output token ceiling; it does not
remove context, time, transport or artifact bounds. Scheduler concurrency remains
configured through the existing evaluation and vLLM settings. Mechanical
nonempty/natural-completion checks do not assess writing quality.

The measured failure behind this correction was a response accepted by the HTTP
client and then discarded by the old 64 KiB recorder bound. Raising the recorder
bound cannot recover that discarded historical text. The regression contract
covers >64 KiB Unicode responses, exact/one-byte-over response and aggregate
bounds, larger signed publication/readback, and unchanged canonical bytes for
legacy-sized records. This is a provider-free correction, not a live serving
qualification or permission to replay a consumed training attempt.

## Verification and release

Provider-free regressions must carry three complete synthetic, non-ASCII context
bundles through compilation, coordinator material, bundle and signed dispatch
round trips with byte/digest equality. Test limit boundaries and rejection before
parsing or effects where readers accept serialized input; exercise the separate
isolated-trainer workload readers. Existing small recipes must retain identical
canonical bytes and digests.

Source edits require the existing hash-only offline-trainer, packaged-worker,
Modal runtime and inference lock checks/refreshes as applicable. Preserve fixed
inventories and all runtime/dependency pins. A passing offline check is not live
provider qualification or authority to replay any previous submission.

## Verified checkpoint (2026-10-02)

The combined provider-free regression suite passed 259 tests on Windows with
CPython 3.12.3. It includes complete Unicode-context round trips through the
compiler, coordinator material, bundle transport, authenticated dispatch and
packaged-worker admission, as well as existing contract and closure regressions.
The separate isolated-trainer exact-limit and one-byte-over-limit tests passed.
All four source-lock/closure maintenance checks report CURRENT; independent
review confirmed unchanged inventories and model, image and dependency pins.

A read-only compilation using three actual held-out contexts (252,355, 249,094
and 259,381 UTF-8 prompt bytes) produced a 779,608-byte workload and preserved
every prompt exactly. This exercised the draft data, not an approved final
training recipe. No provider call or paid training submission was made.
