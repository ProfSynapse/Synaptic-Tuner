# Closed vLLM startup observations

Status: implemented and provider-free reviewed; not released or live-qualified.

The October 4 9B smoke completed two training steps, saved all five artifacts,
and failed same-job evaluation before its first request. Its phase trace records
a roughly 300-second readiness window, but does not distinguish connection,
HTTP, response-shape, served-name or process-state failures. Discarded child
output cannot be recovered retrospectively. No root cause or timeout fix is
claimed, and the consumed training submission is not replayed.

`VLLMStartupDiagnostic` adds bounded observations to a readiness exception:
closed failure/probe labels, last-observed leader liveness, elapsed seconds and
probe count. Unproven liveness, invalid timing and counter overflow remain null.
The original boolean readiness interface and process-family cleanup behavior
are preserved. Probe state is thread-local and cleared before each attempt.
No response bodies, model names, URLs, paths or exception text are retained.

The evaluation record may carry `startup_diagnostic` on startup failure or a
subsequent identity failure. Publication accepts only the exact exception and
diagnostic types; readback independently validates the closed shape and bounds.
Legacy records omit the field and retain unchanged canonical bytes. This is a
terminal diagnostic, not live training telemetry or proof that vLLM loaded.

Independent review checked disclosure, legacy compatibility, cleanup and
unknown-state semantics. The unsupported `leader_exited` value was removed from
readback admission after review; a false ownership/liveness guard cannot prove
process exit. Liveness means the last observation during startup, not a promise
about state after timeout or cleanup.

Initial provider-free verification: 114 Windows tests passed, two skipped;
41 Linux runtime/readiness tests passed. On October 4, the startup-probe slice
then passed 158 combined Linux checks covering the probe, host launcher,
readiness observations, evaluation diagnostic publication and process cleanup.
The isolated test environment does not change launcher or ML dependencies.
The diagnostic publication fixture now isolates unrelated optional evaluator
imports and asserts that its injected serving failure was actually reached;
an earlier import failure had been misreported as the intended test condition.
These provider-free results do not qualify live serving.

Runtime lock checks report stale source in this dirty development checkout.
Do not refresh all dirty source indiscriminately. Before deploying this change
as a packaged train/evaluate release, isolate the reviewed slice, refresh the
affected hashes deliberately, and repeat required package/CPU qualification.
The separate model-first diagnostic described in
`examples/model_chat/STARTUP_PROBE.md` selects an existing provider image and
records host-observed hashes of the newly mounted diagnostic source. It is not
a packaged-runtime release, old-source replay or full source attestation.
Its provider-free tests and independent review must pass before allocation.
No image dependency pins are changed by the diagnostic.

Evaluation-only recovery and chart-ready progress transport remain separate
plans in `docs/plans/evaluation-only-recovery.md` and
`docs/plans/training-progress-streaming.md`; neither is implemented by this
diagnostic change.
