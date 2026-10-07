# Closed vLLM startup observations

Status: recovered onto the current branch and provider-free reviewed;
one saved-adapter startup-only probe passed. Same-job generation is not qualified.

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

October 7 recovery: the earlier development was preserved in WIP commit
`fd2a4b14`, outside the merged checkout at `0936e8c3`. Only the startup slice was
recovered. The latest combined Linux run passes 169 tests. The only changed
member in the inference lock was `tuner/inference/vllm_runtime.py`; the reviewed
hash-only refresh and subsequent read-only check report CURRENT. The separate
training lock also reports CURRENT. Neither check qualifies a new deployed
inference image. The saved-adapter diagnostic remains an explicit source overlay
on the selected existing image, with its own source/configuration records.

The optional saved-input manifest is checked before provider effects, copied
into exclusive private staging files and checked again before submission. The
worker independently rejects wrong mounted bytes before base preparation.
The new result records readiness time while retaining legacy v1 readback.
These changes do not add a chapter request or establish writing quality.

The October 7 saved-adapter probe reached readiness in 229.175754 seconds after
422.422373 seconds of model preparation, then verified exact Sandbox shutdown.
This is below the earlier 300-second readiness window, so increasing the probe's
configured window to 600 seconds is not evidence that the original failure was
caused or fixed by a timeout. The logs confirm LoRA registration, 16.99 GiB
model-loading memory and 49.83 GiB available KV cache. No request was sent and
the theoretical cache concurrency must not be reported as measured throughput.

Evaluation-only recovery and chart-ready progress transport remain separate
plans preserved in WIP commit `fd2a4b14`; neither is implemented by this
diagnostic change or restored as part of this narrow recovery.
