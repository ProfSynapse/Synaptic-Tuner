# Private Modal chat channel

Working-tree qualification, 2026-09-11. Engine only. This record does not claim
a deployed model, a live response, reviewed inference image pins, or confirmed
cloud shutdown. No cloud call has been made for this change.

## Connection and ownership

The remote executable is `tuner.execution.providers.modal.inference_entrypoint`.
It accepts no argv. Its first bounded canonical JSON line carries an exact
signed launch and independently supplied static expectation over the explicit
client's authenticated Sandbox stdin. Credential names must match the selected
Secret requirements exactly; raw values never belong in the frame. This frame
is transport, not an authorization grant or proof of physical Volume mapping.

The image must pre-create `/workspace/modal-chat/model`, `/workspace/modal-chat/base`
and `/workspace/modal-chat/scratch`. These private directories must not overlap
the three selected mounts. The entrypoint creates no directory, cloud object,
client or public endpoint. It constructs the existing HMAC authenticator and
execution-machine pinned-model preparer. Signed launch admission precedes the
preparation-token read, and the concrete bootstrap checks the inference runtime
before any model preparation. The child inference environment uses a fixed PATH
and selected CUDA runtime variables, never operator PATH, PYTHONPATH, provider
credentials, model tokens or evidence keys.

After bootstrap yields the exact existing `ChatSession`, `inference_channel`
serves canonical ready/chat/stop/closed/error frames. Every frame binds session
and signed-launch digest; requests have consecutive integer identities. Only
one input read is outstanding, and a subsequent read is released only after
processing the current request. Output writes are also observed through one
daemon task. The channel polls the existing session watchdog; it does not derive
or restart a second session deadline. EOF, malformed frames, stop, idle expiry,
request failure and output failure close that same session and owned runtime.
The executable suppresses ordinary diagnostics and text-bearing SystemExit at
the process boundary; internal control exceptions still trigger normal cleanup.

The ready frame also carries `PreparedModelIdentity` derived from the worker's
verified materialized target. The bootstrap's internal result now pairs that
portable identity with the exact session; it is not a new grant. The host must
compare its model/tokenizer refs and revisions to the authenticated workload.
Archive kind is learned from the remote verified target, never guessed from a
training option. Full/LoRA bootstrap tests cover that projection.

This replaces the plan's proposed HTTP tunnel/access-token path. The raw vLLM
server remains on loopback with no exposed Sandbox service ports, so cloud
clients cannot bypass the session's turn, history and idle controls. Stream
authentication does not substitute for Foundation authority consumption or
exact-resource binding.

## Bounds and qualification limits

The first-frame wait is 30 seconds and its maximum size is 256 KiB. Signed
launch bytes remain bounded by the existing 96 KiB limit. Subsequent request
and response bounds come from admitted configuration, capped by the channel
codec at 1 MiB. Worker-input/output time bounds are logical waits: Python cannot
cancel an arbitrary already-blocked syscall in a daemon thread. The worker's
runtime watchdog and provider lifetime policy remain necessary independent
cleanup mechanisms, not a proven hard billing deadline.

Measured SDK 1.5.4 source distinguishes `Sandbox.wait()` (returns None) from
`terminate(wait=True)` (returns the exit code). Sandbox stdout is text and
line-oriented. Its internal line accumulator has no application-frame bound
before newline, so the host's post-yield frame validation is not a hard bound on
SDK memory allocation. The fixed worker protocol and suppressed diagnostics
are part of the trusted source contract; no private SDK mutation is used to
pretend this limitation has disappeared. Conversation response frames travel
through Modal's stdout/log transport.

The entrypoint and worker channel are now minimum required members of the
inference runtime source inventory. The actual inference manifest, complete
dependency lock and worker closure remain absent until image/runtime inspection
and independent review. No training lock is reused and no verifier is bypassed.

The initial integrated entrypoint/channel/runtime-verifier selection passed
65 tests in 5.63 seconds on the existing isolated CPython 3.12.9 / pytest 8.4.2
lane, with plugins and pytest cache disabled. This includes an actual synthetic
environment-backed HMAC launch reaching the real missing-lock denial before
model preparation. Positive executable/channel cases use an in-process backend
and replace runtime acquisition; they do not load ML packages or call Modal.
The earlier 40-test selection is a subset, not an additive result. These counts
precede the required ready-model projection update; final qualification must
rerun the affected selections with that protocol change.

Correction (2026-09-11, recovered protocol qualification): the nine-file
entrypoint/channel/client/runtime/capture/inspection/bootstrap selection passed
173 tests in 137.82 seconds after the required ready-model change. The channel
client's 41 tests also passed separately in 0.25 seconds; they are a subset.
One test had computed its future deadline at collection time, making it fail
only after earlier tests consumed that margin; the deadline is now computed at
test execution. This run preceded integration of the separate image-schema and
source-freshness changes and does not qualify the final SDK transport or adapter.

The unchanged training runtime source lock checked CURRENT at 97 members.
Further integrated host-adapter, inspection, regression and packaging results
will be recorded only after their actual runs complete.

Correction (2026-09-11, integrated regression): the broader inference selection
passed 560 tests in 1134.05 seconds before the final creation-ownership changes
and new transport/consumer tests were integrated. A subsequent nine-module run
finished with 173 passing and one failing test in 172.40 seconds. Its twelve
consumer integration tests passed, including interrupted lease transfer; the
failure was the transport test for channel-constructor cleanup ownership.
That failed run is not a green final qualification. The selections overlap;
their counts must not be added.

Correction (2026-09-11, repaired qualification): after the completed-cleanup
ownership fix below, the same nine-module selection passed 175 tests in 193.74
seconds. The channel-client selection separately passed 42 tests in 0.18 seconds.
The final transport tests then split the previously timing-sensitive scenario
into deterministic completed and pending cases; that 21-test selection passed
in 95.32 seconds against the same source. These overlapping provider-free runs
qualify the covered source behavior, not an image, model response or deployment.

## Remaining live path

Correction (2026-09-11, SDK integration): the explicit-client transport now
implements SDK-free STAGE and one SUBMIT attempt behind the existing Foundation
executor. It authenticates the actual retained STAGE record/assessment before
signing launch content, validates the complete startup frame before provider
reads, resolves exactly the selected app, final Image and three Volumes, and
passes only named Secrets. It creates one private stdio Sandbox, sends one
startup frame and publishes its exact ready lease after model-identity checks.
Its 18 tests passed independently against the integrated lead tree in 70.64
seconds. Tests use fake SDK effects and actual Foundation launch evidence;
this is not deployment qualification. The consumer `ModalRunChatRuntime`
is covered by the twelve integration tests recorded above, not that older
transport-only result.

Review caught and corrected post-create control-interrupt cleanup, mutation
during binding authentication, startup validation happening after allocation,
and reuse of the channel constructor's existing cleanup owner. Pending creation
retains process-local late-handle evidence, with no create retry or listing/
adoption. Known ownership is retained before bounded cleanup, including when a
second interruption occurs. A late cleanup callback's completion result is not
retained as authoritative shutdown proof. After process death, this in-memory
ownership is not durable recovery; preserve Foundation ambiguity and never
infer that the resource is absent. Provider lifetime limits remain necessary.

Correction (2026-09-11, ownership transfer): creation ownership is registered
before starting its worker, and returned handles are retained before queue
publication. The capture launcher has the same interruption protection. One-use
ready-lease transfer keeps its retained value available to recovery after take.
Independent review passed these current-process ownership guarantees. The
subsequent failing test exposed a separate fast-cleanup race: constructor errors
must retain the exact channel lease even when termination already completed,
otherwise the outer transport can terminate again. That contract is now fixed;
deterministic completed and pending regressions check exactly one termination.
This does not turn process-local ownership into durable recovery or a hard
provider billing bound.

Inspect the chosen image on a bounded credential-free CPU Sandbox,
review and package actual inference runtime/dependency/source locks, then qualify
the exact pushed source with one authenticated trained run, real chat response
and confirmed exact-instance termination. The CPU candidate report alone cannot
qualify a final installed engine image or authorize a GPU session.
