# Private Modal chat channel

Working-tree qualification, 2026-09-11. Engine only. This record does not claim
a deployed model, a live response, reviewed inference image pins, or confirmed
cloud chat shutdown. The initially provider-free work is followed by the dated
CPU-probe correction below; no GPU model response is claimed.

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

Correction (2026-09-11, exact-source package qualification): source commit
`bbdbbd56d9af4d18b5d4d3dab5e7cfdf3d1494d3` was archived with Windows Git and
built offline with no dependency resolution. The resulting
`synaptic_tuner-1.1.0-py3-none-any.whl` is 2,108,895 bytes, SHA-256
`71474dc27b2eb1566984b986e3b0bb13664c275584c4316962fff30559acf216`.
Installed into a fresh dependency-only environment, it passed the checked-in
neutral-directory import/resource probe: 51 engine imports and both training
resources, with Modal, Torch, NumPy, pandas and pytest absent. All three
inference resources remain absent and unqualified; this check does not bypass
their concrete runtime denial. The final two deterministic transport regressions
also passed after integration in 8.25 seconds (the other 19 were deselected).
The training source lock remained CURRENT at 97 members and the offline SFT
closure at 66 members / 679,487 payload bytes. Skill mirrors were verified in
sync, and both maintenance scripts now trigger PR conformance. No push, merge,
cloud operation or live response is claimed by this record.

Correction (2026-09-11, operator login selection): the maintenance capture CLI
now accepts explicit `--modal-profile NAME`, resolving only that named SDK
profile's credential pair with `use_env=False`; its default remains environment
credentials only. Neither path falls back to the other. The installed SDK 1.5.4
configuration shape was checked without credential lookup, with its configuration
file redirected to `/dev/null`: `from modal.config import config` supplies the
singleton, not the `modal.config` module. Corrected module-shaped tests prevent
that distinction being hidden by a fake. The integrated capture/inspection
selection passed 38 tests in 0.26 seconds. This source-only maintenance update
does not change the installed engine wheel qualified above.

Resume checkpoint (2026-09-11): the operator-selected CLI profile was identified
as `synaptic-labs` without printing credentials or listing cloud objects. The
committed capture/inspection scripts at
`228920705f4ccfb3b42daceeee22aa45a2f12acb` were extracted from an immutable Git
archive. The proposed CPU probe selects only app `synaptic-training-v1` in
environment `synaptic-smoke-v1` and base image
`docker.io/vllm/vllm-openai@sha256:116aa00ee0b68855616a56e1d7e1ae937e591a8bd6969ee45cbcedb246ddf355`.
The platform approval reviewer rejected execution before process launch because
this stored-profile cloud operation can incur charges and requires exact
moment-of-execution approval. No provider request, Sandbox allocation, image
capture or GPU chat occurred. Do not bypass this rejection; obtain explicit
approval for the same bounded command before attempting it again.

Correction (2026-09-11, explicitly approved CPU attempt): the operator approved
the exact command and it executed once. The capture process exited 125 with
`capture_failed`, returning Sandbox `sb-Hf9Gir2Fxu8Ns6b2BppBeg` and reporting
`cleanup_requested=true`, `cleanup_confirmed=true`, `ownership_known=true`.
The code requires exact-handle termination followed by a non-pending poll before
reporting cleanup confirmed. No candidate runtime metadata was returned, and no
second attempt was started. The preserved closed process result is in
`evidence/modal-inference-cpu-probe-2289207.json`. This result does not identify
the remote inspection failure cause; exact-instance read-only diagnosis is next.

The existing maintenance command now supports `--read-sandbox ID` for this
diagnosis. It requires an exact stopped Sandbox, performs no create/list/stop
calls, and reads bounded stdout/stderr under one 30-second deadline after client
construction. It preserves the known exit code and returns only validated
candidate metadata, allowlisted inspector errors or fixed unknown-output codes.
The integrated capture/inspection tests passed 49 tests in 0.26 seconds. This
read mode does not authorize another probe or adopt the Sandbox for serving.

Exact-instance diagnosis (2026-09-11): the committed read-only command read
`sb-Hf9Gir2Fxu8Ns6b2BppBeg` without allocation or termination. The provider exit
code is 125 and the inspector's validated stderr reports `METADATA_INVALID`.
The result is preserved in `evidence/modal-inference-cpu-read-2289207.json`.
This identifies the metadata-validation class, not which metadata constraint
failed; it does not justify weakening duplicate checks or inventing runtime pins.

Follow-up preparation (2026-09-11): the inspector now distinguishes distribution
count limits, enumeration failures, metadata reads, invalid names, invalid
versions and duplicate normalized identities with six closed error codes. No
package values or raw exceptions are emitted, and existing limits and duplicate
rejection remain unchanged. The capture reader accepts these exact codes.
The integrated inspection/diagnosis selection passed 59 tests in 0.27 seconds.
This is improved diagnosis, not a claimed fix for the image's unknown metadata
condition. A further CPU allocation has not yet been made.

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
