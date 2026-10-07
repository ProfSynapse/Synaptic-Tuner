# Owned vLLM runtime slice

This slice adds one dependency-light, internal startup surface for a local
OpenAI-compatible vLLM process. It does not register a public API. The same
change migrates both existing callers and removes the old global lifecycle
API; caller qualification is recorded in `vllm-callers-slice.md`.

`VLLMStartupSpec` has two explicit source forms. A verified-local source accepts
only a validated `ServingTarget`: a full artifact is the model, while a LoRA
artifact is attached to its exact pinned base snapshot. Both use the retrieved
tokenizer. The launcher forces offline library settings and permits only a
small execution-environment allowlist; credentials, proxies, and Python import
injection are rejected before process creation. An explicit-network source
preserves its caller-selected model, revision, tokenizer, LoRA, and environment
without claiming that they are verified, offline, or credential-free.

For LoRA, the vLLM base alias is distinct from the exact requested adapter/chat
alias, and readiness proves both names. The returned endpoint properties are
read-only, and lease cleanup serializes concurrent session/watchdog callers.
LoRA aliases use a closed identifier grammar and cannot contain the adapter
assignment delimiter or whitespace.
Successful cleanup is cached; an unresolved cleanup remains explicitly
retryable while all attempts stay serialized.

The managed server binds exact IPv4 loopback and refuses an already-listening
port before spawn (the ordinary close-to-spawn local race remains). Startup arguments and timeouts are
strict and finite. LoRA rank and the supported Mistral tokenizer mode are
explicit spec fields rather than model-name heuristics. Tensor-parallel
auto-detection and the prequantized-model fallback remain caller policy in
`vllm_setup`; this launcher accepts only a resolved positive count.
Readiness uses a direct stdlib HTTP connection (therefore no
ambient proxy handling), a bounded `/v1/models` response, exact JSON shape, and
the exact expected names. Each probe also confirms the unreaped owned leader's
recorded Linux identity without calling `poll()` or reaping it. Startup failure
and cancellation attempt bounded process-family cleanup; an unresolved cleanup
lease is retained on the original exception rather than masking it.

The HTTP timeout bounds socket operations, not a hard wall-clock interrupt.
The monotonic startup budget is checked around each probe; the later session
slice must supply the watchdog that enforces absolute conversation deadlines.
Offline and telemetry-disable environment flags reduce library egress but are
not an operating-system network sandbox.

This is provider-free unit qualification, not a live vLLM, GPU, model-load,
inference-quality, cloud, or Windows-native qualification. The owned process
primitive requires Linux procfs and POSIX process groups; native Windows may
still act as an HTTP client to an externally managed endpoint. No global
start/stop compatibility functions remain after the caller migration.
