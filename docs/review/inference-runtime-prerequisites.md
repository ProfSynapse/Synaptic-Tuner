# Inference runtime prerequisites

Status: implementation and local qualification in progress, 2026-09-10.
Engine-only; no cloud, model acquisition, real server or GPU execution.

## Boundary

`VLLMStartupSpec.python_executable` selects the exact interpreter used as
`argv[0]`, with a natural default captured from the current interpreter. The
runtime rejects noncanonical, relative, malformed or oversized POSIX path text
before probing a port or spawning. This is selection, not proof of executable
existence, identity, compatibility, dependency contents or an inference lock.

The existing generic backend client accepts an optional `max_request_bytes`.
The verified-chat composition and local run-chat adapter use a 1 MiB default
and reject invalid bounds before startup. The bound covers the complete encoded
UTF-8 JSON request body, including messages, model and generation fields, not
only the prompt. It excludes HTTP headers, request lines, TLS/socket buffering
and total backend memory. History and response bounds remain separate.

Bounded bodies are validated and encoded before HTTP transport creation, then
sent as the exact checked bytes. Malformed or oversized local chat serialization
does not enter the network retry loop. Bodyless health/model-list reads stay
bodyless. Existing proxy/redirect settings, response bounds, process ownership
and cleanup semantics remain in their existing components; no extra server,
request protocol, model downloader or authority layer is added.

## Qualification

Pending final independent source review, integrated regression tests and
exact-commit installed-wheel checks. Full/LoRA integration uses actual
materialization, serving targets, runtime argument construction and backend
serialization with fake process/readiness/HTTP effects. This is not real vLLM
startup, model inference or provider qualification.

## Remaining work

Bind admitted Modal serving configuration to these runtime/client fields,
complete the worker/bootstrap, and inspect the separate inference image,
Python/dependency and source lock. Authenticated remote access, exact Sandbox
ownership and bounded cleanup remain unfinished. A local process/session timer
is not a provider billing guarantee.
