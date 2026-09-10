# Bounded chat-session slice

`Evaluator.chat_session.ChatSession` controls one in-memory conversation with an
already-ready generic `BackendClient` and an exclusively owned runtime lease.
It does not load models, construct backend clients, contact a provider, or prove
that a runtime contains a particular verified model; trusted composition remains
responsible for those bindings.

The immutable policy bounds each request, idle time, total controller lifetime,
turn count, and UTF-8 history bytes. Total lifetime begins when the controller is
created. A real daemon watchdog closes an idle session even when no caller is
active. Requests are single-flight and late responses are discarded. Backend
The retained conversation and its admission work are byte bounded and are never
logged or persisted by this controller. The backend may already have allocated
an arbitrary `BackendResponse.raw`; this controller neither retains that raw
payload in history nor claims to bound the backend's process memory.

The role/content messages are the existing `BackendClient` text-conversation
adapter contract. They are not a dataset parser or a hardcoded tool wrapper.

Close initiation is one-shot and invokes the owned lease exactly once; the lease
provides the actual process-family close serialization. Runtime cleanup normally
runs in a daemon thread so an uncooperative cleanup cannot indefinitely block the
caller or watchdog. If the host cannot start that cleanup thread, the controller
invokes the same finite-timeout lease close directly; this relies on the lease
honoring those timeouts and is not an OS hard deadline. `ChatSession.state`
reports cleanup still pending or a closed nonsecret failure code. It does not
claim to hard-kill a stuck request thread, process, kernel, or GPU allocation.
Explicit recovery remains possible only through the separately retained injected
lease; the session does not retry teardown.

Explicit `close()` remains a nonblocking one-shot initiator. Context-manager exit
waits on the controller's real condition for a fixed bounded interval, allowing a
one-shot process to deliver teardown before exiting. It reports unresolved cleanup
with a closed error when there is no active exception and never masks an active
exception or claims that the runtime was killed.
