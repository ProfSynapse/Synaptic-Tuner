# Verified local vLLM chat composition

`Evaluator.verified_vllm_chat.verified_vllm_chat` is an internal context manager
joining the verified-local runtime, existing evaluation client, and bounded
conversation controller. It accepts the existing startup spec rather than a
second model/startup configuration type, and rejects explicit-network sources.
Authenticated retrieval and any exact pinned-base preparation happen before
this context; this composition adds no downloader or provider calls.

The runtime proves model readiness and the exact requested alias. The caller's
first explicit `session.chat(...)` is the inference check; there is no hidden
prompt or second client. Settings bind the returned loopback endpoint and alias,
disable credential defaults, retries, ambient proxy/netrc behavior and redirects,
and bound the HTTP response bytes and generated token request.

The outer runtime context owns cleanup even when client/controller construction
fails. The controller supplies request, idle and total conversation deadlines.
Cleanup attempts serialize through the same runtime lease; unresolved ownership
is retained on the raised error as `cleanup_lease` for explicit recovery. User
exceptions and interrupts remain intact. Startup has its separate finite budget;
socket timeouts and process cleanup are not hard kernel/GPU deadlines.

This is a composition surface, not a public CLI, durable acceptance receipt,
running Modal endpoint, release claim, or proof of actual model quality. No
conversation text, model output, or credential is persisted by this module.

Integrated verification (2026-09-10): the clean CPython 3.12.9 / pytest 8.4.2
environment passed 181 tests in 1.44 seconds across the controller, composition,
HTTP policy, existing client header regression, runtime/owned-process layers,
and both inference materialization/preparation modules. Composition cases use
real verified full/LoRA targets, runtime projection, client and controller with
fake process/readiness/HTTP effects, including success, error, interruption and
request timeout. Transport tests separately use real requests preparation with
a fake HTTP adapter to verify serialized immutable messages and absence of
ambient proxy/netrc/auth behavior. No actual model was loaded or inferred.

Broader exploratory evaluator selection: 204 collected, 192 passed and 12
failed in 3.37 seconds with plugin autoload disabled. An independent immutable
`b9d33f776ac663da5ba6d6156382a063afd5b219` archive reproduced the same failures
in the two affected modules: 18 passed/12 failed, or 24 passed/6 failed with
`pytest_asyncio.plugin` explicitly enabled. The candidate also measured
24 passed/6 failed with that plugin. Six are runner-environment failures and
six are unchanged multi-step schema expectations; the affected tests, runner
and prompt definitions are byte-identical. This is not an all-evaluator green
claim, and no dataset-specific runtime workaround was introduced.
