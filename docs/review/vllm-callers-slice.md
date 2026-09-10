# vLLM caller regression slice

These tests pin the atomic caller migration from global process lifecycle
functions to `VLLMRuntimeLease`. Interactive evaluation owns and closes only a
runtime it starts; a pre-existing external endpoint is never stopped. Cleanup
is exercised on settings/client setup, evaluation failure, interruption, and
normal completion, while the requested LoRA/chat alias is retained.
Status probing and owned startup use the same explicitly configured loopback
host and port. Network runtime environment tests retain only the intended
PATH, library, CUDA/NVIDIA, Hugging Face cache, TLS, and proxy inputs while
excluding Python import injection and unrelated secrets.

Cloud regressions additionally require runtime cleanup before exact-loss work
and despite progress-sync shutdown failures. All collaborators are local fakes;
the tests perform no GPU, model, provider, or network operation.

Integrated verification (2026-09-10): CPython 3.12.9 / pytest 8.4.2 with
system-site evaluator dependencies passed 77 tests in 2.72 seconds across
this module, both existing cloud/setup modules, migrated startup cleanup,
the explicit runtime, and owned-process regressions. Separately, the clean
SDK-free environment passed 115 runtime/setup/cleanup/inference tests in
1.08 seconds. These are different selections, not a combined total. Both
runtime inventories remained CURRENT (97 Modal pins, 66 offline members).

The CI workflow has a dependency-light conformance lane and a separate
CPU-only evaluator-caller lane with real HTTP/PyTorch imports. CI has not
been dispatched; the system-site result is not a fresh CPU CI qualification.
The old startup cleanup review is retained explicitly as a historical record.
