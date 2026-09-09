# vLLM startup cleanup slice

`start_vllm_server` retains the exact process spawned by the current call and
tears it down when readiness fails, times out, raises, or is interrupted.
Cleanup terminates and waits for ten seconds, then kills and performs a bounded
reap on timeout or a failed terminate. The exact handle is cleared only after
confirmed exit; failed kill or reap remains retained for a later explicit stop
retry. Cleanup failure cannot replace the original startup result or exception.
A successful startup remains running for its caller to stop.

The module refuses a second managed start rather than replacing its process
handle. This prevents a failed invocation from stopping a process it did not
spawn; externally managed servers are never adopted or killed.

Provider-free tests cover 120- and 600-second startup budgets, errors,
KeyboardInterrupt, success, kill/reap fallback and failure, retry, concurrent
handle replacement, and existing-process isolation.
They do not run vLLM, allocate a GPU, download a model, implement chat, or add
an idle-session policy.

Integrated local verification (2026-09-09): the 15 focused tests plus three
existing `tests/cloud/test_vllm_setup.py` regressions passed together: 18 passed
in 1.02 seconds under CPython 3.12.9 / pytest 8.4.2. This temporary environment
uses system-site evaluator dependencies; it is not the isolated SDK-free
coordinator qualification environment. Independent review also passed the
same 18-test selection. No server or GPU was started.
