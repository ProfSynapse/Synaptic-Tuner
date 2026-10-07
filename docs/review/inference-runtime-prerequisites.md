# Inference runtime prerequisites

Status: implemented and locally qualified, 2026-09-10.
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

Qualified source commit: `3f903e6d463bfd1c9a4b645989f03243fae51546`.
Independent source review passed for both prerequisites and their propagation
through verified/local run chat. The final exact-commit regression passed
330 tests in 1.92 seconds under CPython 3.12.9 / pytest 8.4.2, with plugin
autoload disabled and no Modal or Torch installed. The selection covers both
new modules, HTTP transport policy, verified/local chat and full/LoRA integration,
vLLM runtime/startup cleanup, owned processes, chat sessions, evaluator clients,
cloud vLLM setup and `tests/inference`.

The 15 interpreter cases and 38 HTTP request cases are included in that total.
They cover pre-effect path/bound rejection, complete aggregate/escaped UTF-8
body limits, malformed and cyclic inputs, bounded snapshot serialization,
caller mutation, exact bytes across retries and bodyless reads. Full/LoRA
integration uses actual materialization, serving targets, runtime argument
construction and backend serialization with fake process/readiness/HTTP effects.

A separate overlapping caller lane passed 77 tests in 3.76 seconds: evaluator
vLLM callers, cloud HF vLLM callers, cloud setup, runtime, startup cleanup and
owned processes. It used CPython 3.12.9 / pytest 8.4.2 with existing system-site
evaluator dependencies, disabled plugin autoload and `CUDA_VISIBLE_DEVICES=""`;
this was not a fresh CPU CI environment. Do not sum the overlapping lanes or
treat them as whole-repository, real vLLM startup, model inference or provider
qualification.

An offline wheel build from the exact source-commit archive produced
`synaptic_tuner-1.1.0-py3-none-any.whl`, 2,075,084 bytes, SHA256
`085a44bb1debd31a5af398e828846b794980d8a7d1343d36ad6f45a4387d0e0c`.
Independent audit verified 725 unique ZIP entries, 725 RECORD rows with matching
hashes/sizes, and all 717 Python members byte-equal to the source archive.
The filename-only scan found no private/test artifacts; it was not a credential
or content scan.

A fresh installed-wheel environment, using only its declared dependencies,
passed the checked-in neutral-directory probe: 44 engine module imports and
both packaged lock resources. Four strict signature binds (no `**kwargs`)
covered the startup spec, VLLM client, verified chat and local runtime.
Additional pure probes confirmed exact JSON byte-bound acceptance/rejection
and the selected interpreter at `argv[0]`. The four changed runtime modules
resolved inside the installed environment; Modal, Torch, pytest, NumPy and
pandas were absent. These checks made no server, provider or GPU calls.

Audited source SHA256 values:

| Module | SHA256 |
| --- | --- |
| `Evaluator/base_client.py` | `1810e34b99fc05ac6fd70b398779868cb9d64438a1514f920fac6b5a06e5d581` |
| `Evaluator/vllm_runtime.py` | `dc8745fd6ef451af451a7e818c352d2580a3327914cd3d1fd72fd6b29e1216e2` |
| `Evaluator/verified_vllm_chat.py` | `7c57205eee3caf4141d5ba376ac06512436a2860ccf5cb1c559f926564dca1a0` |
| `Evaluator/local_run_chat.py` | `5c97660107d5b501898980eedabe4147bd43532ecdc4f0b5b7afc97b16468cfb` |

Both unchanged training inventories checked CURRENT: 97 runtime-lock files and
66 offline worker members (679,487 payload bytes). Their packaged resources
byte-match the source archive. Canonical skill changes were synchronized and
the mirror check passed. No EHR changes, cloud actions, push or merge occurred.

## Remaining work

Bind admitted Modal serving configuration to these runtime/client fields,
complete the worker/bootstrap, and inspect the separate inference image,
Python/dependency and source lock. Authenticated remote access, exact Sandbox
ownership and bounded cleanup remain unfinished. A local process/session timer
is not a provider billing guarantee.
