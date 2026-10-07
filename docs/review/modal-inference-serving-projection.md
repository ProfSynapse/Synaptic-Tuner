# Modal inference serving projection

Status: implemented and locally qualified, 2026-09-10. Engine-only;
no server, GPU, cloud allocation or EHR changes.

## Contract

The unreleased internal inference configuration now requires `serving`; there
is no old-body fallback or second schema/signature. The existing complete-body
configuration evidence and command binding commit to these fields automatically.
Parsing still does not authenticate configuration or approve an inference image.

| Configuration | Existing runtime/client projection |
| --- | --- |
| `serving.served_model_name` | Exact full-model/LoRA routing alias, at most 96 characters; not access authority |
| `serving.gpu_memory_utilization_milli` | GPU utilization divided by 1000 |
| `resources.accelerator_count` | Tensor parallel size, 1–256; no second independent knob |
| `serving.enforce_eager`, `tokenizer_mode`, `max_lora_rank` | Same explicit startup fields |
| `serving.readiness_request_timeout_milliseconds` | Probe seconds divided by 1000 |
| `serving.max_tokens`, `temperature_milli`, `top_p_milli` | Generation limit and thousandths-to-float values |
| `runtime.python_executable`, `resources.service_port` | Exact interpreter and loopback port |
| `policy.startup_timeout_seconds` | Startup maximum, capped at existing runtime limit of 1800 seconds |
| Remaining session/body policy fields | Existing configured session policy and HTTP request/response limits |

All startup fields are passed explicitly. Loopback remains exactly `127.0.0.1`;
retry/proxy/redirect and credential-free offline behavior stay in existing
components. Provider lifetime/idle configuration belongs to future Sandbox
creation, not the local session policy. Training `load_in_4bit` is not translated
into inference quantization. Integer thousandths preserve the existing
integer-only canonical format without floating-point evidence fields.

`prepare_modal_chat_worker` now returns an internal `ModalChatWorkerPreparation`
instead of a bare target. The target is `prepared.startup.source.target`.
The result preserves a byte-owned copy of original admission and explicit
generation/body values. After preparation it checks the same trusted UTC clock
against the original claim expiry and rechecks callback-sensitive inputs,
retained roots and the target. Expiry equality is rejection; control-flow
interruptions remain intact. Saved model files are not deleted on expiry.

This constructible frozen result is data, not a new authentication receipt or
runtime grant. `configured_policy` contains signed maxima, not remaining time.
A future locked bootstrap must freshly verify admission, rederive/compare the
projection, and clamp startup/session time to the original remaining deadline
immediately before and after startup. The preparation check alone cannot stop
time passing after return or guarantee provider billing limits.

## Qualification

Source commit: `f4589820d7f753c9c6388863ffa64ba8d496ba97`. Independent
source/configuration review passed after correcting the alias limit to the
existing LoRA runtime's 96-character maximum. Interpreter validation matches
the existing runtime's canonical POSIX text and UTF-8 bounds. Source SHA256:

| File | SHA256 |
| --- | --- |
| `inference_preparation.py` | `6ce129f043e87ae2bcb9f750c37a69d750ada55b8542e930709c0d0141119468` |
| `inference_worker.py` | `2b1147db7b222fc858d8b22e89c024edb217dc3604f6e507e14aeb7f76e4fcad` |

The selected lanes passed under CPython 3.12.9 / pytest 8.4.2, with plugin
autoload disabled, a clean lightweight test environment and temporary files on
`/dev/shm`. Execution used the standalone engine execution checkout with the
lead engine worktree selected explicitly; no operator `PYTHONPATH` was exported.

| Selection | Measured result |
| --- | --- |
| Existing local runtime, HTTP, verified/local chat and `tests/inference` lane | 330 passed in 1.62 s |
| Inference configuration/preparation module | 71 passed in 191.51 s |
| Mounted worker and full/LoRA integration modules | 44 passed in 146.99 s |
| Inference commands, effects, broker, launch and launch integration modules | 174 passed in 507.32 s |
| Added maximum accepted alias through real LoRA argv projection | 1 passed in 4.53 s |
| Added all serving numeric min/max/type and DEL/UTF-8 path bounds | 1 passed in 2.99 s |

These are 621 distinct collected tests, not whole-repository or live-provider
qualification. The two added boundary cases were collected separately after
the longer lanes started. Those lanes loaded config before a diagnostic-only
wording correction ("24 hours" to "its runtime bound"); validation logic and
worker source were unchanged. Final boundary cases used the corrected source.
The Sol worktree's earlier 43-case subset is not added to these totals.

Tests use real signed launch, mounted artifact/materialization and full/LoRA
target paths with fake model acquisition, process, readiness and HTTP effects.
They check every explicitly supplied startup field, actual generated argv and
HTTP generation bytes, no hidden prompt, exact cleanup and preservation of
model files. Final-clock expiry equality, expectation/admission/model mutation,
`KeyboardInterrupt` and `SystemExit` are covered. Per-serving-field commitment
tests confirm command/preparation changes without changing the resource-only
quote digest. The result remains non-authoritative prepared data.

An offline wheel build from the exact source archive produced
`synaptic_tuner-1.1.0-py3-none-any.whl`, 2,076,942 bytes, SHA256
`029de06b2aa458e7ceabe78e05788a676724b5241b656cd4225b8da741d14024`.
Independent audit verified 725 unique ZIP entries and RECORD rows, all declared
hashes/sizes, and all 717 Python members byte-equal to the source archive. A
filename-only scan found no private/test artifact paths; it was not a credential
or content scan.

Only this wheel was replaced in the existing isolated declared-dependencies-only
qualification environment, using no index or dependency resolution. The
checked-in neutral-directory probe passed 44 engine imports and both packaged
lock resources. Two strict signature binds (no hidden `**kwargs`) and the typed
worker return contract passed. The changed modules resolved inside the installed
environment; Modal, Torch, NumPy, pandas and pytest were absent. This reused
environment check is not a new dependency-resolution qualification.

Both unchanged training inventories checked CURRENT: 97 runtime-lock files and
66 offline worker members (679,487 payload bytes); packaged resources byte-match
source. Canonical skill mirrors are synchronized, and the scoped single-worker
format check passed. No EHR writes, cloud/server/GPU execution, push or merge
occurred. No live serving or inference runtime-lock qualification is claimed.

## Remaining boundary

Separate inspected inference image/Python/dependency/source pins, executable
bootstrap with continuous deadline enforcement, authenticated remote access,
exact owned Sandbox lifecycle and live qualification remain unfinished. No new
operator command, local Docker launcher, downloader, cache framework, authority
system or publication step was added.
