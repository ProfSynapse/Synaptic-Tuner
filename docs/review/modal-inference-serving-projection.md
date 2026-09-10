# Modal inference serving projection

Status: implementation under local qualification, 2026-09-10. Engine-only;
no server, GPU, cloud allocation or EHR changes.

## Contract

The unreleased internal inference configuration now requires `serving`; there
is no old-body fallback or second schema/signature. The existing complete-body
configuration evidence and command binding commit to these fields automatically.
Parsing still does not authenticate configuration or approve an inference image.

| Configuration | Existing runtime/client projection |
| --- | --- |
| `serving.served_model_name` | Exact full-model/LoRA routing alias; not access authority |
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

Pending integrated tests and independent review. Tests use real signed launch,
mounted artifact/materialization and full/LoRA target paths with fake model
acquisition, process, readiness and HTTP effects. No live serving or inference
runtime-lock qualification is claimed.

## Remaining boundary

Separate inspected inference image/Python/dependency/source pins, executable
bootstrap with continuous deadline enforcement, authenticated remote access,
exact owned Sandbox lifecycle and live qualification remain unfinished. No new
operator command, local Docker launcher, downloader, cache framework, authority
system or publication step was added.
