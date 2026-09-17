# Model-first chat

Chat is a separate process from training. Select a pinned Hub model or an
existing model directory; optionally supply an existing LoRA directory. No
TrainingAPI, RunsAPI, training run, coordinator database, or training signing
key is required. A training-to-chat workflow may compose these operations later.

The configuration is declarative. `smoke.json` selects a small public model;
replace its values in a consuming project's own configuration. Paths refer to
the execution machine. Model download/cache handling uses the existing vLLM/Hub
path there, not an operator weight-staging step. An adapter directory is an
explicitly selected input, not a claim that its training provenance was verified.
Remote archive retrieval and adapter preparation are not implemented here yet.

Provider-free validation, from the engine checkout:

```bash
python -B scripts/chat_model.py --configuration examples/model_chat/smoke.json --check
```

On an execution machine with the compatible vLLM/GPU runtime installed, choose
an existing private output directory owned by the consumer (mode 0700):

```bash
python -B scripts/chat_model.py --configuration /consumer/chat.json --output-directory /consumer/private-chat-attempt
```

Run from a trusted execution directory: the serving child inherits that working
directory, not the transcript output directory. Configuration, model paths and
the execution directory are operator-controlled inputs, not a sandbox against
another process with the same operating-system identity.

The command runs one explicit prompt and appends its claim, reply and normal
context-close record to a new private `chat-result.jsonl`. It refuses to
overwrite an existing attempt. A reply is retained even if later cleanup fails;
a claim without a close record is not proof of success or permission to retry.
Raw model replies are saved privately, not printed. Credentials, proxy variables
and Python import overrides cannot enter the serving child environment, and
implicit Hub authentication is disabled. This initial path handles public Hub
models or prepared local models, not credentialed remote preparation.

The serving runtime binds loopback and uses the existing request, idle,
turn-count and lifetime controller. The configured lifetime is at most 900
seconds and covers startup plus conversation. Local process cleanup is not
provider shutdown proof: a Modal adapter must add a finite Sandbox timeout and
confirm termination of its exact instance. The standalone Modal adapter remains
to be wired; this command does not provision cloud resources.

The embedded entrypoint is `tuner.inference.model_chat.open_model_chat`, using
the existing `VLLMStartupSpec` and `ChatSessionPolicy`. Importing it neither
trains nor starts a runtime. Offline tests cover composition and failures with
simulated OS/backend effects; no live standalone Modal chat success is claimed.

## Local proof on the reviewed inference image

`scripts/chat_model_local_docker.sh <attempt-name>` runs the command above on a
local GPU through Docker, inside the exact image digest recorded as
`base_registry_reference` in
`tuner/execution/providers/modal/inference-runtime.lock.json`, read at launch.
It prints the docker command, the closed status codes, then the retained
`chat-result.jsonl`, and exits with the container's status. Environment
overrides (`DOCKER`, `SYNAPTIC_ENGINE_MOUNT`, `SYNAPTIC_CHAT_CONFIGURATION`,
volume names, `SYNAPTIC_CHAT_GPUS`, `SYNAPTIC_CHAT_SHM_SIZE`) are listed in the
launcher header, including the WSL2 + Docker Desktop form.

Prerequisites: a Docker client reaching a daemon with the NVIDIA container
runtime; a CUDA GPU with enough free memory for the selected model at the
runtime's 0.5 memory utilisation; `python3` on the host to read the lock. The
image and the pinned model are fetched on first use and cached afterwards.

Results live in the named volume `synaptic-local-chat-results` under
`/out/<attempt-name>`, created by the launcher with mode 0700; the Hub cache is
the named volume `synaptic-local-hf-cache`. Named volumes rather than bind
mounts because the command requires euid ownership with no group/other bits,
which a Windows-drive bind mount cannot provide. The engine is mounted
read-only at `/engine`. An attempt name that already exists is refused (exit 3)
and nothing is deleted; choose a new name per attempt.

Reading a success: stdout ends with `{"status":"CHAT_SAVED_AND_CLOSED",...}`
followed by the retained file's three records, `CLAIMED`, `REPLY_SAVED` and
`CHAT_CONTEXT_CLOSED`. Because the launcher prints that file, including the raw
reply, its stdout is not private; the command alone never prints replies.

Reading a failure: the command prints `CHAT_FAILED` (exit 1) or `INTERRUPTED`
(exit 130) and the launcher still prints whatever `chat-result.jsonl` exists.
`CLAIMED` alone means no reply was produced; `CLAIMED` and `REPLY_SAVED`
without `CHAT_CONTEXT_CLOSED` means cleanup failed after a reply. The process
lease discards the serving child's stdout/stderr, so vLLM's own logs are not
in the container output. To see them, start the vLLM server by hand in the
same image with the same model and revision; that is a separate diagnostic
step, not part of this command.

A local success shows that the checked-in command runs in the reviewed image
on this GPU with this configuration. It is not provider shutdown proof, does
not qualify the Modal path, and does not verify adapter provenance.

## Modal proof in a finite-lifetime Sandbox

`examples/model_chat/modal_launch.py` runs the same command once inside a
Modal GPU Sandbox built from the `base_registry_reference` digest in the
inference runtime lock, with `Evaluator`, `tuner`, `synaptic_tuner`,
`scripts/chat_model.py` and the chat configuration mounted under `/engine`
(never the whole checkout; `__pycache__`, `tests` and `Datasets` are excluded).
The Sandbox runs the container script of the local proof: `mkdir -m 700`,
`--check`, the chat run, then `cat chat-result.jsonl`. Nothing else is
mounted: no Secrets, no Volumes, no persistent endpoint.

Two inputs. `--configuration` is the unchanged six-key chat file above,
validated by `scripts/chat_model.load_configuration` and shipped as
`/engine/chat.json`. `--provider` is a small file such as
`modal-smoke-provider.json` (`environment_name`, `app_name`, `gpu`,
`cpu_millicores`, `memory_mb`, `startup_margin_seconds` 1..600). Provider-free
validation of both, including the lock digest, without importing Modal:

```bash
python -B examples/model_chat/modal_launch.py --configuration examples/model_chat/smoke.json --provider examples/model_chat/modal-smoke-provider.json --check
```

An attempt needs `--attempt <name>`, `--modal-profile <name>` and
`--output-directory <private dir>` (same checks as the command: absolute,
canonical, owned by the caller, mode 0700). Credentials are read from the named
profile only, never from the environment. The attempt name is the Sandbox name
inside the App; a name still in use is refused (`MODAL_CHAT_ATTEMPT_EXISTS`)
and an output directory that already holds `attempt.json` is refused before
any cloud call.

Lifetime chain: the command's own lifetime is at most 900 seconds; the Sandbox
`timeout` is that lifetime plus `startup_margin_seconds`, and `idle_timeout`
is set to the same value, so the provider stops the Sandbox even if the
launcher dies. After the entrypoint exits the launcher calls
`terminate(wait=True)` and then `Sandbox.from_id(<id>).poll()` on a fresh
handle; an integer from that readback is the only provider shutdown proof and
is recorded in `shutdown.json`. The launcher exits 0 only when the container
exit code is 0, the three records were parsed, and that proof is present.

Retained files in the output directory (all mode 0600): `attempt.json`
(written before waiting, with the Sandbox id), `sandbox-stdout.log`,
`sandbox-stderr.log` (each at most 1 MiB), `chat-result.jsonl` (the records
parsed after the `--- chat-result.jsonl` marker) and `shutdown.json`. Because
a finished Sandbox exposes only its log streams, the container prints the
result file, so the raw reply lands in Modal's log stream as well as in the
private file; the launcher's own stdout carries closed status lines only.

Worst case cost, from `docs/review/evidence/modal-chat-rates-20260914.json`
(read 2026-09-14, A10 at 1.10 USD/h): the smoke's 900 s lifetime plus 300 s
margin is 20 minutes, about 0.37 USD of GPU time; the Sandbox CPU and memory
rates add about 0.32 USD at 4 cores and 16 GiB, so about 0.69 USD in total.
Not a quote; rates must be read again at issuance.

Not claimed: adapter provenance (an `adapter_path` is an operator input),
any coupling to a training run, a persistent endpoint, or a live standalone
Modal chat success; the tests are provider-free with a fake SDK.
