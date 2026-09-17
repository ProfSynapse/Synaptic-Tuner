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
