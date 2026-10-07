# Standalone Modal chat: preparation findings

Status: research record, 2026-09-17. Read-only; no Modal call, cloud object,
credential or install was used. SDK facts come from the installed Modal 1.5.4
package (`site-packages/modal`, CPython 3.12.9) and modal.com/docs. Line
numbers refer to this checkout. Anything not measured is marked UNVERIFIED.

## 1. Goal restated

Run the proven standalone command `scripts/chat_model.py` inside a finite
Modal GPU Sandbox on the reviewed vLLM image digest, retrieve
`chat-result.jsonl`, and confirm that exact Sandbox stopped. No training run,
coordinator, Foundation or signed admission. Compare `examples/model_chat/README.md`
(lines 60-66: "a Modal adapter must add a finite Sandbox timeout and confirm
termination of its exact instance").

## 2. Reusable engine pieces (question 1)

| Need | Location | Standalone? | Import surface |
|---|---|---|---|
| Profile -> client, no env fallback | `examples/modal_chat/launch.py:148-161` `_client(sdk, profile)`; same inline at `scripts/capture_modal_inference_runtime.py:1195-1212` | yes (needs only `safe_ref` + `modal.config`) | `modal.config.config.get(key, profile=, use_env=False)` then `sdk.Client.from_credentials` |
| Create a Sandbox | `scripts/capture_modal_inference_runtime.py:963-996` (CPU, `block_network=True`, `timeout`+`idle_timeout`) | yes | `sdk.Sandbox.create(*cmd, app=, image=, cpu=, memory=, timeout=, idle_timeout=, client=)` |
| Create a GPU chat Sandbox | `tuner/execution/providers/modal/inference_transport.py:614-630` | no: needs Foundation, signed launch, Volumes, `Image.from_id` | do not reuse |
| Image from registry digest + local files | `scripts/capture_modal_inference_runtime.py:889-892` (`from_registry(image).entrypoint([]).add_local_file(..., copy=True)`); `coordinator_deployment.py:129-132` (`add_local_python_source(..., copy=False)`) | first yes, second training-coupled | `sdk.Image.from_registry`, `.entrypoint([])`, `.add_local_dir/.add_local_file` |
| Read results back | `scripts/capture_modal_inference_runtime.py:481-495,529-575` reads stdout/stderr of a stopped Sandbox via `Sandbox.from_id` | yes | `sandbox.stdout.read()` |
| Read a Volume file | `tuner/execution/providers/modal/facade.py` `read_complete` | no: bound to `ModalClientBinding` + volume-id catalogue | avoid for MVP |
| Terminate + confirm | `scripts/capture_modal_inference_runtime.py:444-470` (`terminate(wait=False)` then `poll()` loop); `examples/modal_chat/launch.py:294-347` (`poll()` int == `provider_shutdown_proof`); `inference_channel_client.py:381-400` (`terminate(wait=True)` must return int) | first two yes | `sandbox.terminate`, `sandbox.poll`, `Sandbox.from_id` |
| Closed diagnostics | `examples/modal_chat/diagnostics.py:145-159` | partly: fixed phase set at lines 10-29 and fixed admitted paths; depends on `foundation_v2.canonical` only | copy the pattern, do not extend the phase set |
| Bounded call with late-cleanup ownership | `scripts/capture_modal_inference_runtime.py:266-420` (`_PendingOperation`, `_OperationOwnership`, `_bounded_call`) | yes, but file-private (`_` names) | lift into a small module if reused |
| Image verifier | `tuner/execution/providers/modal/inference_runtime.py:449-494` `verify_modal_inference_runtime` | only called inside the packaged worker (`inference_bootstrap.py:242,279`) | not applicable, see section 3 |

No engine code reads a Sandbox's filesystem or mounts a directory into a
Sandbox with `copy=False`; both are new surface.

## 3. Packaged wheel versus mounted source (owned_process fix)

- Lock state after commit 9dfe346: `scripts/regenerate_modal_inference_lock.py`
  reports `{"status":"CURRENT","member_count":118}`; `Evaluator/owned_process.py`
  sha256 `f4cbb4e1...` matches the inventory. No further refresh is needed.
- The 118-member closure (`scripts/regenerate_modal_inference_lock.py:24-124`)
  does NOT contain `tuner/inference/model_chat.py` or `scripts/chat_model.py`.
  The packaged inference image (engine wheel 1.1.0 installed under
  `/opt/synaptic-inference`, `inference-runtime.lock.json` `python.executable`)
  therefore cannot run the standalone command at all, and any image built
  before 9dfe346 carries the old `owned_process.py`.
- The standalone path needs only the base image plus mounted source. The local
  proof already does this: `scripts/chat_model_local_docker.sh:106-112` mounts
  the checkout read-only at `/engine` and runs `python3 -B /engine/scripts/chat_model.py`
  in `base_registry_reference` (`docker.io/vllm/vllm-openai@sha256:116aa00e...`).
  Third-party imports of that path are stdlib plus `requests`
  (`Evaluator/base_client.py:15`, `Evaluator/openai_compat_client.py:13`);
  the base image has `requests 2.32.5`, `vllm 0.17.1`, `torch 2.10.0+cu129`,
  no `peft` (lock `distributions`; vLLM loads LoRA natively). Interpreter
  packages live in `/usr/local/lib/python3.12/dist-packages` (evidence
  `modal-inference-inventory-eefc6a6.json`).
- Consequence: `verify_modal_inference_runtime` is irrelevant here. The
  standalone adapter's image commitment is the digest string from the lock
  plus the mounted source bytes. Mounting source also means the owned_process
  fix ships automatically with every launch.

## 4. Modal SDK 1.5.4 Sandbox facts (question 2)

Source: `modal/sandbox.py` unless noted.

- `Sandbox.create(*args, app=, name=, tags=, image=, env=, secrets=,
  timeout=300, idle_timeout=None, workdir=, gpu=, cpu=, memory=,
  block_network=False, outbound_cidr_allowlist=, outbound_domain_allowlist=,
  volumes=, pty=False, ..., client=)` lines 549-585. `environment_name` and
  `cidr_allowlist` are deprecated (lines 583-585, 650-664); the environment
  comes from the App. `app` is required outside a container (lines 356-362);
  use `App.lookup(name, client=, environment_name=, create_if_missing=)`
  (`modal/app.py:309-315`). `name` is unique within an App and raises
  `AlreadyExistsError` (lines 621, 640): a free one-shot attempt claim.
- `timeout`: "Maximum lifetime of the sandbox in seconds" (line 568); wired
  as `timeout_secs` (line 501). Docs: "default maximum lifetime of 5 minutes.
  You can change this by passing a `timeout` of up to 24 hours"
  (https://modal.com/docs/guide/sandboxes). Expiry yields status TIMEOUT.
- `idle_timeout`: docs define activity as a running `exec`, stdin writes, or an
  open Tunnel TCP connection (same page). Whether an entrypoint-only Sandbox
  with none of those counts as idle from t=0 is UNVERIFIED; the capture script
  sets `idle_timeout == timeout` (`capture_modal_inference_runtime.py:987-988`).
- Exit codes: `_result_returncode` lines 207-214: `None` while running,
  `124` on TIMEOUT, `137` on TERMINATED, else the process exit code.
  `returncode` property line 2488.
- `poll()` lines 1895-1914: one `SandboxWait(timeout=0)`; returns the code or
  `None`. `wait(raise_on_termination=True)` lines 1694-1720: loops 10 s waits;
  raises `SandboxTimeoutError` on TIMEOUT and `SandboxTerminatedError` on
  TERMINATED unless `raise_on_termination=False`.
- `terminate(wait=False)` lines 1868-1893: "no-op if the Sandbox has already
  finished running"; `wait=True` returns the exit code (int).
- `Sandbox.from_id(id, client=)` lines 1299-1328: issues `SandboxWait(timeout=0)`
  and returns a hydrated handle; `from_id(...).poll()` is the exact-instance
  stopped proof (`launch.py:319-330` already relies on it). This works from any
  process that holds the id, so the readback survives a launcher crash.
- `Sandbox.list(app_id=, tags=, client=)` lines 2497-2545 sends
  `include_finished=False`: it lists only still-running Sandboxes, a cheap
  "nothing left running in this App" check after the attempt.
- Client disconnect: the Sandbox is a server-side object; docs list the only
  stop reasons as entrypoint exit, explicit terminate, timeout/idle timeout, or
  OOM (same page, "Lifecycle"). So a dead laptop leaves the Sandbox running
  until `timeout`. No official sentence says "billing stops at termination";
  the pricing page says "You never pay for idle resources — just actual
  compute time" (https://modal.com/pricing). Treat billing-stop as inferred.
- stdout/stderr: `sandbox.stdout` is a server-log stream (`modal/io_streams.py:56-106`);
  `read()` drains to EOF and is readable after the Sandbox stopped (proven by
  `read_modal_inference_runtime_sandbox`, evidence `sb-Sj9OVAZ63CtcLlZ9SnPbuO`).
- `exec(*args, stdout=, stderr=, timeout=, workdir=, env=, secrets=, text=)`
  lines 1991-2060 returns a `ContainerProcess` with `stdout/stderr/returncode/wait/poll`
  (lines 2648-2745). It requires a running task (`_get_task_id(raise_if_task_complete=True)`
  line 2092).
- Filesystem: `sandbox.open()` (deprecated 2026-03-09, line 2348) and
  `sandbox.ls()` (deprecated 2026-04-15, line 2379) are superseded by
  `sandbox.filesystem` (line 2314; `modal/sandbox_fs.py`: `read_bytes` 275,
  `read_text` 317, `copy_to_local` 119, `list_files` 183, `make_directory` 238,
  `stat` 405). Every one of them is implemented as an `exec` of a helper inside
  the Sandbox, so none can read a file out of a Sandbox whose entrypoint has
  exited. Reading `chat-result.jsonl` therefore needs one of: (a) the
  entrypoint prints the file to stdout before exiting (what
  `chat_model_local_docker.sh:96-99` does), (b) a long-lived entrypoint plus
  `exec` for the chat run and `filesystem.read_bytes` before `terminate`, or
  (c) writing the result to a mounted Volume. See section 8.
- `snapshot_filesystem(timeout=55, ttl=)` lines 1495-1531 returns an Image;
  not needed.
- Image: `Image.from_registry(tag, ...)` (`modal/_image.py:2200-2245`; "the
  image is expected to have Python on PATH as `python`"; the reviewed base
  has already been used this way in the qualified captures).
  `add_local_dir(local_path, remote_path, *, copy=False, ignore=[])` and
  `add_local_file(local_path, remote_path, *, copy=False)` (`_image.py:864-871`,
  `image.pyi:91-125`): `copy=False` mounts at container start and must be the
  last build steps. V1 `Sandbox.create` passes `image._mount_layers` as
  `mount_ids` (`sandbox.py:499`), so `copy=False` works for the main Sandbox;
  only `mount_image()` and sidecars reject it (lines 1581-1587, 2863-2870).
  GPU Sandboxes always take the V1 path (line 534: V2 only when `gpu is None`).
  Read-only-ness of the mount is UNVERIFIED; the command only needs `-B` and
  `sys.path` insertion (`scripts/chat_model.py:20-21`).
- Python version: nothing in this path is serialized. Modal's metadata
  requires Python `>=3.10,<3.15`; the installed 3.12.9 launcher is fine. The
  CPython 3.11.14 rule (`examples/modal_chat/README.md:14-17`,
  `launch.py:164-176`) exists only because `coordinator_deployment.py:134-139`
  deploys a `serialized=True` function. Do not reuse `_check_launcher_python`.

## 5. Model weights and adapters (question 3)

- Smoke model: `HuggingFaceTB/SmolLM2-135M-Instruct` at
  `12fd25f77366fa6b3b4b768ec3050bf629380bac` is `private: false, gated: false`
  (Hub API, read 2026-09-17). No HF token; `chat_model.py:178` already sets
  `HF_HUB_DISABLE_IMPLICIT_TOKEN=1`. Download happens inside the Sandbox, so
  `block_network` must stay `False`. An `outbound_domain_allowlist` for
  `huggingface.co` plus the xet CDN hosts is possible but the exact host set
  for `hf-xet 1.3.2` is UNVERIFIED; leave unrestricted for the MVP.
- Hub cache: optionally mount a Volume at `/root/.cache/huggingface`
  (`volumes={...}`), mirroring the local named volume
  (`chat_model_local_docker.sh:109`). Not required for correctness.
- Trained LoRA: the SFT runtime writes `run_dir/final_model`
  (`Trainers/sft/runtime_v1.py:1101`) and packages it as the `final_model`
  artifact `final_model.tar` (lines 1456-1458); a LoRA is recognised by
  `adapter_config.json` (lines 59, 1726). The training worker mounts its
  artifact Volume at `/workspace/run` (`deployment_v1.py:20`). The exact
  per-run directory layout inside that Volume and whether `final_model` is
  stored extracted or only as a tar are UNVERIFIED in this pass; the adapter
  directory must be an existing absolute, canonical (non-symlink) directory
  (`chat_model.py:104-110`, `model_chat.py:97-104`). vLLM is started with
  `--enable-lora --max-lora-rank 64 --lora-modules selected-model=<path>`
  (`Evaluator/vllm_runtime.py:92,390-397`); the smoke trains rank 8.
  Recommended contract: `adapter_path` names a directory on a Volume mounted
  read-only at a fixed path such as `/adapters`; extraction, if needed, is a
  later step.

## 6. Private output directory (question 4)

`scripts/chat_model.py:139-151` requires: absolute, `resolve()==self`,
directory, `st_uid == geteuid()`, no group/other bits, then `O_EXCL` creation
of `chat-result.jsonl`. Sandboxes run as root, and the qualified capture's
`--check-private-directories` measured "effective-user ownership, exact 0700"
at fixed paths in this very image (`docs/review/modal-inference-image-qualification.md:6-10`).
A `mkdir -m 700 /root/chat-attempt` executed by the same root entrypoint
satisfies the checks. Use the Sandbox's own filesystem, not a Volume (Volume
uid/mode semantics UNVERIFIED; the local launcher notes bind-mount failures
for the same reason, `chat_model_local_docker.sh:33-36`). The configuration
file is read with `O_NOFOLLOW` and must be a regular file <= 64 KiB
(`chat_model.py:38-43`); a `copy=False` `add_local_file` mount worked for the
capture script's inputs, so mount the JSON the same way.

## 7. Cost and lifetime controls (question 5)

Chain, innermost first:
1. `lifetime_seconds <= 900` (`chat_model.py:95-99`); startup timeout is
   `min(600, lifetime)` (line 118); `ChatSessionPolicy(60, 120, lifetime, 1, 16384)`
   (line 175) caps request, idle, turns.
2. Sandbox `timeout = lifetime_seconds + margin` (e.g. 1020 s) as the
   server-side hard cap; `idle_timeout` equal to it. Exit code 124 identifies
   an expiry (`sandbox.py:210-211`).
3. After the entrypoint exits: `terminate(wait=True)` (no-op if finished,
   returns int), then a fresh `Sandbox.from_id(id, client).poll()` from the
   recorded id; an int is the exact-instance stopped proof
   (`launch.py:319-330` pattern). Optionally `Sandbox.list(app_id=...)` must
   not yield that id.
4. Persist the Sandbox id locally before anything else so a crashed launcher
   can rerun step 3 alone (`read_modal_inference_runtime_sandbox` shows the
   shape, `capture_modal_inference_runtime.py:529-575`).

Laptop death: step 2 still terminates the Sandbox (docs "Lifecycle"). Rates
(`docs/review/evidence/modal-chat-rates-20260914.json`, read 2026-09-14):
A10G 1.10 USD/h, Sandbox CPU 0.1419 USD/core-h, Sandbox memory 0.024 USD/GiB-h;
pricing page lists A10 at 0.000306 USD/s. A 900 s worst case at 4 cores /
16 GiB is about 0.51 USD; at 2 cores / 8 GiB about 0.39 USD. Not a quote.

## 8. Recommended architecture (question 6)

Separate small launcher, engine-side: `examples/model_chat/modal_launch.py`
with its own tiny JSON, not a new `--mode` of `examples/modal_chat/launch.py`.
Reasons: that launcher hard-requires CPython 3.11.14 (`launch.py:164-176`),
a project manifest, consumer storage and Foundation composition
(`check_inputs`, `execute` at lines 180-600), and a closed phase set
(`diagnostics.py:10-29`) that would all have to be relaxed.

Flow (one process, sync SDK, nothing serialized):
1. Parse launcher JSON; run `chat_model.py --check` locally on the chat JSON.
2. `_client(sdk, profile)`; `App.lookup(app_name, environment_name=,
   create_if_missing=<explicit flag>, client=)`.
3. `Image.from_registry(<lock base_registry_reference>).entrypoint([])`
   `.add_local_dir(<engine>/Evaluator, "/engine/Evaluator", copy=False)`, same
   for `tuner` and `synaptic_tuner` (`tuner/__init__.py:27` imports
   `synaptic_tuner._version`), `.add_local_file(scripts/chat_model.py,
   "/engine/scripts/chat_model.py")`, `.add_local_file(<chat.json>,
   "/engine/chat.json")`. Use `ignore` to exclude `__pycache__` and tests.
4. `Sandbox.create("bash", "-lc", <script>, app=, image=, gpu="A10",
   cpu=, memory=, timeout=, idle_timeout=, name=<attempt>, workdir="/root",
   volumes=<optional adapter/cache>, client=)` where `<script>` is the
   container script of `chat_model_local_docker.sh:85-103` (mkdir -m 700,
   `--check`, run, print status, `cat chat-result.jsonl`). Record the id.
5. `sandbox.wait(raise_on_termination=False)`, read `stdout`, parse the
   closed status line and the three JSONL records, save them to the local
   private directory (mode 0700, `launch.py:120-146` `_private_directory`).
6. `terminate(wait=True)`; `Sandbox.from_id(...).poll()`; write the cleanup
   record with `provider_shutdown_proof`.

Privacy trade-off: option (a) puts the raw reply into Modal's log stream
(`README.md:60-63` notes the command alone never prints replies). If that is
unacceptable, option (c): mount a results Volume and write there, then read
with `Volume.read_file` (`facade.py:112-118` shows the call shape); this adds
a Volume and the 0700 check on a Volume is UNVERIFIED. Option (b) keeps the
file private but adds a second process and idle-timeout interplay. Recommend
(a) for the smoke and note it in the result record.

Consumer must supply: Modal profile name; environment name; app name (and
whether creation is allowed); GPU string; the six-key chat JSON; local output
directory; optional Volume name + mount path for an adapter or Hub cache.
From `configuration/smoke.json` `inference` block only `resources.accelerator`,
`resources.cpu_millicores`, `resources.memory_mb`,
`resources.provider_timeout_seconds` and `policy.absolute_lifetime_seconds`
map onto this path; `serving.*` values are fixed inside `chat_model.py`
(`gpu_memory_utilization=0.5`, temperature 0, top_p 1, line 117 and
`model_chat.py:115-117`).

## 9. Provider-free testing (question 7)

- Fake SDK as a `SimpleNamespace(__version__="1.5.4", App=, Image=, Sandbox=)`
  with recording `lookup/from_registry/create` and a `Sandbox` exposing
  `object_id`, `stdout`, `returncode`, `wait`, `terminate`, `poll`:
  `tests/execution/providers/test_modal_inference_runtime_capture.py:304-360`.
- Injecting a fake `modal` module for `main()` without importing the real
  SDK: same file lines 1056-1083 (`_fake_modal`).
- Profile read isolation: `tests/examples/test_modal_chat_launch.py:224-249`
  patches `modal.config.config.get` and asserts `use_env=False`.
- Cleanup readback fakes: `test_modal_chat_launch.py:313-320` (`poll` raises or
  returns int); `test_modal_inference_channel_client.py:93-118` (`terminate`
  gate). The command itself is tested by patching `open_model_chat`
  (`tests/scripts/test_chat_model.py:35-60`).
- Design the adapter as `run(*, sdk, client, ...)` like
  `capture_modal_inference_runtime(sdk=, client=, ...)` (line 820) so tests
  never import Modal, and keep the `pytest.importorskip("modal")` guard for
  the one profile test.

## 10. Corrections to the brief

- The packaged engine wheel cannot serve this path: the closure omits
  `tuner/inference/model_chat.py` and `scripts/chat_model.py`.
- `sandbox.open()`/`sandbox.ls()` are deprecated in 1.5.4 and, like the new
  `filesystem` API, need a running Sandbox; a finished Sandbox exposes only
  stdout/stderr logs.
- `cidr_allowlist` is deprecated; the parameter is `outbound_cidr_allowlist`.
- The rate file keys the GPU as `gpu_hour_cost_a10g`, matching the pricing
  page's A10 line.
