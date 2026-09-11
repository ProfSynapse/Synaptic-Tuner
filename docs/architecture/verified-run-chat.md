# Chat with a verified training run

The engine provides an embedded workflow, `tuner.inference.run_chat.open_run_chat`,
that opens a verified SFT run through a consumer-selected runtime adapter. That
adapter owns authentication, retrieval and preparation on its execution machine,
followed by one bounded chat session. The consuming project
owns run selection, authentication, storage and its user interface. No new CLI,
run registry, database, publication step or authority loader is introduced.

## Consumer wiring

Use the consuming project's already-composed `APIHost.runs` and retained
`TrainingRunRef`. The `RunsAPI` wrapper is not authentication by itself: its
injected operations must authenticate the retained run and evidence. Do not
substitute a legacy `list-runs` record, arbitrary output folder or bucket URL.

For Linux/WSL local vLLM, the checked-in adapter is
`Evaluator.local_run_chat.LocalVLLMRunChatRuntime`. This is embedding code for a
host that already supplies the named values, not a standalone script to execute:

```python
from Evaluator.chat_session import ChatSessionPolicy
from Evaluator.local_run_chat import LocalVLLMRunChatRuntime
from tuner.inference.run_chat import open_run_chat

runtime = LocalVLLMRunChatRuntime(
    policy=ChatSessionPolicy(
        request_timeout_seconds=30,
        idle_timeout_seconds=120,
        absolute_lifetime_seconds=600,
        max_turns=20,
        max_history_bytes=65536,
    ),
    cwd=runtime_directory,
    environment={},
    destination=private_artifact_directory,
    preparer=pinned_model_preparer,
    startup_options={"port": 9137, "startup_timeout_s": 300.0},
    max_tokens=128,
    max_request_bytes=8192,
)
with open_run_chat(
    host.runs,
    retained_run,
    runtime=runtime,
) as opened:
    reply = opened.session.chat(user_prompt)
    # The consumer decides how to display/store reply.message and model identity.
    retained_model = opened.local_model  # Present for this local adapter.
```

Both directories are existing absolute `Path` values. The artifact destination
must additionally satisfy the materializer's private, canonical, link-free
directory checks; it creates an exclusive attempt beneath that directory.
Keep the destination in consuming-project private storage. `environment` is an
explicit dictionary of allowed runtime settings, never `dict(os.environ)`:
credential, proxy and import-injection names are rejected, and offline model
loading is forced. The Python runtime must already have compatible vLLM/GPU
dependencies installed. This helper does not install packages or provision GPUs.
Set `startup_options["python_executable"]` to select an exact canonical absolute
POSIX interpreter path; otherwise a startup spec captures the current interpreter.
The runtime uses that path directly without shell/PATH lookup or fallback. This
selects an interpreter; it does not authenticate its version, executable hash,
installed packages or image. Those checks remain required for a locked deployment.
Cheap configuration is snapshotted at adapter construction; full startup and
generation validation occurs on opening, before process creation, and can occur
after artifact retrieval. Preparation validates the target, and the runtime
performs a fresh file/inventory validation immediately before spawning. The
adapter itself adds no third validation scan, copy, materialization or download.

For a full model, `pinned_model_preparer` can be `None`. For LoRA, inject a
`PinnedModelPreparer` returning the exact authenticated upstream revision; the
serving target pairs that base with the retrieved adapter and tokenizer. Use
the existing reviewed model-preparation implementation on the execution machine,
including its SDK cache handling, instead of adding a downloader or making the
user stage weights manually. Credentials belong to preparation, not inference.
The local adapter never downloads a model as a fallback.

## Ownership, saved files and limits

Entering the context invokes the selected adapter once. The local adapter
performs verification and bounded artifact materialization before opening vLLM.
It sends no hidden prompt. A consumer explicitly
calls `opened.session.chat(...)` for each request; replies and conversation
history are not automatically written to disk.

`opened.run`, `opened.artifacts` and `opened.model` expose the run, canonical
five-artifact inventory and prepared model identity without requiring local
filesystem paths. `opened.local_model` is an optional, explicitly local-only
capability: this adapter returns its factory-issued retrieved model for access
to retained paths; a remote adapter must return `None`. Never serialize local
device/inode identities as proof of a remote filesystem.
Successfully materialized local files remain after exit or a later runtime
failure. Closing the session tears down its owned process family, not those
files and not unrelated servers. The consumer owns retention/deletion decisions.

Request, idle, absolute session lifetime, turn and history bounds use the existing
`ChatSession` controller. Startup has its separate timeout; the session lifetime
begins after readiness, not during retrieval/preparation. An error or context
exit triggers owned cleanup. Unresolved cleanup is reported rather than claimed
successful; retain an exposed cleanup lease when retrying that exact cleanup.
These local bounds are not a provider billing guarantee or protection against
machine shutdown, kernel failure or a killed controlling process.

The lower-level `verified_vllm_chat`, `start_vllm_runtime` and `ChatSession`
also accept `deadline`, an optional absolute value in the controlling process's
monotonic clock domain. This permits a bootstrap to preserve one deadline across
startup and conversation rather than granting a fresh lifetime after readiness.
The earliest external deadline and local limits win. Omission retains the local
adapter's existing separate-startup/session behavior. With an injected session
clock, the deadline must use that same clock domain. Never serialize this value
as portable evidence or treat it as provider termination authority; a remote
adapter must authenticate its UTC claim, conservatively derive local remaining
time, verify the runtime and separately enforce provider lifecycle controls.

The local adapter and `verified_vllm_chat` also enforce `max_request_bytes`
(1 MiB by default) for the complete serialized UTF-8 HTTP JSON body, including
model/generation fields and accumulated messages. This is separate from the
history bound and may reject a request before that bound is reached. Invalid or
oversized bodies are rejected before constructing the HTTP transport and are
not retried; the owned chat session still cleans up its runtime. The checked
body is sent as the same bytes, not re-serialized by the HTTP library. Headers,
request lines, TLS/socket buffers and backend memory are outside this byte cap.

## Adapter boundary and qualification

Another runtime implements `RunChatRuntime.open(runs, run)` as a context manager
yielding an exact `PreparedRunChat` containing the existing exact `ChatSession`.
It must perform current run reverification and artifact admission before serving
effects, bind any provider-native locator to that same proof, prepare on its
execution machine, own the runtime it starts, and honor bounded cleanup.
The generic helper checks inputs and returned metadata, not authentication:
the selected adapter is trusted consumer code, not an arbitrary plugin sandbox.
There is no registry, dynamic import,
silent local/cloud fallback or new public API re-export.

Correction (2026-09-10): the earlier target-first signature was machine-local;
it downloaded weights before invoking an adapter. This internal API now delegates
the run before materialization. Destination and upstream-preparer policy belong
to the local adapter constructor. There is no deprecated signature or wrapper.

CPU tests exercise real run retrieval, target preparation, controller and local
runtime/client composition with fake OS/HTTP effects. They do not establish
model quality, real GPU loading or a newly working Modal inference service.
The current Modal deployment is for training, not interactive inference. A Modal
chat adapter still needs exact-source deployment, remote artifact/model
preparation, authenticated access and provider-side lifetime/cost safeguards,
followed by separate live qualification. Do not keep a GPU endpoint alive merely
because a local client disappears.
