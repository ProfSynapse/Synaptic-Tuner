# Minimal consumer: train, retain evidence, chat once

This example belongs to the consuming-project layer, not the engine runtime.
It uses `APIHost.training`, `APIHost.runs`, and `open_run_chat` without changing
EHR, adding a provider-specific engine CLI, or staging model weights locally.

It is **not yet a standalone live Modal launcher**. The caller must compose the
real authenticated host and selected chat runtime. No test fixtures, invented
training outputs, or arbitrary model directories may replace those inputs.

## Saved state and ownership

The example owns its private SQLite file and immutable catalogs. A durable
attempt claim precedes the training or chat path. Reusing the same attempt is
refused, including after a crash or successful completion. An unknown outcome
is not permission to retry with another attempt identifier.

This is a single-process example. Its durable claim and saved records do not
implement restart recovery of the entire coordinator/Foundation state. Keep
the original process and composed stores alive through training, verification,
and chat. After interruption, inspect the exact retained provider identity and
evidence; do not infer shutdown or launch a replacement.

Keep the database under a consumer-owned `private/` directory outside committed
source. Never store credential values in configuration, argv, the database, or
captured exception text. The database is trusted consumer storage, not a hostile
shared-Volume security boundary. Credentials stay in the existing authority
ports and explicitly selected provider mechanisms.

## Required composition

Use `synaptic_tuner.api.v1.modal.compose_modal_coordinator` with the consumer's
configuration, authorities, stores, and explicit Modal client. Wrap its training
and runs operations in `APIHost`, using the same clock throughout. For Modal
chat, compose the existing `ModalRunChatRuntime` with its source/workload binders,
authenticated inference configuration and fresh quote, separate grants,
Foundation broker, SDK transport, and ready-lease handoff. The example supplies
neither fake authorities nor ambient provider fallback.

Training acceptance is not completion. Observe the exact returned run through
`RunsAPI`; successful verification and its authenticated five-artifact inventory
must precede chat. The selected adapter reverifies and prepares on its execution
machine. A chat turn uses one context-managed session and one explicit prompt.
Saved training artifacts are not deleted when the session ends.

Provider timeout, idle timeout, session deadline, and owned cleanup remain
mandatory. A local journal or context exit is not a hard billing guarantee.
Use a current quote and bounded resources within the operator's cumulative
budget; changing an attempt identifier does not renew budget or authority.

## Remaining live prerequisites

The current Modal source finalizer requires host/superproject provenance: a
clean, pushed consumer commit containing an exact engine gitlink, and evidence
binding both repositories. A folder inside a standalone engine checkout does
not satisfy this. Copy the example into a real consumer; do not pretend the
engine is its own host or disable source verification.

Live qualification also needs the final inference image with its reviewed
runtime/dependency/source resources, actual model artifacts, authenticated
configuration and grants, and confirmed exact-Sandbox termination. The previous
engine-installed CPU inspection is package evidence only, not proof of a
successful model response. No push, new remote repository, deployment, or paid
job happens merely by importing this example.

## Embedding the example

These are calls inside an already-composed consumer, not shell commands or a
complete host factory. `private_directory` is an existing canonical private
POSIX directory; this example targets Linux/WSL, not native Windows SQLite
locking. Open the storage explicitly and retain it for the consumer lifetime:

```python
from examples.modal_chat.storage import ModalChatStorage
from examples.modal_chat.consumer import submit_training_once, chat_once

with ModalChatStorage(private_directory / "smoke.sqlite3", "my-consumer") as storage:
    submitted = submit_training_once(
        host.training,
        storage,
        attempt_ref=training_attempt_ref,
        request_json=canonical_training_request,
        provider=selected_provider,
    )
    # Continue the host's RunsAPI observation/verification workflow here.
    # Do not call chat merely because submitted.start.accepted is true.
    # After that exact run is verified, and while this host remains composed:
    turn = chat_once(
        host.runs,
        storage,
        attempt_ref=chat_attempt_ref,
        run=submitted.start.run,
        runtime=authenticated_chat_runtime,
        prompt="Say hello in one short sentence.",
    )
```

The host supplies every named input above. The request digest is retained,
not raw training configuration. Chat response text is private consumer output;
do not put credentials into prompts. The helpers do not print raw provider
exceptions or automatically publish the model or conversation.

`chat-results` records that a response was received; it is saved before cleanup.
`chat-contexts` is written only after normal context exit and explicitly records
`provider_shutdown_proof: false`. A saved reply without a context record means
cleanup was not successfully recorded, not that the model call should be retried.
The original runtime retains any known cleanup lease while its process lives.

## Verification status

The integrated example has 22 passing provider-free tests, including actual
`open_run_chat` and `ChatSession` composition with a simulated backend, permanent
claims across reopening, cleanup failures, and oversized ASCII/Unicode replies.
Independent consumer/storage review passed. Separately, 27 existing run-chat
tests and 114 inference capture/inspection/preparation tests passed. These tests
do not establish a real GPU response or qualify the new image preparation.
