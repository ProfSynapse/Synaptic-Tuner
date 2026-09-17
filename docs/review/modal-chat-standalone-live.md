# Standalone Modal chat: first live attempt

One attempt of `examples/model_chat/modal_launch.py` ran on 2026-09-17 from the
consumer checkout with the launcher at engine commit `00ea563`, provider file
`modal-smoke-provider.json` (environment `synaptic-smoke-v1`, App
`synaptic-model-chat-v1`, A10, 4 cores, 16 GiB) and the unchanged
`examples/model_chat/smoke.json` (sha256 `de684e1f…` matches the checked-in
file). Evidence: `docs/review/evidence/modal-chat-standalone-live-20260917.json`.

What was observed, in order:

| Step | Value |
|------|-------|
| Sandbox | `sb-ZqwyONml2vQ3rP1nnr94kn`, created 13:40:43Z |
| Container exit | returncode 0; stderr empty |
| Command statuses | `CHAT_INPUTS_CHECKED`, `CHAT_SAVED_AND_CLOSED` |
| Records parsed | `CLAIMED`, `REPLY_SAVED`, `CHAT_CONTEXT_CLOSED` |
| Shutdown | `terminate` clean, `Sandbox.from_id(...).poll()` returned 0 |
| Stopped | 13:41:51Z (68 s end to end) |
| After | both Apps in `synaptic-smoke-v1` list 0 tasks |

The launcher exited 0 on its own three conditions: container exit 0, three
records parsed, integer poll readback. The reply text stays in the private
0600 `chat-result.jsonl` under `.synaptic/state/`; it is not copied here.

This closes the "live standalone Modal chat success" item that the README and
the Modal skill note previously listed as not claimed. It does not touch the
coupled train-then-chat path, adapter provenance, cost at issuance, or the
older `ea66c4c` inference-image capture, which no longer matches the closure
of the current engine pin and needs a fresh capture before any coupled attempt.

## Closeout (2026-09-17, operator decisions)

Cleanup executed after the live attempt, each item chosen by the operator:

- `synaptic-smoke-v1`: the 30 Volumes and 10 Secrets of the 2026-09-14
  coupled attempts a..j deleted by exact name (40/40); the 2026-09-07 training
  smoke App `synaptic-training-v1` stopped and its two Volumes and one Secret
  deleted. The environment now holds no Volumes and no Secrets; Apps are
  `synaptic-model-chat-v1` (deployed, 0 tasks, reused by the launcher) and
  `synaptic-training-v1` (stopped).
- Nothing in the Modal `main` environment was touched.
- Local Docker Desktop: 15 exited `synaptic-modal-submit:34d6623d` containers
  and the `synaptic-local-chat-results` volume removed;
  `synaptic-local-hf-cache` kept for the next local proof.
- The private attempt directory under the consumer checkout's
  `.synaptic/state/model-chat/` is retained as the only copy of the reply.
