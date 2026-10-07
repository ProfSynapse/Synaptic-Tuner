# GPU Cloud Execution Architecture

## Status

This document describes the submodule-first cloud architecture. The public
surface is the provider-neutral `TrainingAPI`; provider names are configuration
values, not public verbs. The engine contains reusable contracts and mechanisms.
The consuming host project owns configuration, data, credentials, grants,
durable state, persistence, and user-facing policy.

There is no compatibility layer for the removed cloud backend hierarchy or its
launcher aliases.

## Public flow

Every provider follows the same public sequence:

1. `load(document)` validates a canonical training request.
2. `resolve(request)` finalizes host-owned inputs and authenticated source and
   provider evidence.
3. `plan(resolved)` produces an immutable, fingerprinted training plan.
4. `preflight(plan)` performs bounded read-only checks and returns expiring
   evidence plus exact authorization requirements.
5. The host binds a grant to the exact operation.
6. `start(plan, preflight, grant)` durably records preparation before the one
   provider mutation.
7. `status`, `logs`, `artifacts`, and `outcome` reconcile authenticated provider
   evidence into host-owned lifecycle state.

Provider implementations sit behind this interface. They cannot select a host
database, read ambient credentials, invent storage locations, or weaken grant
binding.

## Ownership

| Concern | Engine/submodule | Consuming host project |
|---|---|---|
| Public contracts and schemas | Owns | Consumes |
| Recipe compilation and semantic artifact verification | Owns | Configures |
| Provider adapters and mutation broker | Owns | Composes explicit clients |
| Provider profile and runtime-lock schema | Owns | Selects checked-in config |
| Dataset and model inputs | Never owns | Owns |
| Credentials and named secret bindings | Never persists | Owns |
| Grants and spend policy | Defines binding contract | Issues and persists |
| Lifecycle/preparation persistence | Defines protocol only | Implements database |
| Logs, dashboards, retention, and product UX | Emits canonical evidence | Owns |

The engine has no SQLite implementation. A standalone engine checkout can
compile and verify mechanisms, but cloud execution requires host ports.

## Provider status

### Modal v1

Modal is the first clean provider slice. Its fixed deployment uses Modal SDK
1.5.4, an authenticated explicit client, one digest-pinned Unsloth image, one A10,
zero retries, a bounded timeout, two pre-existing Volume v1 mounts, and named
Modal Secrets. Raw secrets are forbidden from the image environment.

The host preflight binds the exact deployment attestation, account/workspace/
environment/client scope, Volume identities, quote digest and expiry, operation
identity, resource digest, and public plan fingerprint. The packaged runtime lock
binds the registry reference, CPython runtime, dependency closure, deployment
wrapper, SFT runtime, and ML stack.

`start` uses this order:

1. build the exact operation and grant binding;
2. atomically persist durable preparation in the host repository;
3. stage only missing byte-identical Volume members without overwrite;
4. durably claim the canonical mutation command;
5. call the fixed Modal Function with `.spawn()` exactly once;
6. persist the exact FunctionCall identity.

Restarting `start` reuses a confirmed durable submission and never spawns a
second job. Completion is accepted only when authenticated terminal, log, and
five-member completion evidence passes structural, cryptographic, workload, and
semantic artifact verification. A FunctionCall return value is auxiliary
reconciliation evidence, never success evidence.

### Hugging Face Jobs

The protected HF A10G smoke remains behind its existing
`hf-training-smoke {preflight,approve,execute,recover,observe,verify}` protocol.
It is not routed through legacy cloud aliases. Its exact source, quote,
authorization, isolated launcher, one-use bucket slot, diagnostic codes, and
artifact inventory constraints remain mandatory.

### RunPod

RunPod is not yet a completed provider slice. It must implement the same public
contracts, durable mutation boundary, authenticated evidence, and host ownership
rules. It must not reintroduce provider-specific public commands or engine-owned
persistence.

## Security and isolation invariants

- Credentials enter only through explicit host ports and provider-native named
  secret objects.
- Plans, preflights, grants, staged material, commands, provider identities, and
  completion evidence cross-bind by canonical SHA-256 digests.
- Source execution uses exact pushed commits and a verified host/submodule
  topology.
- Remote Git runs without global/system configuration, prompts, or inherited
  credential helpers.
- Mounted Linux I/O traverses through retained directory descriptors and opens
  leaves relative to them, defeating ancestor substitution; bounded reads also
  verify leaf identity, while writes are exclusive and reject collisions.
- Provider exceptions, logs, and returns are not automatically trusted or
  exposed.
- Transient evidence failures become `INCONCLUSIVE` and may legally retry;
  semantic mismatches become `INVALID` and fail closed.

## Release gates

No authenticated preflight or paid smoke runs until all provider-free contract,
host-composition, packaging, runtime-lock, hostile-path, lifecycle, and legacy
absence tests pass and an independent review returns no blocker. Live provider
proof then proceeds in increasing-cost order: read-only identity/readiness,
deployment preflight, one authorized smoke, authenticated outcome verification,
and host persistence/restart proof.
