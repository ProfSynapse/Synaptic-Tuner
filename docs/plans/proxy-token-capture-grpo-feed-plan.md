# Proxy Token Capture → GRPO Feed Plan

**Date:** 2026-05-29
**Status:** Implemented (capture path); consumer-side GRPO replay is future work
**Paper / Prior Art:** NVIDIA POLAR — [arXiv:2605.24220](https://arxiv.org/html/2605.24220v1) ([MarkTechPost writeup](https://www.marktechpost.com/2026/05/27/nvidia-releases-polar-a-token-faithful-rollout-framework-for-grpo-training-across-codex-claude-code-and-qwen-code/))
**Branch:** `claude/polar-token-rollout-MTzCp`

---

## Findings from the codebase (revise the plan)

Investigating the real flywheel code changed two plan assumptions:

- **No catalog migration needed.** `catalog.py` only *indexes* lightweight fields
  (`log_id`, `model_id`, `has_tool_calls`, score, tag, `source_file`/`line_number`,
  …) — it never stored `messages` or `response_content`. Full content (and now
  token ids/logprobs) lives in the date-partitioned JSONL; the stager reads it
  back via `read_log_content()` using `source_file`/`line_number`. So token
  capture is purely additive JSONL fields — Phase 1's catalog work is dropped.
- **A GRPO staging path already exists.** `stager._write_grpo` /
  `_format_grpo_example` already emit `{conversations, reward}`. Rather than a
  parallel rollout file, token fields are passed through that existing example
  when present — additive and back-compatible.

## Phase 0 — vLLM capture contract (resolved by reading vLLM behavior)

- **Logprobs:** request `logprobs: true`; vLLM returns
  `choices[].logprobs.content[]` with per-token `{token, logprob, bytes}`. We read
  `logprob` directly.
- **Token ids:** the OpenAI schema has no token-id field. vLLM's
  `return_tokens_as_token_ids: true` renders each `token` as `"token_id:<int>"`,
  which we parse. If any token isn't in that form, token ids are reported `None`
  (logprobs may still be captured).
- **Prompt token ids:** not reliably returned by chat-completions; left `None`
  for now (a downstream consumer can re-tokenize the logged messages, accepting
  drift — documented as future work, design fallback B).
- **Caveat:** exact `return_tokens_as_token_ids` behavior is vLLM-version
  dependent and must be confirmed against the running server (like Plan #1's GPU
  smoke). The capture is best-effort and never fails the request.

---

## Goal

Teach the flywheel logging proxy to capture **token-faithful rollout data** — the actual completion token
IDs and per-token logprobs — so logged production inference can eventually feed GRPO, not just the current
SFT/KTO auto-retrain path. Today `shared/flywheel/inference_logger.py` records messages, tool calls,
finish reason, and token *counts* (`usage.prompt_tokens` / `usage.completion_tokens`) but throws away the
token IDs and logprobs. That is enough for SFT/KTO (which re-tokenize text) but **not** for token-faithful
GRPO, which — per POLAR — needs the exact sampled tokens and their sampling logprobs.

**Design principle:** Additive and opt-in. New optional fields on the log record, populated only when the
upstream request asks vLLM for logprobs and the response includes them. Zero behavior change for existing
SFT/KTO consumers; the catalog and JSONL stay readable by current code.

---

## Why This Fits

The proxy is already structurally what POLAR calls a "model API proxy" — it just doesn't keep the tokens.

| POLAR Concept | Existing Infrastructure | Gap |
|---|---|---|
| API proxy in front of the model | `services/proxy/app.py` — OpenAI-compatible, forwards to vLLM, logs chat completions | None — the proxy exists |
| Record token-level interactions | `inference_logger._build_record()` captures `prompt_tokens`/`completion_tokens` counts | No token IDs, no logprobs |
| Token-faithful trajectory for RL | `InferenceLogRecord` + date-partitioned JSONL + `LogCatalog` | Record schema has no token-ID/logprob fields |
| Reward → sampled tokens (GRPO feed) | `shared/flywheel/stager.py` routes logs to SFT/KTO datasets | No GRPO/rollout staging path |
| Reconstruct faithful sequence | `token-faithful-grpo-rollout-plan.md` assembly logic | Reuse the same assembly once token IDs are available |

---

## What Gets Built

| # | Artifact | Type | Est. Lines | Purpose |
|---|----------|------|-----------|---------|
| 0 | vLLM logprob/token-id capability spike | Investigation | — | Confirm what vLLM's chat-completions response exposes (token IDs, logprobs) and under which request flags |
| 1 | `shared/flywheel/catalog.py` (`InferenceLogRecord`) | Edit | +~25 | Optional `completion_token_ids`, `completion_logprobs`, `prompt_token_ids` fields + schema migration |
| 2 | `shared/flywheel/inference_logger.py` | Edit | +~40 | Extract token IDs/logprobs from response when present; never fail if absent |
| 3 | `services/proxy/app.py` | Edit | +~10 | Pass through `logprobs` request flag handling; no forced injection |
| 4 | `shared/flywheel/stager.py` | Edit | +~70 | New `rollout`/GRPO staging path that emits token-faithful rollout records |
| 5 | `configs/flywheel/*.yaml` | Edit | +~12 | `capture_token_ids` + `rollout_staging` config |
| 6 | `tests/flywheel/test_token_capture.py` | New test | ~150 | Capture-present, capture-absent, schema back-compat, GRPO staging shape |
| 7 | `.skills/fine-tuning/` + flywheel docs | Skill update | +~40 | Token-capture + GRPO-feed section |

**Total: ~145 net lines Python, ~12 lines YAML, ~150 lines tests, ~40 lines docs. 4 modules edited, 1 new test, config + skill updates.**

No new service. No new dependency. No change to the SFT/KTO staging paths.

---

## Data Flow

```
Agent / client → POST /v1/chat/completions (logprobs: true)
       │
       ▼
┌────────────────────────────┐
│ services/proxy/app.py      │  Forward transparently to vLLM (unchanged).
│  proxy() catch-all          │  Fire-and-forget log on 200.
└──────────┬─────────────────┘
           │ request + response JSON
           ▼
┌────────────────────────────┐
│ inference_logger            │  _build_record(): extract token IDs +
│  ._build_record()           │  logprobs from response.choices[].logprobs
│                             │  when present; leave None otherwise.
└──────────┬─────────────────┘
           │ InferenceLogRecord (+ token fields)
           ▼
┌────────────────────────────┐
│ date-partitioned JSONL      │  Same files; new optional keys. Old
│  + LogCatalog index         │  readers ignore unknown fields.
└──────────┬─────────────────┘
           │ (offline)
           ▼
┌────────────────────────────┐
│ stager.py  rollout path     │  Select records WHERE token_ids present
│                             │  AND tool/reward signal qualifies. Emit
│                             │  token-faithful rollout JSONL for GRPO.
└──────────┬─────────────────┘
           │
           ▼
   GRPO dataset (reuses token-faithful assembly from companion plan)
```

---

## Config Schema

### `flywheel` config additions

```yaml
flywheel:
  logging:
    # Persist completion token IDs + per-token logprobs when the upstream
    # response includes them. Required for any future GRPO/rollout feed.
    # Off by default — SFT/KTO do not need it and it grows log size.
    capture_token_ids: false

  staging:
    # New rollout staging target alongside sft/kto.
    rollout:
      enabled: false
      # Only records with captured token IDs qualify.
      require_token_ids: true
      # Minimum signal to include a record as a rollout sample.
      require_tool_call: true
      output_path: Datasets/flywheel/rollouts/
```

**Backward compatibility:** all new fields default off/false. With them off, the proxy, logger, catalog,
and stager behave exactly as today. SFT/KTO staging is untouched.

---

## Implementation Phases

### Phase 0: vLLM Capability Spike

**Delegate to:** `pact-architect`

| Task | Details |
|------|---------|
| Inspect vLLM response | Confirm the shape of `choices[].logprobs` from the deployed vLLM build: token strings, `logprob` values, `bytes`, and whether token **IDs** are available (directly or via a vLLM `extra_body` flag). |
| Determine request flags | Identify what the client/proxy must send (`logprobs: true`, `top_logprobs`, any `extra_body`) for IDs/logprobs to appear. |
| Decide capture surface | If token IDs are not natively returned, decide whether to (A) capture logprob+token-string sequences only, or (B) re-tokenize server-side at stage time. Document the tradeoff. |
| Record findings | Append a "vLLM capture contract" note to this plan before Phase 1. |

**Gate:** Phase 1 schema design follows from what vLLM actually exposes. Do not guess field availability.

---

### Phase 1: Record Schema + Migration

**Delegate to:** `pact-backend-coder`

| Task | Details |
|------|---------|
| Extend `InferenceLogRecord` | Add optional `completion_token_ids: list[int] | None`, `completion_logprobs: list[float] | None`, `prompt_token_ids: list[int] | None`. Default `None`. Update `to_json()` to omit when `None` (keep JSONL compact). |
| Catalog migration | Add columns / index handling in `catalog.py` for both sqlite and postgres backends. Migration must be additive and tolerate existing rows (NULL). |
| Back-compat assertion | Existing records (no token fields) must still deserialize and index. |

---

### Phase 2: Capture in Logger + Proxy Pass-Through

**Delegate to:** `pact-backend-coder`

| Task | Details |
|------|---------|
| `inference_logger._build_record()` | When `response.choices[].logprobs` (and IDs, per Phase 0) are present, populate the new fields; otherwise leave `None`. Never raise if absent — capture is best-effort. |
| Credential scrubbing | Token-ID/logprob arrays carry no credentials, but keep `_scrub_*` on text fields unchanged. Confirm scrubbing is not accidentally applied to numeric arrays. |
| `services/proxy/app.py` | Honor a client-sent `logprobs` flag transparently (already forwarded as part of the body). Do **not** force-inject `logprobs: true` — capture only what the caller requested, to avoid changing latency/cost for callers that didn't ask. |
| Config gate | `capture_token_ids: false` → skip population even if present, to keep log size down when GRPO feed is not in use. |

---

### Phase 3: GRPO/Rollout Staging

**Delegate to:** `pact-backend-coder`

| Task | Details |
|------|---------|
| `stager.py` rollout path | New staging target that selects records with token IDs + qualifying signal (`require_tool_call`, reward/fitness hooks reused from existing tagger). Emit a rollout JSONL whose schema matches what the env-GRPO faithful assembly expects (`prompt_ids`, `completion_ids`, `logprobs`, mask metadata). |
| Reuse faithful assembly | Where multi-turn logs exist, reuse the sequence/mask assembly defined in `token-faithful-grpo-rollout-plan.md` rather than re-implementing. (Hard dependency on that plan landing.) |
| Interleave invariant | If the rollout feed ever crosses into KTO, preserve the existing interleaving rule (see CLAUDE.md / `_write_kto`). Rollout staging itself does not interleave. |

---

### Phase 4: Tests

**Delegate to:** `pact-test-engineer`

| Test | Type | What |
|------|------|------|
| `test_capture_present` | Unit | Response with logprobs/token IDs → fields populated correctly. |
| `test_capture_absent` | Unit | Response without logprobs → fields `None`, no error, SFT/KTO record unchanged. |
| `test_schema_backcompat` | Unit | Old JSONL row (no token fields) deserializes and indexes in both catalog backends. |
| `test_rollout_staging_shape` | Integration | Stager emits rollout JSONL matching env-GRPO faithful-assembly schema. |
| `test_capture_disabled` | Unit | `capture_token_ids: false` → fields stay `None` even when present in response. |
| Existing flywheel tests | Regression | SFT/KTO staging + proxy passthrough unaffected. |

---

### Phase 5: Documentation

**Delegate to:** `pact-backend-coder`

| Task | Details |
|------|---------|
| Flywheel + fine-tuning skill | Document `capture_token_ids`, `rollout` staging, the vLLM request flags, and the dependency on the token-faithful rollout assembly. |
| Sync mirrors | `python3 .skills/scripts/sync_skill_trees.py` after edits. |

---

## Risk Mitigation

| Risk | Severity | Mitigation |
|------|----------|------------|
| vLLM does not return token IDs | High | Phase 0 gate. Fallback (B): capture logprob+token-string sequences, re-tokenize at stage time with the served tokenizer (accept minor drift, documented). |
| Log size blowup | Medium | Capture is opt-in (`capture_token_ids: false` default); `to_json()` omits `None`; arrays only kept for qualifying rollout candidates downstream. |
| Catalog migration breaks existing rows | Medium | Additive NULL-tolerant columns; `test_schema_backcompat` on both backends. |
| Latency/cost change for unrelated callers | Medium | Never force `logprobs: true`; capture only what the caller requested. |
| Stale tokenizer at stage time (fallback B) | Low | Pin tokenizer to `model_id` recorded on the log; refuse staging on tokenizer mismatch. |
| Scope creep into harness-agnostic RL | Medium | Explicitly out of scope (below). This plan only enriches the existing on-vLLM logging path. |

---

## Connection to Existing Systems

| Existing System | Relationship |
|---|---|
| **Flywheel proxy** (`services/proxy/app.py`) | Direct target. Already proxies + logs; this adds optional token capture. |
| **Inference logger** (`inference_logger.py`) | Direct target. New optional fields; existing scrubbing/queue/writer untouched. |
| **Catalog** (`catalog.py`) | Additive schema migration on sqlite + postgres. |
| **Stager** (`stager.py`) | New `rollout` target alongside existing `sft`/`kto`. |
| **Token-faithful rollout** (`token-faithful-grpo-rollout-plan.md`) | **Hard dependency** — Phase 3 reuses its sequence/mask assembly. Land that plan first. |
| **Experiment loop** (`experiment_loop.py`) | Future: a captured-rollout dataset could become an input the loop selects. Not in scope. |

---

## Out of Scope (explicitly)

- **Native multi-API proxying (Anthropic Messages / OpenAI Responses / Google generateContent).** POLAR
  speaks all four to be harness-agnostic. Our proxy is OpenAI-compatible in front of our own vLLM; adding
  the other API surfaces is only worthwhile if we pursue full harness-agnostic RL (also out of scope).
- **RL training through external agent harnesses.** Same reasoning as the companion plan — a generation-path
  re-architecture, deferred unless we commit to RL-ing real coding agents.

---

## Success Criteria

1. Phase 0 produces a written vLLM capture-contract decision appended to this plan.
2. With `capture_token_ids: true` and a `logprobs`-requesting client, logged records carry `completion_token_ids` + `completion_logprobs`.
3. With capture off or logprobs absent, records are byte-compatible with today's SFT/KTO consumers.
4. Catalog migration applies cleanly to existing sqlite and postgres rows (NULL token fields).
5. Stager emits a rollout JSONL whose schema is consumable by the env-GRPO token-faithful assembly.
6. All existing flywheel/proxy tests pass unchanged.
7. Skill + flywheel docs updated and mirrors synced.
