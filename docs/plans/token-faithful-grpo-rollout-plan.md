# Token-Faithful Multi-Turn GRPO Rollout Plan

**Date:** 2026-05-29
**Status:** Proposed
**Paper / Prior Art:** NVIDIA POLAR — [arXiv:2605.24220](https://arxiv.org/html/2605.24220v1) ([MarkTechPost writeup](https://www.marktechpost.com/2026/05/27/nvidia-releases-polar-a-token-faithful-rollout-framework-for-grpo-training-across-codex-claude-code-and-qwen-code/))
**Branch:** `claude/polar-token-rollout-MTzCp`

---

## Goal

Make our environment-backed GRPO rollout **token-faithful**: train on the exact token sequence the model actually produced and consumed during a multi-turn episode, with a per-token loss mask that trains only the assistant-sampled tokens and keeps tool-result / user-feedback tokens as masked context.

Today `Trainers/grpo/src/env_rollout.py` flattens every assistant turn's `completion_ids` into one continuous completion and keeps only the **first** turn's `prompt_ids`. The interleaved tool-result and user-feedback tokens that the model genuinely saw between turns are dropped from the trained sequence entirely. This is exactly the train/inference mismatch POLAR was built to eliminate: GRPO ends up optimizing a fictional contiguous assistant stream that never existed at sampling time.

**Design principle:** Minimal new code, no new entrypoint, no new dependency. Fix the faithfulness gap inside the existing `rollout_func` path and assert the contract with TRL. Behavior for single-turn episodes (the common case) stays byte-identical; the change only adds correct handling for multi-turn episodes.

---

## Why This Fits

We already have most of POLAR's token-faithful machinery — we just discard part of it on the prompt side:

| POLAR Concept | Existing Infrastructure | Gap |
|---|---|---|
| Record actual sampled token IDs | `_generate_one_completion()` already returns `completion_ids` + per-token `logprobs` per turn | None for completions — we keep them |
| Reward attached to sampled tokens | `EpisodeRolloutResult.completion_ids` → TRL `GRPOTrainer` | Sequence is flattened; intermediate context missing |
| Token-faithful trajectory stitching | Per-turn `prompt_ids` available from `_generate_one_completion()` | We keep only `first_prompt_ids`; tool-result/user tokens never enter the sequence |
| Loss masking (train assistant, not context) | TRL GRPO completion-mask mechanism | Not used — all `all_completion_ids` are treated as trainable |
| Multi-turn episode loop | `_run_single_episode()` already drives turns + env feedback | Loop is correct; only the *serialization* of its result is unfaithful |

The episode loop, env execution, and reward path are all sound. The defect is localized to how `_run_single_episode()` assembles its `EpisodeRolloutResult` and how `build_rollout_func()` hands it to TRL.

---

## What Gets Built

| # | Artifact | Type | Est. Lines | Purpose |
|---|----------|------|-----------|---------|
| 0 | TRL contract spike (notes in plan) | Investigation | — | Confirm `rollout_func` return schema supports a completion mask / masked context before coding |
| 1 | `Trainers/grpo/src/env_rollout.py` | Edit | ~80 net | Faithful token-sequence assembly + per-token loss mask |
| 2 | `Trainers/grpo/configs/env_config.yaml` | Edit | +~8 | `env_training.token_faithful` toggle + mask policy |
| 3 | `tests/trainers/grpo/test_env_rollout_faithful.py` | New test | ~180 | Single-turn parity + multi-turn faithfulness + mask alignment |
| 4 | `.skills/fine-tuning/reference/grpo-training.md` | Skill update | +~40 | Token-faithful rollout section |
| 5 | `.skills/fine-tuning/SKILL.md` | Skill update | +~3 | Note in env-GRPO gotchas |

**Total: ~80 net lines of Python, ~8 lines YAML, ~180 lines tests, ~43 lines docs. 1 module edited, 1 config edited, 1 new test, 2 skill files.**

No new training entrypoint. No new dependency. No change to reward functions, dataset loader, or callbacks.

---

## The Faithfulness Gap (current vs. target)

### Current (`env_rollout.py:122-233`)

```
first_prompt_ids  = prompt_ids(turn 1)            # only turn 1's prompt kept
all_completion_ids = completion_ids(t1) ++ completion_ids(t2) ++ ...   # flattened
all_logprobs       = logprobs(t1) ++ logprobs(t2) ++ ...

return prompt_ids = first_prompt_ids
       completion_ids = all_completion_ids        # contiguous assistant stream
       logprobs = all_logprobs                    # sampling logprobs, flat
```

Problem: between `completion_ids(t1)` and `completion_ids(t2)` the model actually saw rendered
tool-result + user-feedback tokens. Those are appended to `messages` (`env_rollout.py:157,188,193`)
and re-rendered next turn via `apply_chat_template` (`env_rollout.py:132`), but they never appear in
the trained sequence. When GRPO recomputes current-policy logprobs over `prompt_ids + completion_ids`,
the positions no longer line up with the cached sampling `logprobs`, and the advantage is applied to a
sequence the policy never generated in that form.

### Target (token-faithful)

```
sequence_ids  = prompt_ids(t1)
                ++ completion_ids(t1)        [mask=1, trainable]
                ++ context_ids(t1→t2)        [mask=0, tool result + user feedback, as actually rendered]
                ++ completion_ids(t2)        [mask=1]
                ++ context_ids(t2→t3)        [mask=0]
                ++ ...
loss_mask     = aligned 0/1 per token (1 only on assistant-sampled tokens)
old_logprobs  = sampling logprobs, placed at mask=1 positions
```

`context_ids(tN→tN+1)` are obtained by tokenizing the **delta** of the rendered chat between turns
(render after appending the assistant+feedback messages, diff against the previous render's token
prefix), so the masked context is the exact bytes the next turn's prompt contained — no re-tokenization
drift on the trainable tokens.

---

## Config Schema

### New `token_faithful` block under `env_training:` in `env_config.yaml`

```yaml
env_training:
  # ... existing fields unchanged ...

  # Token-faithful multi-turn serialization (POLAR-style).
  # When true, the rollout preserves the full interleaved token sequence with a
  # per-token loss mask instead of flattening assistant turns. Single-turn
  # episodes are unaffected either way.
  token_faithful: true

  # How to treat tokens the model saw but did not generate (tool results,
  # user feedback prompts). "mask" = keep as context, exclude from loss.
  # "drop" = legacy flattened behavior (not faithful; kept only for A/B).
  context_token_policy: mask   # "mask" | "drop"
```

**Backward compatibility:** `token_faithful: false` (or `context_token_policy: drop`) reproduces today's
flattened behavior exactly, for A/B comparison. Default flips to faithful once Phase 0 confirms the TRL
contract.

---

## Implementation Phases

### Phase 0: TRL `rollout_func` Contract Spike

**Delegate to:** `pact-architect`

| Task | Details |
|------|---------|
| Confirm return schema | Inspect the installed `trl.experimental.openenv` / GRPO `rollout_func` contract: does TRL accept a per-sample `completion_mask` (or equivalent) alongside `completion_ids`, or does it derive the mask from `prompt_ids` length? |
| Decide mask delivery | Two candidate designs: (A) extend the rollout dict with a `completion_mask` key TRL honors; (B) fold masked context into `prompt_ids` per-turn and emit one rollout record per assistant turn (turn-level GRPO). Pick based on what TRL actually supports. |
| Verify logprob alignment | Confirm how TRL aligns supplied `logprobs` to `completion_ids` so masked positions don't corrupt the importance ratio. |
| Record findings | Append a short "TRL contract" note to this plan before Phase 1 starts. |

**Gate:** Do not start Phase 1 until the mask-delivery design is chosen and written down. If TRL exposes no
mask hook, design (B) (turn-level records) becomes the implementation and the rest of the plan adjusts to
emit N records per episode.

---

### Phase 1: Faithful Sequence Assembly (`env_rollout.py`)

**Delegate to:** `pact-backend-coder`

| Task | Details |
|------|---------|
| Capture per-turn `prompt_ids` | In `_run_single_episode()`, stop discarding non-first `prompt_ids`. Track the running rendered-token prefix so each turn's *new* context tokens can be isolated. |
| Build interleaved sequence | Assemble `sequence_ids` = initial prompt + alternating (assistant completion, masked context) segments, using the exact tokens from `_generate_one_completion()` for assistant spans and the rendered delta for context spans. |
| Build `loss_mask` | 1 on assistant-sampled token positions, 0 on prompt/context positions. Length must equal `len(sequence_ids)`. |
| Align `old_logprobs` | Place per-turn sampling `logprobs` at their mask=1 positions; context positions carry no contribution. |
| Extend `EpisodeRolloutResult` | Add `sequence_ids`, `loss_mask` (and keep `completion_ids`/`logprobs` for the legacy/`drop` path). |
| Wire `build_rollout_func()` | Emit the mask per the Phase 0 design (extra dict key **or** turn-level records). Keep the existing metric keys (`env_reward`, `stop_reason`, `total_turns`, ...) untouched. |
| Honor config toggle | `context_token_policy: drop` → current flattened path; `mask` → faithful path. |

**Patterns to follow:**
- Existing turn loop and env feedback (`env_rollout.py:131-209`) is reused as-is; only result assembly changes.
- `parse_response` / `format_tool_results_message` usage stays identical.

**Invariant to preserve (call out in review):** single-turn episodes must produce a byte-identical
`(prompt_ids, completion_ids, logprobs)` to today. The faithful path only diverges when `total_turns > 1`.

---

### Phase 2: Tests

**Delegate to:** `pact-test-engineer`

| Test | Type | What |
|------|------|------|
| `test_single_turn_parity` | Unit | A 1-turn episode yields identical `prompt_ids`/`completion_ids`/`logprobs` under `mask` and `drop` policies (no regression for the common case). |
| `test_multi_turn_sequence_faithful` | Unit | A 3-turn episode: assert `sequence_ids` == prompt ++ comp1 ++ ctx1 ++ comp2 ++ ctx2 ++ comp3, with stubbed tokenizer/generation. |
| `test_loss_mask_alignment` | Unit | `len(loss_mask) == len(sequence_ids)`; mask=1 positions exactly cover assistant-sampled tokens; `old_logprobs` align to mask=1 count. |
| `test_context_token_policy_drop` | Unit | `drop` reproduces the legacy flattened result. |
| `test_rollout_func_contract` | Integration | `build_rollout_func()` output matches the Phase 0 TRL contract (key presence / per-record shape). |
| Existing env-GRPO tests | Regression | Confirm no breakage in current suite. |

Use stubbed `trainer` / `generate_rollout_completions` (as in current tests) so no model load is required.

---

### Phase 3: Skill Documentation

**Delegate to:** `pact-backend-coder`

| Task | Details |
|------|---------|
| `.skills/fine-tuning/reference/grpo-training.md` | New "Token-Faithful Multi-Turn Rollout" section: what faithfulness means, the `token_faithful` / `context_token_policy` config, when to A/B against `drop`, and the single-turn-parity invariant. |
| `.skills/fine-tuning/SKILL.md` | One gotcha line under env-GRPO notes pointing at the new section. |
| Sync mirrors | `python3 .skills/scripts/sync_skill_trees.py` after edits. |

---

## Risk Mitigation

| Risk | Severity | Mitigation |
|------|----------|------------|
| TRL `rollout_func` exposes no mask hook | High | Phase 0 gate. Fallback design (B): emit one rollout record per assistant turn with masked context folded into `prompt_ids`. |
| Re-tokenization drift on context deltas | Medium | Derive context tokens from rendered-string deltas, not independent re-tokenization of fragments; assert prefix-consistency in `test_multi_turn_sequence_faithful`. |
| Silent regression for single-turn episodes | Medium | `test_single_turn_parity` + explicit invariant in review. Default could stay `drop` until parity test is green. |
| Sequence length blowup on long episodes | Low | `max_completion_length` / `max_turns` already cap episode size; faithful path adds only the context tokens that were already rendered into prompts. |
| logprob/length mismatch crashes TRL | Medium | `test_loss_mask_alignment` asserts counts before any training run; dry-run a 5-step smoke job before a full launch. |

---

## Connection to Existing Systems

| Existing System | Relationship |
|---|---|
| **Env-GRPO** (`train_env_grpo.py`, `env_rollout.py`) | Direct target. This plan hardens the rollout serialization without touching the entrypoint or trainer wiring. |
| **PivotRL** (`pivot_profiler.py`, `pivot_config.yaml`) | Orthogonal. PivotRL selects *which* turns to train on; this selects *how faithfully* a chosen episode is tokenized. They compose. |
| **Flywheel proxy** (`services/proxy/`, `inference_logger.py`) | See companion plan `proxy-token-capture-grpo-feed-plan.md` — the proxy would supply faithful token IDs/logprobs for a future flywheel→GRPO feed; this plan fixes the on-policy trainer path first. |
| **Reward system** (`env_rewards.py`) | Unchanged. Rewards still score the episode; only the token sequence the reward is attached to becomes faithful. |

---

## Out of Scope (explicitly)

- **Harness-agnostic RL through external agents (Claude Code / Codex / Qwen Code).** POLAR's headline
  capability — proxying a real coding-agent harness and RL-training through it — is a re-architecture of
  our generation path (we use TRL colocate vLLM against SynthChat envs). Not pursued here; revisit only if
  we decide to RL real coding agents.

---

## Success Criteria

1. Phase 0 produces a written TRL-contract decision appended to this plan.
2. A 1-turn episode yields byte-identical `prompt_ids`/`completion_ids`/`logprobs` under `mask` and `drop`.
3. A multi-turn episode produces `sequence_ids` containing the rendered tool-result/user-feedback tokens between assistant turns, with a `loss_mask` that is 1 only on assistant-sampled tokens.
4. `len(loss_mask) == len(sequence_ids)` and `old_logprobs` align to the mask=1 count.
5. A 5-step env-GRPO smoke run completes with `token_faithful: true` and no length/logprob mismatch.
6. `context_token_policy: drop` reproduces today's behavior; all existing env-GRPO tests pass.
7. Skill docs updated and mirrors synced.
