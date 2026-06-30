# Distiller Native Tool Emit + Flywheel Verdict Rationale Plan

**Date:** 2026-06-30
**Status:** Proposed
**Prior Art:** Arize Phoenix / OpenInference — [Phoenix docs](https://arize.com/docs/phoenix), [OpenInference semantic conventions](https://github.com/Arize-ai/openinference/blob/main/spec/semantic_conventions.md)
**Branch:** TBD (`feat/distiller-native-tool-emit`)

---

## Origin

A scan of Arize Phoenix to see if anything was worth borrowing. Direct code comparison **deflated** most of it: we already have a richer judge than Phoenix's `llm_classify` "rails" (`shared/judge/models.py` — JSON-Schema output, per-dimension reason-first scoring, `quality_gate` floors-as-gates), and we already attach a score to each inference record and filter by it (`InferenceLogRecord.fitness_score` + catalog `min_score`/`max_score`/`unscored_only` + `update_score()`). We also already have a canonical native tool schema internally (`InferenceLogRecord` stores OpenAI-native `messages`/`tools`/`tool_calls`; SynthChat ships an `openai_native` format). So **OpenInference is not worth adopting wholesale** — it is useful only as a cross-check schema.

What the comparison *did* surface is three concrete, self-contained gaps, in priority order:

1. **The distiller flattens tool calls and discards tool I/O** — internal inconsistency with the native shape the rest of the repo already uses. (Highest payoff; self-contained.)
2. **The flywheel record stores the judge *score* but not the judge *rationale*** — Phoenix's one genuine differentiator (label **and** explanation on every span).
3. **No frozen regression fixtures** — the flywheel routes failures to retrain, but never freezes a curated set of failing logs into a re-runnable `Evaluator/` scenario.

---

## Findings from the codebase

- **`render_completion()` flattens tool calls into prose.** `distill.py:46-52` builds the completion as `text + "\nTool calls: " + json.dumps(tool_calls)`. The emitted `completion` is a string; the assistant turn carries no structured `tool_calls` array.
- **Tool results are stubbed to a literal placeholder.** In `emit_rows`, `_content()` returns the constant string `"[tool result]"` for any `role == "tool"` event (`distill.py:72-73`). The training row's `conversations` context therefore never contains what a tool actually returned.
- **The parse layer already keeps the data the emit layer throws away.** The Claude Code adapter preserves `command` and `output` on each tool event (`adapters/claude_code.py:89-91`); the adapter contract is `{"role":"assistant", "tool_calls":[{"name","input"}]}` (`adapters/base.py:14`). So this is purely an **emit-side** loss, matching the standing note in memory: *"distiller stubs tool I/O but parse layer keeps it → tool-use needs native-format emit."*
- **We already own the target shape.** `InferenceLogRecord` stores OpenAI-native `messages` + `tools` + `tool_calls` (`catalog.py:37-45`); SynthChat's `tool_call_formats.yaml` ships an `openai_native` format (lines 57-82). The distiller is the only producer that does not emit this shape.
- **The flywheel record has score but no rationale.** `InferenceLogRecord` carries `fitness_score`, `is_valid`, `tag`, `errors` (`catalog.py:59-63`) and the catalog indexes/filters on `fitness_score` with `update_score()` (`catalog.py:161,194,328`). `JudgeScore.feedback` + `per_dimension` reasoning exist in the judge module (`shared/judge/models.py:84-90`) but there is **no field on the record to hold them**.

---

## Goal

Bring the transcript distiller's output into line with the OpenAI-native tool shape the rest of the repo already uses (structured `tool_calls` on assistant turns, real tool output linked by id on tool turns), and let the flywheel persist the judge's *rationale* alongside its score so production logs can be triaged by **why**, not just by number. Optionally, add a path to freeze curated flywheel failures into a re-runnable `Evaluator/` regression set.

**Design principle:** Internal consistency over external adoption. The native target schema is the one we already emit elsewhere — do not introduce OpenInference as a dependency. All flywheel changes are additive and opt-in; existing SFT/KTO/DPO/GRPO consumers and old JSONL rows stay byte-compatible.

---

## Why This Fits

| Phoenix / OpenInference Concept | Existing Infrastructure | Gap |
|---|---|---|
| Structured tool-call span (`tool_call.function.name`/`.arguments`) | `InferenceLogRecord.tool_calls`, SynthChat `openai_native` format | Distiller flattens to a `"Tool calls: …"` string (`distill.py:46-52`) |
| Tool result linked to its call (`message.tool_call_id` ↔ `tool_call.id`) | Adapter keeps `command`+`output` per tool event (`claude_code.py:89-91`) | Emit layer replaces it with `"[tool result]"` (`distill.py:72-73`) |
| Eval label **and explanation** attached to each span | `fitness_score`/`is_valid` on record + `update_score()` + score index | No field for the judge's `feedback`/`per_dimension` rationale |
| Datasets-as-experiments (freeze examples, re-run versions) | Stager routes logs to sft/kto/skip; `Evaluator/` runs YAML scenarios | No path from flywheel failures → frozen, re-runnable eval set |
| Rails (constrained label + forced explanation) | `RubricDef.output_schema` + `quality_gate` (`models.py`) | **None — already richer than Phoenix.** Not borrowed. |

---

## What Gets Built

| # | Artifact | Type | Est. Lines | Purpose |
|---|----------|------|-----------|---------|
| 0 | Trainer conversation-schema spike | Investigation | — | Confirm exactly what conversation/tool shape the SFT/KTO/DPO/GRPO loaders accept, so native emit stays consumable through method-specific projections |
| 1 | `.skills/transcript-distillation/scripts/distill.py` | Edit | +~60 | Emit structured `tool_calls` on assistant turns + real tool output (linked by id) on tool turns, behind a config flag; keep flat-string mode as default until trainer-verified |
| 2 | `.skills/transcript-distillation/scripts/adapters/*` | Edit | +~25 | Surface a stable `tool_call_id` ↔ result linkage in the normalized event contract (`base.py`, `claude_code.py`, `codex.py`) |
| 3 | `.skills/transcript-distillation/configs/*.yaml` | Edit | +~10 | `render_mode: flat \| native` (+ tool-output truncation budget) |
| 4 | `shared/flywheel/catalog.py` (`InferenceLogRecord` + backends) | Edit | +~30 | Add `verdict_rationale: str \| None` (+ optional per-rubric labels); additive sqlite/postgres column |
| 5 | `shared/flywheel/*` writer of scores | Edit | +~15 | Extend `update_score()` path to also persist judge `feedback`/`per_dimension` when present |
| 6 | Frozen-fixtures exporter (flywheel → `Evaluator/`) | New | ~120 | CLI to select failing logs by filter and emit a versioned assertion-YAML scenario set |
| 7 | `tests/` (distiller + flywheel) | New test | ~200 | Native-emit shape, id linkage, flat back-compat, rationale persistence, fixture export shape |
| 8 | Skill + docs updates | Skill update | +~50 | `transcript-distillation`, `fine-tuning`/flywheel, `evaluation` sections + mirror sync |

**Total: ~110 net lines Python, ~10 lines YAML, ~200 lines tests, ~50 lines docs/skills. 5 files edited, 2 new artifacts, config + skill updates.** No new dependency. No change to the default distiller output or SFT/KTO/DPO/GRPO staging until trainer-verified.

---

## Data Flow

```
Agent transcript (Claude Code / Codex JSONL)
        │
        ▼
┌─────────────────────────────┐
│ adapters/*.py               │  Parse → normalized events. Already keep
│  (parse layer)              │  tool name/input + result command/output;
│                             │  ADD: stable tool_call_id ↔ result linkage.
└──────────┬──────────────────┘
           │ normalized events
           ▼
┌─────────────────────────────┐   render_mode: native
│ distill.py emit_rows()      │  ── assistant turn → {content, tool_calls:[…]}
│                             │  ── tool turn → real output, linked by id
│                             │   render_mode: flat  (default, unchanged)
│                             │  ── "Tool calls: …" string + "[tool result]"
└──────────┬──────────────────┘
           │ training rows ({conversations, completion, label, …})
           ▼
   SFT / KTO / DPO / GRPO projections  (shape confirmed in Phase 0)

─────────────────────────────────────────────────────────────────────

Production inference (separate path)
   proxy → inference_logger → InferenceLogRecord (fitness_score)
        │
        ▼  judge scores a record
   update_score(log_id, fitness_score, is_valid, errors,
                + verdict_rationale, + per_rubric_labels)   ← NEW
        │
        ▼  (offline, optional)
   frozen-fixtures exporter → versioned Evaluator/ scenario set
```

---

## Config Schema

### Distiller config additions

```yaml
distill:
  render:
    # flat  = today's behavior: tool calls flattened into the completion
    #         string ("Tool calls: …"), tool results stubbed to "[tool result]".
    # native = structured tool_calls on assistant turns + real tool output
    #          (linked by tool_call_id) on tool turns.
    # Default stays `flat` until the trainer loaders are confirmed (Phase 0).
    render_mode: flat
    # Cap per tool-result body in native mode (chars). Oversized output is
    # truncated with a marker so context budget stays bounded.
    max_tool_output_chars: 2000
```

### Flywheel config / record additions

```yaml
flywheel:
  scoring:
    # Persist the judge's rationale + per-rubric labels onto the record,
    # not just the numeric fitness_score. Off by default (additive column).
    persist_verdict_rationale: false
```

**Backward compatibility:** `render_mode: flat` reproduces today's distiller output byte-for-byte. `persist_verdict_rationale: false` and the new NULL-able `verdict_rationale` column leave the catalog readable by current code; old JSONL rows deserialize unchanged.

---

## Implementation Phases

### Phase 0: Trainer Conversation-Schema Spike

**Delegate to:** `pact-architect`

| Task | Details |
|------|---------|
| Confirm loader contract | Determine exactly what conversation/tool shape the SFT, KTO, DPO, and GRPO loaders accept today (do they read `conversations` as `{role, content}` only, or can they consume `tool_calls` + `role:"tool"` turns with `tool_call_id`?). |
| Decide native target | Pick the native emit shape: OpenAI-native messages (`tool_calls` on assistant, `role:"tool"` + `tool_call_id`) vs. a render that keeps a single `completion` string but structured. Reuse SynthChat's `openai_native` conventions where possible. |
| Cross-check vs OpenInference | One-time check that the chosen shape covers id linkage (`tool_call.id` ↔ `message.tool_call_id`) and multi-call turns. Do **not** adopt OpenInference attribute names. |
| Record findings | Append a "native emit contract" note to this plan before Phase 1. |

**Gate:** Phase 1 emit design follows what the trainers actually accept. Do not change the default `render_mode` until this confirms native rows train cleanly.

### Phase 0 Native Emit Contract Decision

Native emit is a distiller source/emit format, not a replacement for every trainer's final input schema. The distiller keeps `render_mode: flat` as the default and adds opt-in `render_mode: native`; downstream training paths must project native rows into the shape each method actually accepts.

Native rows use the existing OpenAI-native message convention:

```json
{
  "conversations": [
    {"role": "user", "content": "..."},
    {
      "role": "assistant",
      "content": "...",
      "tool_calls": [
        {
          "id": "call_stable_id",
          "type": "function",
          "function": {"name": "Edit", "arguments": "{\"file_path\":\"...\"}"}
        }
      ]
    },
    {"role": "tool", "tool_call_id": "call_stable_id", "content": "..."}
  ],
  "prompt": "...",
  "completion": "...",
  "label": true,
  "metadata": {"render_mode": "native"}
}
```

Every assistant tool call and its matching tool-result turn must share a stable `tool_call_id`. Existing source ids should be preserved where the transcript format provides them; otherwise the adapter must derive a deterministic id from stable transcript position data so repeated distillation runs produce the same linkage.

Trainer compatibility findings:

| Method | Native compatibility decision |
|--------|-------------------------------|
| SFT | Prepare currently renders native tool messages back into text for training. Native emit is useful as source data, but SFT still needs a text projection until the SFT loader intentionally accepts structured tool messages. |
| KTO | KTO partially handles assistant tool calls, but drops tool-result target context. KTO rows need a projection that preserves the intended preference target rather than assuming native rows can be consumed directly. |
| DPO | DPO requires paired role/content `prompt`, `chosen`, and `rejected` examples. Native rows require a pair-building projection before DPO training. |
| Static GRPO | Static GRPO needs a prompt plus reward-ground-truth columns. Native rows require projection into the configured reward data shape. |
| Env-GRPO | Environment-backed GRPO is rollout-row based. Native distiller rows can seed prompts or references, but do not replace rollout records. |

This means `render_mode: native` is the high-fidelity emitted source contract for tool-use data; each trainer owns an explicit method-specific projection from that source contract into its runtime dataset contract.

---

### Phase 1: Adapter Linkage Contract

**Delegate to:** `pact-backend-coder`

| Task | Details |
|------|---------|
| Stable call↔result id | Extend the normalized event contract (`adapters/base.py`) so each assistant `tool_calls[i]` and its corresponding `role:"tool"` result share a `tool_call_id`. Claude Code already has `tool_use_id` (`claude_code.py:90`); thread it through instead of only `id2cmd`. |
| Codex parity | Mirror the linkage in `adapters/codex.py` (`function_call` / `function_call_output`). |
| No emit change yet | Adapters only enrich events; `emit_rows` still defaults to flat. Keep adapters independently testable. |

---

### Phase 2: Native Emit in the Distiller

**Delegate to:** `pact-backend-coder`

| Task | Details |
|------|---------|
| `render_mode` switch | Add the config flag; `flat` keeps `render_completion()` exactly as-is (`distill.py:46-52`). |
| Native assistant turn | In native mode, emit structured `tool_calls` (name + parsed input) on the assistant turn instead of the `"Tool calls: …"` string. |
| Native tool turn | Replace the `"[tool result]"` stub (`distill.py:72-73`) with the real captured `output`, truncated to `max_tool_output_chars`, carrying its `tool_call_id`. |
| Context-budget accounting | `est()`/budget logic in `emit_rows` must count real tool output; ensure oversize handling still trims oldest-first without dropping the linked call/result pair inconsistently. |
| Labeling untouched | The deterministic good/bad/borderline labeling and `failure_labels` must be byte-identical between flat and native (only the rendered content differs). |

---

### Phase 3: Flywheel Verdict Rationale

**Delegate to:** `pact-backend-coder`

| Task | Details |
|------|---------|
| Record field | Add `verdict_rationale: str \| None` (and optional `rubric_labels: dict \| None`) to `InferenceLogRecord` (`catalog.py`), `None` by default; `to_json()` omits when unset. |
| Catalog migration | Additive NULL-able column on both sqlite and postgres backends; existing rows tolerate NULL. |
| Persist on score | Extend the `update_score()` write path so that, when `persist_verdict_rationale` is on and a `JudgeResult` is available, the judge's `feedback`/`per_dimension` reasoning is written alongside `fitness_score`. |
| Make it filterable | Ensure the rationale is retrievable in the same catalog query path that already filters by `fitness_score`. |

---

### Phase 4: Frozen Regression Fixtures (optional, lower priority)

**Delegate to:** `pact-backend-coder`

| Task | Details |
|------|---------|
| Exporter CLI | Select flywheel logs by existing filters (e.g. `is_valid == false`, `max_score`, `tag`) and emit a versioned `Evaluator/` assertion-YAML scenario set. |
| Reuse Evaluator schema | Output must be a valid Evaluator scenario consumable by the existing config-driven runner — no new eval engine. |
| Versioning | Stamp the frozen set with a version/date so re-runs across candidate models are comparable. |

**Gate:** Only build after Phases 1–3 land and there is demand for a standing regression set.

---

### Phase 5: Tests

**Delegate to:** `pact-test-engineer`

| Test | Type | What |
|------|------|------|
| `test_emit_flat_unchanged` | Unit | `render_mode: flat` output is byte-identical to current emit on a fixture transcript. |
| `test_emit_native_tool_calls` | Unit | Native mode emits structured `tool_calls` on the assistant turn; no `"Tool calls: …"` string. |
| `test_emit_native_tool_result` | Unit | Tool turn carries real (truncated) output with the linking `tool_call_id`, not `"[tool result]"`. |
| `test_adapter_id_linkage` | Unit | Claude Code + Codex adapters emit matching `tool_call_id` on call and result. |
| `test_native_budget_trim` | Unit | Oversize tool output truncates; call/result pairing stays consistent under context-budget trimming. |
| `test_verdict_rationale_persist` | Unit | With the flag on, `update_score` writes rationale; with it off, record is byte-compatible. |
| `test_catalog_backcompat` | Unit | Old rows (no `verdict_rationale`) deserialize and index on both backends. |
| `test_fixture_export_shape` | Integration | Exporter emits a valid Evaluator scenario the runner can load. |

---

### Phase 6: Documentation

**Delegate to:** `pact-backend-coder`

| Task | Details |
|------|---------|
| Skill updates | `transcript-distillation` (render modes), `fine-tuning`/flywheel (verdict rationale, fixture export), `evaluation` (frozen regression set). |
| Sync mirrors | `python3 .skills/scripts/sync_skill_trees.py` after editing `.skills/` (mirrors under `.agents/skills`, `.claude/skills`, `.codex/skills`). |

---

## Risk Mitigation

| Risk | Severity | Mitigation |
|------|----------|------------|
| Native rows don't train cleanly in SFT/KTO/DPO/GRPO loaders | High | Phase 0 gate; default stays `flat` until method-specific projections are verified; native is opt-in. |
| Default distiller output changes silently | High | `render_mode: flat` is the default and asserted byte-identical (`test_emit_flat_unchanged`). |
| Context-budget blowup from real tool output | Medium | `max_tool_output_chars` truncation; oversize trimming covered by test. |
| Call/result pairing breaks under trimming | Medium | Linkage by `tool_call_id`; explicit pairing-consistency test. |
| Catalog migration breaks existing rows | Medium | Additive NULL-able column on both backends; back-compat test. |
| Skill mirrors drift from `.skills/` source | Low | `sync_skill_trees.py --check` in Phase 6. |
| Scope creep into adopting OpenInference as a dependency | Medium | Explicitly out of scope — internal native shape only. |

---

## Connection to Existing Systems

| Existing System | Relationship |
|---|---|
| **Transcript distiller** (`.skills/transcript-distillation/scripts/`) | Direct target. New `render_mode`; adapters enrich linkage; flat path unchanged. |
| **SynthChat tool formats** (`SynthChat/config/tool_call_formats.yaml`) | Reference shape — native emit reuses `openai_native` conventions for consistency. |
| **Flywheel record/catalog** (`shared/flywheel/catalog.py`) | Additive `verdict_rationale` column; existing score index/filters untouched. |
| **Judge** (`shared/judge/models.py`) | Source of the rationale (`JudgeScore.feedback`/`per_dimension`) now persisted onto the record. |
| **Evaluator** (`Evaluator/`) | Frozen-fixtures exporter emits scenarios consumable by the existing runner. |
| **`project_finetune_personal_transcripts` memory** | This plan resolves the standing "tool-use needs native-format emit" gap it records. |

---

## Out of Scope (explicitly)

- **Adopting OpenInference / OpenTelemetry as a schema or dependency.** We already have an internal native tool shape; the borrow is internal consistency, not external instrumentation.
- **A Phoenix-style trace collector or UI.** We are not debugging a deployed app; the flywheel JSONL + catalog already serve offline triage.
- **Phoenix prompt playground / span replay.** Assumes a live app surface we don't run.
- **Changing the judge's scoring mechanics.** The judge is already richer than Phoenix's rails; only its *output persistence* changes here.

---

## Success Criteria

1. Phase 0 produces a written "native emit contract" decision appended to this plan, confirming the trainer-accepted shape.
2. `render_mode: flat` output is byte-identical to today's distiller on fixture transcripts.
3. `render_mode: native` emits structured `tool_calls` on assistant turns and real, id-linked tool output on tool turns — no `"Tool calls: …"` string, no `"[tool result]"` stub.
4. Claude Code and Codex adapters emit matching `tool_call_id` on each call and its result.
5. With `persist_verdict_rationale: true`, scored flywheel records carry the judge's rationale; with it off, records are byte-compatible with current readers on both backends.
6. (If Phase 4 built) The exporter produces a versioned Evaluator scenario set the existing runner loads and runs.
7. All existing distiller, flywheel, and evaluator tests pass unchanged; skills updated and mirrors synced.
