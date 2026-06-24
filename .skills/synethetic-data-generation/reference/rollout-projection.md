# Rollout Projection & Dataset Curation Reference

The projector `SynthChat/scripts/project_rollout_datasets.py` turns saved environment-rollout artifacts into training datasets. It always writes canonical, KTO, and GRPO outputs, and can optionally write a full multi-turn SFT output. A config-driven filter engine lets you curate which rollouts reach each projection target.

This reference covers three things:

1. The generic, config-driven **rollout-filter engine** (`--filter-config`)
2. **Full-trajectory SFT projection** (`--sft-output`)
3. **Data-recipe guidance** heuristics for planning generation runs

---

## 1. Generic Rollout-Filter Engine

The projector accepts `--filter-config <yaml>` pointing at a YAML file with a `projection.filters` list. The engine lives at `shared/validation/rollout_filters.py` and is fully generic — it filters arbitrary record dicts on ANY field via dot-path, with a generic operator table. It knows nothing about "turns" or "tools" specifically.

**Default OFF.** An empty or absent `--filter-config` is a no-op: every record passes for every target. You opt into filtering by supplying a config.

### YAML shape

```yaml
projection:
  filters:
    - field: metadata.environment.episode_trace.total_turns   # dot-path into the record
      op: gte
      value: 5
      applies_to: [grpo, kto_positive]   # optional; default = sft, grpo, kto_positive (NOT kto_negative)
      on_missing: keep                   # keep (default) | drop
```

- `field` — dot-path addressed into the rollout record as-is (top-level keys include `metadata`, so e.g. `metadata.environment.episode_trace.total_turns`).
- `op` — comparison operator (see below).
- `value` — the comparison operand. `in`/`not_in` take a list; `exists`/`missing` ignore it.
- `applies_to` — optional list of targets this filter applies to. When omitted, it applies to the **default target set** (see below).
- `on_missing` — `keep` (default) or `drop`; decides the verdict when the addressed field is absent.

### Operators

`eq, ne, gt, gte, lt, lte, in, not_in, exists, missing`

- `in` / `not_in` take a **list** value (membership test).
- `exists` / `missing` test field presence and ignore `value`.
- Spec validation is eager and fail-closed: an unknown `op`, bad `on_missing`, or a missing required key raises at load time rather than being silently ignored.

### Targets

`sft, grpo, kto_positive, kto_negative`

The **default target set** when `applies_to` is omitted is `sft, grpo, kto_positive`. Hard negatives (`kto_negative`) are **NEVER** filtered unless `kto_negative` is explicitly listed in `applies_to`. This is deliberate: quality filters that drop short or low-scoring rollouts should never silently discard your hard-negative training signal.

### `on_missing` semantics

Records that lack the addressed field **pass through by default** (`on_missing: keep`). Set `on_missing: drop` to exclude records missing the field. There is no silent data loss either way — the choice is explicit per filter.

### AND semantics + drop breakdown

Within a target, all applicable filters must match (logical AND) for a record to pass. When filters are active, the run summary (the JSON the projector prints) includes a `filter_stats` block with a per-target / per-filter drop breakdown so you can see exactly how many records each filter removed.

### Worked examples

**(a) Keep only multi-step trajectories** (applies to the default set: sft, grpo, kto_positive; negatives untouched):

```yaml
projection:
  filters:
    - field: metadata.environment.episode_trace.total_turns
      op: gte
      value: 5
```

**(b) Scope a quality-score floor to SFT only** (KTO/GRPO targets unaffected):

```yaml
projection:
  filters:
    - field: metadata.final_judge.score
      op: gte
      value: 0.8
      applies_to: [sft]
      on_missing: drop          # SFT rows without a judge score are excluded
```

**(c) Explicitly filter a stop_reason on hard negatives** (opt-in; this is the only way kto_negative is filtered):

```yaml
projection:
  filters:
    - field: metadata.environment.episode_trace.stop_reason
      op: ne
      value: max_tool_steps_exceeded
      applies_to: [kto_negative]
      on_missing: drop
```

---

## 2. Full-Trajectory SFT Projection (`--sft-output`)

The projector has an **optional** `--sft-output <path>`. When provided, each quality-positive rollout is projected into a single FULL multi-turn SFT row:

```json
{
  "conversations": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": null, "tool_calls": [...]},
    {"role": "tool", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "label": true,
  "total_turns": 5
}
```

Behavior:

- It only projects **quality positives** — failed/negative rollouts are skipped for this output.
- It reconstructs the **whole episode** from `conversation_trace`, dropping the `judge_feedback`, `validation_feedback`, and `final_text_request` scaffolding turns (these are environment scaffolding, not training signal).
- Tool results become `tool`-role messages on disk; assistant tool-call turns carry `tool_calls`.
- Schema: `Datasets/environment_rollouts/sft_projection.schema.json`.

### Chat-template behavior

`shared/sft_preprocessing.py` renders the `tool` role to **text** (re-tagged consistent with how the environment presents tool output to the model) at template time. Consequences:

- **Existing single-turn SFT data is unaffected** — rows that use only system/user/assistant roles template identically to before.
- **Multi-turn rows template into a coherent transcript** — the on-disk `tool` role is rendered into the house ChatML-style text so a chat template that only understands system/user/assistant still produces a sensible transcript.
- The on-disk `tool` role is **preserved for auditing** — only the template-time rendering flattens it to text; the saved JSONL keeps the structured `tool` messages.

### Full invocation example (all four outputs + filter config)

```bash
python SynthChat/scripts/project_rollout_datasets.py \
  --input Datasets/environment_rollouts/run_a.jsonl \
  --input Datasets/environment_rollouts/run_b.jsonl \
  --canonical-output Datasets/environment_rollouts/canonical.jsonl \
  --kto-output Datasets/environment_rollouts/kto.jsonl \
  --grpo-output Datasets/environment_rollouts/grpo.jsonl \
  --sft-output Datasets/environment_rollouts/sft.jsonl \
  --filter-config configs/projection/keep_multistep.yaml
```

**Omitting `--sft-output` keeps the original 3-output behavior** (canonical/KTO/GRPO only). `--filter-config` is independent and optional in either mode.

### Flywheel staging filters

The same generic filter engine (section 1) is wired into the **flywheel** `DatasetStager` (`shared/flywheel/stager.py`), so you can declaratively filter which inference logs get staged into SFT/KTO/GRPO training sets. It is the **same engine, different field namespace**: flywheel filters evaluate against an inference-log record, not an environment rollout.

**Filter view (the dot-path namespace).** Each record is exposed as a "filter view" dict with two layers:

- **Record fields at the top level** (from `InferenceLogRecord`): `log_id`, `timestamp`, `model_id`, `adapter_name`, `temperature`, `max_tokens`, `tools_requested`, `finish_reason`, `prompt_tokens`, `completion_tokens`, `latency_ms`, `fitness_score`, `is_valid`, `tag`, `dataset_version`, `source_file`, `line_number`, `tenant_id`.
- **Parsed log content under a `content.` prefix**: e.g. `content.messages`, `content.response_content`, `content.tool_calls`, `content.completion_token_ids`.

So a researcher can write `field: fitness_score` (record field) or `field: content.response_content` (transcript). For records read from the catalog index, prefer the `content.*` view for transcript data — only index-backed fields are populated at the top level.

**Config key.** Filters live under the top-level `filters:` key of the flywheel config YAML (loaded into `FlywheelConfig.filters`), using the same spec shape as section 1:

```yaml
# configs/flywheel/<your>.yaml
filters:
  - field: fitness_score
    op: gte
    value: 0.85
```

Empty/absent `filters:` is a no-op — default staging behavior is unchanged.

**Default targets.** The stager's targets are `sft`, `kto_positive`, `kto_negative`, `grpo`. The **default target set** (when `applies_to` is omitted) is `sft, kto_positive, grpo` — it **excludes `kto_negative`**, matching the projector convention: quality filters must not silently drop hard negatives unless a researcher explicitly lists `kto_negative` in `applies_to`. KTO interleaving (per `KTO_TRAINING_REFERENCE.md`) is preserved — filtering happens before the positive/negative lists are built. The applied filter specs and a per-target drop breakdown are recorded in the staged `DatasetVersion.filter_criteria` (keys `staging_filters` and `staging_filter_stats`).

---

## 3. Data-Recipe Guidance

Heuristics for planning generation runs, drawn from OpenThoughts-Agent (arXiv 2606.24855). These are **planning heuristics, NOT codified behavior** in this repo — use them to decide what to generate, not as automatic pipeline steps.

- **Task source is the highest-leverage knob.** Swapping/mixing task sources (scenarios) swings results the most. Prefer mixing the top ~4 task sources over over-specializing on a single one; adding many weak sources hurts.
- **Strongest model ≠ best teacher.** Before committing a big generation run, run a small teacher bake-off across candidate generation models. The SOTA benchmark model is often NOT the best teacher (~5pp swings observed). Use the per-stage model split in scenario YAML to test generation models cheaply.
- **Difficulty ≈ teacher response length.** When selecting which tasks to keep, prefer tasks where a strong model produces longer responses — a free difficulty proxy worth roughly +3pp. Do **NOT** use embedding-diversity to filter which tasks to keep; in the paper it underperformed random selection.
- **Keep multi-step trajectories.** Longer agentic trajectories (e.g. `total_turns >= 5`) are higher-quality SFT signal even at a matched token budget. Enforce it with the `total_turns` filter from section 1.
  - **Caveat for THIS repo:** today's scenarios are mostly single-step (turn counts cluster at 1-3), so the real lever is **authoring genuinely multi-step tasks first** — the filter only helps once such data actually exists.
- **Augmentation caution.** LLM task-hardening/constraining rewriting was within noise. What scales past the upsampling plateau is **expanding surface forms (paraphrases) of GOOD sources**, not making tasks artificially harder.
