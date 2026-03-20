# Enterprise Data Flywheel — Preparation Research

> Prepared by: pact-preparer | Date: 2026-03-20
> Plan reference: `docs/plans/enterprise-data-flywheel-plan.md` (APPROVED)

---

## Executive Summary

The approved plan is well-grounded in existing codebase components. Research confirms:

1. **FitnessEvaluator** (`shared/validation/fitness.py`) is fully reusable for the tagger with **no GPU or model-loading dependencies** — it runs pure CPU string parsing and validation. The 0.8/0.3 thresholds from the plan are reasonable starting points but need empirical calibration.
2. **RunRegistry** (`shared/experiment_tracking/`) supports new `run_type` values (`inference_log_batch`, `flywheel_cycle`) with **zero schema changes** — `run_type` is a free-form string field.
3. **vLLM LoRA hot-swap API** is well-documented with exact endpoints and payloads (see Section 4).
4. **Most required packages are already installed** — only `asyncpg`, `hypothesis`, and `watchdog` are missing from the current environment.
5. **No blockers identified.** Implementation can proceed.

---

## 1. Reusable Components Inventory

### 1.1 FitnessEvaluator (`shared/validation/fitness.py`)

**Class**: `FitnessEvaluator` (line 50)
**Factory**: `create_fitness_evaluator()` (line 228)
**Result type**: `FitnessResult` dataclass (line 21)

**What it does**: Config-driven fitness scoring pipeline:
1. **Layer 1 — Parse**: Calls `parse_response(model_output)` from `shared/validation/parsing/` which auto-detects format (Qwen `<tool_call>`, Mistral `[TOOL_CALLS]`, ChatML `tool_call:`, OpenAI dict) and normalizes to `ParsedResponse`.
2. **Layer 2 — Validate**: Runs `StructureValidator.validate()` against config-driven validation rules (XML, JSON, YAML, regex, code validators from `shared/validation/validators/`).
3. **Layer 3 — Score**: Computes 0.0-1.0 score via configurable method (`binary`, `error_count`, `error_penalty`).

**Batch support**: `evaluate_batch()` (line 151) processes lists of outputs — directly usable in the flywheel pipeline.

### 1.2 RunRegistry (`shared/experiment_tracking/registry.py`)

**Class**: `RunRegistry` (line 39)
**Schema**: `RunRecord` dataclass (line 25 in `schema.py`)
**Storage**: Append-only JSONL at `.tracking/registry.jsonl`
**Linkage**: Separate `links.jsonl` for parent/child relationships

**Key APIs for flywheel**:
- `register_run(record)` -> appends RunRecord (idempotent on `output_dir`)
- `link_runs(child_run_id, parent_run_id, relationship)` -> records lineage
- `find_runs(RunFilter)` -> query by type, status, tags, date range
- `get_linked_runs(run_id)` -> find related runs

### 1.3 InteractionLogger (`shared/judge/interaction_logger.py`)

**Class**: `InteractionLogger` (line 21)
**Pattern**: Thread-safe JSONL writer with `threading.Lock`, ChatML-compatible format, KTO labels.

**Relevance to flywheel**: The `inference_logger.py` should follow this same pattern:
- Thread-safe writes via lock
- JSONL append format
- Configurable output directory
- `enabled` toggle for zero-overhead disable
- Timestamped filenames

### 1.4 BaseLLMClient (`shared/llm/base.py`)

**Abstract class**: `BaseLLMClient` with `chat()`, `structured_output()`, `test_connection()`, `list_models()`
**Providers**: `openrouter`, `lmstudio`, `ollama`, `unsloth` (via `shared/llm/factory.py`)
**Factory**: `create_client(provider, model, config)` in `shared/llm/factory.py`

**Flywheel integration point**: The plan calls for adding `enable_logging=True` to `create_client()` which wraps the returned client in a `LoggingLLMClient` decorator. The `chat()` method is the primary interception point — it accepts `messages`, `temperature`, `max_tokens`, `**kwargs` and returns `str`.

### 1.5 Existing Adapters (`shared/experiment_tracking/adapters.py`)

**Current adapters**: `sft_lineage_to_run_record`, `kto_lineage_to_run_record`, `ml_tracking_to_run_record`, `manifest_to_run_record`, `grpo_log_to_run_record`, `eval_to_run_record`, `register_grpo_run`.

**Flywheel needs**: Add `flywheel_cycle_to_run_record()` adapter following the same pattern. All adapters return `RunRecord` — no interface changes needed.

---

## 2. FitnessEvaluator Analysis

### 2.1 Scoring Logic Deep-Dive

**For tool-call responses** (when `parsed.has_tool_calls` is True):
1. Parsed tool calls are converted to OpenAI-style dicts (`_build_validation_data`, line 172)
2. `StructureValidator.validate()` runs config-driven rules against tool call names, arguments, text content
3. Score computed by selected method:

| Method | Logic | Score Range |
|--------|-------|-------------|
| `binary` | 1.0 if all validations pass, 0.0 otherwise | {0.0, 1.0} |
| `error_count` | `max(0.0, 1.0 - len(errors) / max_errors)` | 0.0 to 1.0 (linear) |
| `error_penalty` | `max(0.0, 1.0 - len(errors) * penalty)` | 0.0 to 1.0 (linear) |

With `error_count` (default, `max_errors=5`):
- 0 errors -> 1.0
- 1 error -> 0.8
- 2 errors -> 0.6
- 3 errors -> 0.4
- 4 errors -> 0.2
- 5+ errors -> 0.0

**For non-tool-call responses** (text-only, no `<tool_call>` / `[TOOL_CALLS]` / etc.):
- Immediately returns `no_tool_call_score` from config (default: **0.0**)
- `is_valid` = False
- Error: `"No tool calls found in response"`

### 2.2 Score Distribution Analysis for Flywheel Thresholds

The plan proposes: score >= 0.8 -> SFT positive, 0.3-0.8 -> KTO negative, < 0.3 -> discard.

**Mapping to `error_count` scoring**:

| Errors | Score | Plan Classification |
|--------|-------|-------------------|
| 0 | 1.0 | SFT positive |
| 1 | 0.8 | SFT positive (borderline) |
| 2 | 0.6 | KTO negative |
| 3 | 0.4 | KTO negative |
| 4 | 0.2 | Discard |
| 5+ | 0.0 | Discard |
| No tool call | 0.0 | Discard |

**Assessment**: The thresholds are reasonable for a tool-calling focused pipeline:
- **0.8 threshold for SFT**: Allows at most 1 validation error. This is conservative — good for maintaining training data quality.
- **0.3 threshold for discard**: Discards responses with 4+ errors or no tool calls. Reasonable — these are too broken to learn from even as negative examples.
- **Middle tier (0.3-0.8)**: 2-3 error responses become KTO negative examples. These are "almost right" responses — ideal for teaching the model what NOT to do.

**Caveats**:
1. **Non-tool-call responses always score 0.0 by default** — they'll be discarded, not routed to KTO. If the model sometimes correctly responds with text-only (no tool call needed), those get discarded. The tagger should handle this case separately (check if `tools` were in the request).
2. **Score depends entirely on validation config** — the YAML config determines what gets validated. A minimal config (few rules) will produce mostly 1.0 scores. The flywheel config needs to be tuned to the specific tool-calling schema.
3. **`max_errors` parameter matters** — with `max_errors=5` (default), the score distribution is coarse (0.2 increments). Consider `max_errors=10` for finer granularity, or `error_penalty` with `penalty=0.05` per error.

### 2.3 Dependencies — Can FitnessEvaluator Run Without GPU?

**YES — no GPU or model-loading dependencies.**

Dependency chain:
```
FitnessEvaluator
  shared.utilities.load_yaml          -> PyYAML (CPU-only)
  shared.validation.parsing           -> Pure Python string parsing (regex, JSON)
    response_parser.py                -> ParsedResponse, ParsedToolCall
    tool_call_parser.py               -> Format detection (Qwen/Mistral/ChatML)
    utilities.py                      -> JSON fixup helpers
  shared.validation.validators        -> Config-driven validation
    structure_validator.py            -> Field/pattern/tool validation
    cross_scope_validator.py          -> Cross-section validation
    content/                          -> XML, JSON, YAML, regex, code validators
      All pure Python (lxml, json, yaml, re, ast)
```

**No imports from**: `torch`, `transformers`, `unsloth`, `vllm`, or any GPU library.
**No model loading**: All parsing is regex/string-based format detection.
**Safe for pipeline context**: Can run in a data processing pipeline, CI/CD, or background worker without GPU.

---

## 3. RunRegistry Extension Analysis

### 3.1 Can RunRegistry Support New Run Types Without Schema Changes?

**YES — zero schema changes needed.**

The `run_type` field on `RunRecord` (schema.py, line 41) is typed as `str` with no enum constraint:
```python
run_type: str  # "sft" | "kto" | "grpo" | "ml" | "evaluation" | "cloud_sft" | "cloud_kto" | "cloud_grpo"
```

The comment is documentary only — the code does not validate `run_type` against any whitelist. The `RunFilter.run_type` field (line 97) accepts `str | list[str] | None` and does simple string equality matching.

**New types to add**:
- `"inference_log_batch"` — batch of ingested inference logs
- `"flywheel_cycle"` — tagging + staging run

These will "just work" with the existing `register_run()`, `find_runs()`, and `link_runs()` methods.

### 3.2 Adapter Extension

A new `flywheel_cycle_to_run_record()` function is needed in `adapters.py`, following the exact same pattern as existing adapters (e.g., `grpo_log_to_run_record`). It should:
- Accept flywheel cycle metadata (dataset version, record counts, score thresholds)
- Return a `RunRecord` with `run_type="flywheel_cycle"`
- Populate `tags` with `dataset_version`, `sft_count`, `kto_count`, `grpo_count`
- Set `dataset_source` to the versioned dataset path

### 3.3 Linkage Chain

The flywheel creates a lineage chain via `link_runs()`:
```
inference_log_batch -> flywheel_cycle -> sft/kto/grpo training run -> evaluation
```

This uses the existing `links.jsonl` mechanism with no changes needed. The `get_linked_runs()` method traverses both directions.

---

## 4. vLLM API Reference for LoRA Serving + Hot-Swap

> Source: Prior research at `docs/preparation/vllm-vs-sglang-inference-serving-research.md` and official vLLM documentation.

### 4.1 Server Startup

```bash
# Start vLLM with LoRA support
python -m vllm.entrypoints.openai.api_server \
  --model org/base-model \
  --enable-lora \
  --max-loras 4 \
  --max-lora-rank 64 \
  --lora-extra-vocab-size 256 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --port 8000

# Required env var for runtime LoRA updates
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
```

### 4.2 Inference with LoRA Adapter

```bash
# Use a LoRA adapter by specifying its name as the model
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "user_v2_adapter",
    "messages": [{"role": "user", "content": "Hello"}],
    "temperature": 0.7,
    "max_tokens": 1024
  }'
```

### 4.3 Dynamic LoRA Loading

**Load a new adapter**:
```
POST /v1/load_lora_adapter
{
  "lora_name": "user_v2",
  "lora_path": "/path/to/adapter/directory"
}
Response: 200 OK (non-blocking — inference continues during load)
```

**Hot-swap (replace adapter weights in-place)**:
```
POST /v1/load_lora_adapter
{
  "lora_name": "user_v2",
  "lora_path": "/path/to/NEW/adapter/directory",
  "load_inplace": true
}
Response: 200 OK (atomically replaces weights, no name change needed)
```

**Unload an adapter**:
```
POST /v1/unload_lora_adapter
{
  "lora_name": "user_v2"
}
Response: 200 OK
```

### 4.4 Key Properties

| Property | Value |
|----------|-------|
| Load blocking? | No — inference continues during load |
| In-place replacement? | Yes — `load_inplace: true` |
| Auto-discovery? | Yes — filesystem resolver watches directory |
| Env var required | `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` |
| Memory per adapter | ~10-50MB (depends on rank + target modules) |
| LRU eviction | Yes — when `--max-loras` exceeded |

### 4.5 Tool Calling Configuration

```bash
# Enable auto tool calling with parser for target model family
python -m vllm.entrypoints.openai.api_server \
  --model org/base-model \
  --enable-auto-tool-choice \
  --tool-call-parser hermes  # or: mistral, llama3_json, qwen
```

---

## 5. Missing Dependencies Analysis

### 5.1 Already Installed

| Package | Version | Used For |
|---------|---------|----------|
| `aiosqlite` | 0.22.1 | SQLiteLogCatalog (local async SQLite) |
| `fastapi` | 0.122.0 | Logging proxy service |
| `uvicorn` | 0.38.0 | ASGI server for proxy |
| `vllm` | 0.11.2 | Model serving (optional, not imported by pipeline code) |

### 5.2 Not Installed — Need to Add

| Package | Purpose | Required For |
|---------|---------|-------------|
| `asyncpg` | Async Postgres client | `PostgresLogCatalog` (cloud backend) |
| `hypothesis` | Property-based testing | Test phase — dedup invariant tests |
| `watchdog` | Filesystem monitoring | Optional file watcher for new logs (can defer) |

### 5.3 Assessment

- **`asyncpg`**: Only needed for cloud deployment (`PostgresLogCatalog`). Can be an optional dependency. Not needed for v1 local-first implementation.
- **`hypothesis`**: Test-phase only. Install as dev dependency.
- **`watchdog`**: Optional. The CLI can use manual `tuner flywheel ingest` instead of automatic file watching for v1.

**Recommendation**: For v1 (local-first), only `hypothesis` is needed as a new dependency (test-only). `asyncpg` and `watchdog` can be deferred to optional extras.

---

## 6. Implementation Readiness Assessment

### 6.1 Blockers

**None identified.** All critical paths are clear:

| Area | Status | Notes |
|------|--------|-------|
| FitnessEvaluator reuse | Ready | No GPU deps, batch API available |
| RunRegistry extension | Ready | Free-form `run_type`, no schema changes |
| vLLM LoRA API | Ready | Well-documented, v0.11.2 installed |
| InteractionLogger pattern | Ready | Thread-safe JSONL pattern to follow |
| LLM client decoration | Ready | `BaseLLMClient.chat()` is the interception point |
| Dependencies | Ready | Core deps installed; only test/cloud extras missing |

### 6.2 Risks to Flag for Architect

1. **Non-tool-call response handling**: FitnessEvaluator scores text-only responses as 0.0 by default. The tagger needs logic to distinguish "correctly didn't use tools" from "failed to use tools" — check if `tools` were in the request payload.

2. **Scoring granularity**: With `error_count` default (`max_errors=5`), scores jump in 0.2 increments. Consider `error_penalty` with smaller penalty (e.g., 0.1) for finer-grained threshold routing.

3. **Config dependency**: FitnessEvaluator's usefulness depends entirely on the validation config YAML. A default `configs/flywheel/fitness.yaml` must ship with rules appropriate for the project's tool-calling schema. Without it, everything scores 1.0 (no validations = no errors = perfect score).

4. **Async interface cost**: The plan specifies async from day one (aiosqlite). This is architecturally correct for cloud-readiness, but adds complexity to the local-only v1. The `InteractionLogger` pattern (sync with threading.Lock) is simpler and proven in this codebase. Consider whether the proxy service truly needs async SQLite or if sync-with-threadpool is sufficient for v1 volumes.

### 6.3 Clarifications for Coding Phase

1. **GRPO reward signal**: Plan says tool call schema validation provides reward. The `FitnessEvaluator` already does this — the `FitnessResult.score` IS the reward signal. Coders should use `evaluate()` directly rather than reimplementing scoring.

2. **Dataset format**: Flywheel-produced JSONL must match the existing ChatML format (`conversations` array with `role`/`content`, optional `label` for KTO). See CLAUDE.md "Dataset Format (ChatML)" section.

3. **Idempotency**: `RunRegistry.register_run()` already has idempotency on `output_dir` (line 82). The flywheel stager should use the dataset version path as `output_dir` to get free dedup.

---

## 7. Prior Research Available

The following research documents already exist and should be referenced:

| Document | Path | Relevance |
|----------|------|-----------|
| vLLM vs SGLang comparison | `docs/preparation/vllm-vs-sglang-inference-serving-research.md` | Confirms vLLM choice, documents LoRA API, tool-calling stability |
| ML pipeline expansion | `docs/preparation/ml-pipeline-expansion-research.md` | Context on ML pipeline architecture |

---

## 8. Open Research Items (Deferred to Runtime)

These items from the plan's "Require Further Research" section cannot be fully resolved during PREPARE — they require runtime experimentation:

| Item | Status | Recommendation |
|------|--------|---------------|
| Inference volume estimate | Cannot determine without usage data | Design for configurable log rotation (daily default, 7-day retention) |
| Target model size (3B/7B/14B) | User-dependent | Design config to accept any; document 3090 constraints (7B FP16 or 14B INT4) |
| FitnessEvaluator score distribution calibration | Needs real inference data | Ship with 0.8/0.3 defaults; add `tuner flywheel calibrate` CLI to analyze score distribution on accumulated logs |
| vLLM LoRA hot-swap stability (Qwen/Mistral) | Needs live test | vLLM v0.11.2 supports both; test during CODE phase integration |
| Arena Learning applicability | v2 feature | Document as future enhancement; no v1 impact |

---

## References

- Plan: `docs/plans/enterprise-data-flywheel-plan.md`
- FitnessEvaluator: `shared/validation/fitness.py`
- RunRegistry: `shared/experiment_tracking/registry.py`
- RunRecord schema: `shared/experiment_tracking/schema.py`
- Adapters: `shared/experiment_tracking/adapters.py`
- InteractionLogger: `shared/judge/interaction_logger.py`
- LLM client base: `shared/llm/base.py`
- LLM factory: `shared/llm/factory.py`
- Response parsing: `shared/validation/parsing/__init__.py`
- Validators: `shared/validation/validators/__init__.py`
- vLLM research: `docs/preparation/vllm-vs-sglang-inference-serving-research.md`
