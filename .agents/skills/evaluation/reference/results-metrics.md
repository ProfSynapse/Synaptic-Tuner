# Results & Metrics Reference

Understanding evaluation output, metrics, and failure details.

---

## Output Formats

### Live Console Output

```text
Running 27 evaluations...

  PASS  storageManager_copy (6.78s)
  FAIL  memoryManager_updateWorkspace (4.82s)
         Model called: useTools
         Expected: flat_memory_update_workspace
```

`Expected` is the name of the configured `correct.any` path, not a hardcoded tool id.

### JSON Results

Results are written to `Evaluator/results/*.json`.

```json
{
  "metadata": {
    "backend": "vllm",
    "model": "finetuned",
    "temperature": 0.0,
    "scenarios": ["tool_prompts.yaml"],
    "preset": null,
    "model_artifact": {
      "model": "gguf/model-Q4_K_M.gguf",
      "backend": "llamacpp",
      "quantization": "Q4_K_M",
      "quantization_source": "detected"
    },
    "generated_at": "2026-04-24T14:55:00Z"
  },
  "summary": {
    "total": 27,
    "passed": 25,
    "failed": 2,
    "pass_rate": 0.926,
    "schema_passed": 27,
    "schema_pass_rate": 1.0,
    "correctness_tested": 27,
    "correctness_passed": 25,
    "correctness_pass_rate": 0.926,
    "environment_tested": 0,
    "environment_passed": 0,
    "environment_pass_rate": 0.0,
    "by_tag": {}
  },
  "records": []
}
```

### Markdown Report

Markdown reports summarize pass/fail counts, tag breakdowns, and failures:

```markdown
# Evaluation: finetuned

- **Passed:** 25/27 (92.6%)
- **Failed:** 2

## Results by Category
| Category | Passed | Total | Rate |
|----------|--------|-------|------|
| storageManager | 6/6 | 100.0% |
```

---

## Metrics Explained

| Metric | What It Measures |
|--------|------------------|
| `pass_rate` | Overall PASS / total |
| `correctness_pass_rate` | Fraction of cases where a configured `correct` path matched |
| `schema_pass_rate` | Structural parser/schema success rate; useful debugging signal, not the task contract |
| `environment_pass_rate` | Runtime execution/assertion success when `--env-backend` is enabled |
| `judge_pass_rate` | Judge success when `--judge` is enabled |
| `scoring_tested` | Count of cases with optional scoring config |
| `average_score` | Average configured score across scored cases |
| `by_tag` | Pass/fail breakdown for each tag |

`correctness_pass_rate` is the primary task-quality metric for assertion-driven scenarios.

---

## Per-Test Record

Important fields in each record:

| Field | Description |
|-------|-------------|
| `case_id` | Test identifier |
| `question` | User prompt |
| `tags` | Tags from YAML |
| `passed` | Overall pass/fail |
| `correctness_passed` | Whether a configured correctness path matched |
| `correctness` | Detailed assertion results for every path |
| `schema_passed` | Structural validation status |
| `validator` | Parsed tool calls and structural issues |
| `environment_passed` | Optional runtime validation result |
| `judge_passed` | Optional LLM judge result |
| `response_text` | Full model response |
| `raw_response` | Backend raw response when available |
| `conversation_trace` | Prompt/response trace |

### Correctness Detail

Failed assertions show expected, actual, path, and message:

```json
{
  "correctness": {
    "passed": false,
    "matched_path": null,
    "paths": [
      {
        "name": "flat_prompt_archive_prompt",
        "passed": false,
        "assertions": [
          {
            "type": "jsonpath_regex",
            "path": "$.tool_calls[0].arguments.tool",
            "expected": "^prompt archive-prompt\\b(?=.*QA Prototype)",
            "actual": "prompt archive-prompt \"agent_1732300800004_qa_prototype\"",
            "message": "expected regex ..."
          }
        ]
      }
    ]
  }
}
```

Use this section to decide whether the model failed, or the YAML should allow an additional schema-valid form.

---

## Status Semantics

| Status | Meaning |
|--------|---------|
| **PASS** | A configured `correct` path matched and optional environment/judge checks did not fail |
| **FAIL** | No `correct` path matched, backend errored, or optional environment/judge checks failed |
If behavior matters, express it as assertions or use a judge rubric.

---

## Comparing Models

Run the same scenarios with the same settings, then compare the results files
with `python -m Evaluator.compare`. It joins records on `case_id` and reports,
per candidate, pass-rate and correctness deltas (percentage points), per-tag
deltas, per-case flips (regressions pass to fail, improvements fail to pass),
request errors, latency ratio, and an exact two-sided McNemar p-value over the
discordant cases.

```bash
python -m Evaluator.cli --backend vllm --model base \
  --scenario tool_prompts.yaml --temperature 0 \
  --output Evaluator/results/base_tools.json

python -m Evaluator.cli --backend vllm --model finetuned \
  --scenario tool_prompts.yaml --temperature 0 \
  --output Evaluator/results/finetuned_tools.json

python -m Evaluator.compare \
  --reference Evaluator/results/base_tools.json \
  --candidate finetuned=Evaluator/results/finetuned_tools.json \
  --markdown Evaluator/results/base_vs_finetuned.md
```

Reading the report:

- `dPass` / `dCorrect` are candidate minus reference over the shared case ids.
- `Reg` / `Imp` list the flipped case ids; start debugging from `Reg`.
- `McNemar p` asks whether the flips are consistent with noise. On small
  suites a one or two case swing gives p near 1; do not read it as a real
  regression. At least six one-directional flips (6 vs 0) are needed for p < 0.05.
- A candidate marked `(!)` is not comparable: case id sets differ, a recorded
  setting differs (`temperature`, `top_p`, `max_tokens`, `seed`, `scenarios`,
  `preset`, `tags_filter`, `limit`, `dry_run`, `environment`), or a file is a
  partial/dry-run result. Settings missing from older results files are not
  compared.

### Gating

Thresholds turn the report into a gate. With no thresholds the command only
reports and exits 0.

| Flag | Breach when |
|------|-------------|
| `--max-pass-rate-drop PP` | Overall pass rate drops more than PP percentage points |
| `--max-correctness-drop PP` | Correctness pass rate drops more than PP points |
| `--max-tag-drop PP` | Any tag's pass rate drops more than PP points |
| `--min-tag-cases N` | (modifier) only gate tags with at least N shared cases |
| `--max-regressions N` | More than N cases flip pass to fail |
| `--alpha A` | (modifier) drops only count when McNemar p < A |
| `--allow-mismatch` | (modifier) do not fail non-comparable candidates |

Exit codes: `0` all candidates pass (or report-only), `1` a gate breached, `2`
input error (missing/invalid results file, duplicate labels). `--output` writes
the comparison JSON (`schema_version: synaptic-evaluation-comparison/v1`) and
`--markdown` a report with per-tag tables and flipped case ids.

### Quantization Regression Check

Measure what each GGUF quant costs against the full-precision model, on the
same scenarios and generation settings.

1. Reference: evaluate a full-precision artifact. Pick one of:
   - the base `F16`/`BF16` GGUF via `--backend llamacpp` (same runtime as the
     quants, so only quantization differs; preferred when the converter kept it);
   - the `merged-16bit` model served by vLLM (`--backend vllm`);
   - the LoRA `final_model/` via `--backend unsloth --no-load-in-4bit`. The
     unsloth backend loads 4-bit by default, which would hide part of the cost.
2. Candidates: evaluate each GGUF with `--backend llamacpp`. The quant label is
   detected from the file name (`model-Q4_K_M.gguf`); pass `--quantization` to
   set it explicitly and `--artifact-manifest gguf/gguf_manifest.json` to record
   the file's `sha256`, `imatrix_used` and the calibration block.
3. Compare with thresholds.

```bash
COMMON="--scenario tool_prompts.yaml --temperature 0 --max-tokens 768 --seed 7 --no-dashboard"

python -m Evaluator.cli --backend llamacpp --model gguf/model.gguf $COMMON \
  --quantization F16 --artifact-manifest gguf/gguf_manifest.json \
  --output Evaluator/results/quant_f16.json

for Q in Q8_0 Q5_K_M Q4_K_M; do
  python -m Evaluator.cli --backend llamacpp --model gguf/model-$Q.gguf $COMMON \
    --artifact-manifest gguf/gguf_manifest.json \
    --output Evaluator/results/quant_$Q.json
done

python -m Evaluator.compare \
  --reference Evaluator/results/quant_f16.json \
  --candidate Evaluator/results/quant_Q8_0.json \
  --candidate Evaluator/results/quant_Q5_K_M.json \
  --candidate Evaluator/results/quant_Q4_K_M.json \
  --max-pass-rate-drop 3 --max-tag-drop 10 --min-tag-cases 5 --alpha 0.05 \
  --output Evaluator/results/quant_compare.json \
  --markdown Evaluator/results/quant_compare.md
```

The compare command warns when the reference itself was loaded 4-bit or is a
quantized artifact, and when an older unsloth reference did not record its load
precision.

---

## Using Results For Iteration

- If `correctness` failed and the model output violates the schema or prompt, add training examples.
- If `correctness` failed but the output is schema-valid and acceptable, add another `correct.any` path.
- If `schema_passed` is false but `correctness` is true, inspect structural parsing but do not treat schema diagnostics as task truth.
- If `environment` failed, inspect runtime execution traces and environment assertions.
- If judge failed, inspect the judge output and rubric before turning it into training data.

---

## Evaluation Lineage

Use `--lineage` to create a structured provenance file:

```bash
python -m Evaluator.cli --backend unsloth --model path/to/model \
  --scenario tool_prompts.yaml \
  --lineage eval_lineage.json
```

This JSON contains model info, scenario files, summary results, and timestamps.

`metadata.model_artifact` in the results JSON, and `model_artifact` in the
lineage, describe what was evaluated:

| Field | Meaning |
|-------|---------|
| `quantization` | llama.cpp quant label (`Q4_K_M`, `Q8_0`, `IQ2_XS`, `F16`, ...) or null |
| `quantization_source` | `flag` (`--quantization`), `manifest` (`quant_type` from `--artifact-manifest`), `detected` (from the model path/name), or null |
| `load_settings` | Backend load settings; for `unsloth`, `load_in_4bit` and `max_seq_length` |
| `manifest` | With `--artifact-manifest`: the matching `file` entry (`filename`, `quant_type`, `sha256`, `imatrix_used`, ...), the `calibration` block and scalar `header` fields |
| `warnings` | Non-fatal problems, e.g. no manifest entry matched the model file name |
