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

Run the same scenario and compare `summary.correctness_pass_rate`:

```bash
python -m Evaluator.cli --backend vllm --model base \
  --scenario tool_prompts.yaml \
  --output Evaluator/results/base_tools.json

python -m Evaluator.cli --backend vllm --model finetuned \
  --scenario tool_prompts.yaml \
  --output Evaluator/results/finetuned_tools.json
```

```bash
python -c "
import json
base = json.load(open('Evaluator/results/base_tools.json'))
ft = json.load(open('Evaluator/results/finetuned_tools.json'))
print(f'Base correctness:      {base[\"summary\"][\"correctness_pass_rate\"]:.1%}')
print(f'Finetuned correctness: {ft[\"summary\"][\"correctness_pass_rate\"]:.1%}')
"
```

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
