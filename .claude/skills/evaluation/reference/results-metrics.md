# Results & Metrics Reference

Understanding evaluation output, metrics, and how to compare models.

---

## Output Formats

### Live Console Output

```
Running 51 evaluations...

  PASS  IH_ambiguous_deletion (2.34s)
  FAIL  SM_move_note (1.82s)
         Model called: storageManager_archive
         Expected: storageManager_move
  WARN  IH_unclear_request (1.56s)
         Why: Should have asked for clarification
```

### JSON Results (`Evaluator/results/run_TIMESTAMP.json`)

```json
{
  "metadata": {
    "backend": "lmstudio",
    "model": "qwen2.5-7b-instruct",
    "temperature": 0.2,
    "generated_at": "2025-02-14T10:30:00Z"
  },
  "summary": {
    "total": 51,
    "passed": 42,
    "warned": 3,
    "failed": 6,
    "pass_rate": 0.82,
    "schema_passed": 45,
    "schema_pass_rate": 0.88,
    "behavior_tested": 51,
    "behavior_passed": 45,
    "behavior_pass_rate": 0.88,
    "environment_tested": 40,
    "environment_passed": 37,
    "environment_pass_rate": 0.925,
    "by_tag": { ... },
    "top_failure_reasons": [ ... ],
    "top_environment_failures": [ ... ]
  },
  "records": [ ... ]
}
```

### Markdown Report

Auto-generated summary with tables:

```markdown
# Evaluation: qwen2.5-7b-instruct

- **Passed:** 42/51 (82.4%)
- **Failed:** 6
- **Behavior tests:** 45/51 (88.2%)

## Results by Category
| Category | Passed | Total | Rate |
|----------|--------|-------|------|
| intellectual_humility | 8/10 | 80% |
| storageManager | 12/15 | 80% |
```

---

## Metrics Explained

### Overall Metrics

| Metric | What It Measures |
|--------|-----------------|
| `pass_rate` | Percentage of tests fully passed (PASS / total) |
| `schema_pass_rate` | Correct tool selection rate |
| `behavior_pass_rate` | Behavioral expectations met rate |
| `environment_pass_rate` | Runtime execution + assertion pass rate |
| `total` | Number of tests run |
| `passed` / `warned` / `failed` | Count per status |

### Per-Test Record

| Field | Description |
|-------|-------------|
| `case_id` | Test identifier |
| `passed` | Overall pass (schema + behavior) |
| `schema_passed` | Correct tool called? |
| `behavior_passed` | Behavioral expectations met? |
| `environment_passed` | Runtime validation passed? |
| `latency_s` | Response time in seconds |
| `response_text` | Full model response |
| `validator` | Schema validation details |
| `behavior` | Behavior validation details |
| `environment` | Runtime execution trace + assertion issues |

### Per-Tag Breakdown

`summary.by_tag` gives pass/warn/fail counts per tag:
```json
{
  "intellectual_humility": {"passed": 8, "warned": 1, "failed": 1},
  "storageManager": {"passed": 12, "warned": 0, "failed": 3}
}
```

Use this to identify which capabilities need improvement.

---

## Status Semantics

| Status | Schema | Behavior | Interpretation |
|--------|--------|----------|----------------|
| **PASS** | Correct | Met | Model is working correctly |
| **WARN** | Correct | Not met | Right tool but suboptimal behavior |
| **FAIL** | Wrong or runtime fail | N/A | Wrong/missing tool call or environment assertions failed |

**WARN is valuable** — it means the model's tool selection is correct but its behavior (explaining, asking, reasoning) needs refinement. This is exactly what KTO training addresses.

---

## Comparing Models

### Manual Comparison

```bash
# Run both evaluations
python -m Evaluator.cli --backend lmstudio --model base-model \
  --output Evaluator/results/base.json
python -m Evaluator.cli --backend unsloth --model finetuned-lora \
  --output Evaluator/results/finetuned.json

# Compare pass rates
python -c "
import json
base = json.load(open('Evaluator/results/base.json'))
ft = json.load(open('Evaluator/results/finetuned.json'))
print(f'Base:      {base[\"summary\"][\"pass_rate\"]:.1%}')
print(f'Finetuned: {ft[\"summary\"][\"pass_rate\"]:.1%}')
"
```

### Tracking Progress Across Checkpoints

```bash
for ckpt in checkpoint-100 checkpoint-200 checkpoint-300; do
  python -m Evaluator.cli --backend unsloth \
    --model training_output/$ckpt \
    --scenario behavior_prompts.yaml \
    --output Evaluator/results/$ckpt.json
done
```

### Key Comparison Points

1. **Overall pass_rate** — Primary metric
2. **schema_pass_rate** — Tool selection accuracy (SFT focus)
3. **behavior_pass_rate** — Behavioral quality (KTO focus)
4. **by_tag breakdown** — Which capabilities improved/regressed
5. **latency** — Inference speed impact
6. **top_failure_reasons** — What to address next

---

## Using Results for Training

### WARN → KTO Negative Examples

Tests with WARN status (correct tool, bad behavior) are ideal KTO negatives:
- Model output → `label: false`
- Expected behavior → write a `label: true` example

### FAIL → Identify Training Gaps

Failed tests reveal:
- Missing tool capabilities → need more SFT examples
- Wrong tool selection → need more diverse SFT scenarios
- Format issues → check dataset format alignment

### Coverage Gaps

If `by_tag` shows 0 tests for a capability, add new test scenarios to cover it.

---

## Evaluation Lineage

Use `--lineage` to create a structured provenance file:

```bash
python -m Evaluator.cli --backend unsloth --model path/to/model \
  --lineage eval_lineage.json
```

This JSON contains:
- Model info (name, path, backend)
- Scenario files used
- Summary results
- Timestamp
- Can be uploaded to HuggingFace with `--upload-to-hf`
