# Evaluator Quick Start

Simple guide to running evaluations on your trained models.

---

## Basic Usage

```bash
python -m Evaluator.cli --model <your-model-name> --prompt-set <prompt-file>
```

---

## Available Prompt Sets

### 1. **Behavior Rubric Tests** (Recommended for trained models)
Tests all 6 behavior patterns from the training rubric.

```bash
python -m Evaluator.cli \
  --model your-model-name \
  --prompt-set Evaluator/prompts/behavior_prompts.json \
  --output Evaluator/results/behavior_test.json \
  --markdown Evaluator/results/behavior_report.md
```

**43 prompts testing:**
- Intellectual Humility (8)
- Verification Before Action (6)
- Context Continuity (7)
- Strategic Tool Selection (6)
- Error Recovery (4)
- Workspace Awareness (3)
- Multi-Behavior (4)
- Anti-Patterns (5)

### 2. **Baseline Tests** (Quick functionality check)
6 general prompts with behavior expectations.

```bash
python -m Evaluator.cli \
  --model your-model-name \
  --prompt-set Evaluator/prompts/baseline.json
```

### 3. **Full Tool Coverage** (Every tool tested once)
45 prompts covering all tools.

```bash
python -m Evaluator.cli \
  --model your-model-name \
  --prompt-set Evaluator/prompts/tool_prompts.json
```

### 4. **Multi-Step Workflows** (Complex sequences)
7 prompts testing tool chaining.

```bash
python -m Evaluator.cli \
  --model your-model-name \
  --prompt-set Evaluator/prompts/tool_combos.json
```

---

## Common Options

### Filter by Tags
```bash
python -m Evaluator.cli \
  --model your-model \
  --prompt-set Evaluator/prompts/behavior_prompts.json \
  --tags intellectual_humility,verification
```

### Limit Number of Prompts
```bash
python -m Evaluator.cli \
  --model your-model \
  --prompt-set Evaluator/prompts/behavior_prompts.json \
  --limit 10
```

### Use LM Studio Instead of Ollama
```bash
python -m Evaluator.cli \
  --backend lmstudio \
  --model your-model \
  --prompt-set Evaluator/prompts/behavior_prompts.json
```

### Save Results with Markdown Report
```bash
python -m Evaluator.cli \
  --model your-model \
  --prompt-set Evaluator/prompts/behavior_prompts.json \
  --output results/test.json \
  --markdown results/report.md
```

### Dry Run (Test Setup Without API Calls)
```bash
python -m Evaluator.cli \
  --model your-model \
  --prompt-set Evaluator/prompts/behavior_prompts.json \
  --dry-run
```

---

## Quick Test Examples

### Test a Specific Behavior
```bash
# Only test intellectual humility prompts
python -m Evaluator.cli \
  --model your-model \
  --prompt-set Evaluator/prompts/behavior_prompts.json \
  --tags intellectual_humility

# Only test verification patterns
python -m Evaluator.cli \
  --model your-model \
  --prompt-set Evaluator/prompts/behavior_prompts.json \
  --tags verification_before_action
```

### Compare Models
```bash
# Base model
python -m Evaluator.cli \
  --model mistral-7b-base \
  --prompt-set Evaluator/prompts/behavior_prompts.json \
  --output results/base_model.json

# Your SFT model
python -m Evaluator.cli \
  --model your-sft-model \
  --prompt-set Evaluator/prompts/behavior_prompts.json \
  --output results/sft_model.json

# Your KTO model
python -m Evaluator.cli \
  --model your-kto-model \
  --prompt-set Evaluator/prompts/behavior_prompts.json \
  --output results/kto_model.json
```

### Quick Smoke Test
```bash
# Test first 5 prompts only
python -m Evaluator.cli \
  --model your-model \
  --prompt-set Evaluator/prompts/baseline.json \
  --limit 5
```

---

## Understanding Results

### JSON Output
Results saved to `--output` path (default: `Evaluator/results/run_<timestamp>.json`)

Contains:
- Model responses
- Tool calls made
- Validation results
- Metadata (model, settings, timing)

### Markdown Report
Optional `--markdown` generates human-readable report with:
- Summary statistics
- Pass/fail for each prompt
- Tool usage breakdown
- Response examples

### Console Output
Shows real-time summary:
```
Evaluation Summary:
  Total:   43
  Passed:  38
  Failed:  5
  Errors:  0
```

### Exit Codes
- `0` - All tests passed
- `2` - Some tests failed (validation)
- `3` - Request errors occurred

---

## Tips

### Make Sure Model is Running
**Ollama:**
```bash
ollama list  # Check if model exists
ollama run your-model  # Start model
```

**LM Studio:**
- Start LM Studio
- Load your model
- Start local server (default port 1234)

### Default Settings
If you don't specify, defaults are:
- Backend: `ollama`
- Host: `127.0.0.1`
- Port: `11434` (Ollama) or `1234` (LM Studio)
- Temperature: `0.2`
- Top-p: `0.9`
- Max tokens: `1024`
- Prompt set: `Evaluator/prompts/baseline.json`

### Custom Backend Settings
```bash
python -m Evaluator.cli \
  --model your-model \
  --host 192.168.1.100 \
  --port 8080 \
  --temperature 0.1 \
  --max-tokens 2048 \
  --prompt-set Evaluator/prompts/behavior_prompts.json
```

---

## Full Evaluation Suite

Run all prompt sets for comprehensive testing:

```bash
# 1. Behavior patterns (43 prompts)
python -m Evaluator.cli \
  --model your-model \
  --prompt-set Evaluator/prompts/behavior_prompts.json \
  --output results/behavior.json \
  --markdown results/behavior.md

# 2. Tool coverage (45 prompts)
python -m Evaluator.cli \
  --model your-model \
  --prompt-set Evaluator/prompts/tool_prompts.json \
  --output results/coverage.json

# 3. Multi-step workflows (7 prompts)
python -m Evaluator.cli \
  --model your-model \
  --prompt-set Evaluator/prompts/tool_combos.json \
  --output results/combos.json

# 4. Baseline functionality (6 prompts)
python -m Evaluator.cli \
  --model your-model \
  --prompt-set Evaluator/prompts/baseline.json \
  --output results/baseline.json

# Total: 101 prompts
```

---

## Troubleshooting

### "Connection refused" error
- Make sure Ollama/LM Studio is running
- Check the port is correct
- Try `--host 127.0.0.1` explicitly

### "Model not found" error
- Verify model name: `ollama list` or check LM Studio
- Model name is case-sensitive

### "No prompts matched" error
- Check `--tags` filter is valid
- Check `--limit` isn't 0
- Verify prompt set file exists

### Slow responses
- Increase `--timeout` (default: 60s)
- Reduce `--max-tokens` (default: 1024)
- Check model size (smaller = faster)

---

## Next Steps

1. **Run baseline test** to verify setup works
2. **Run behavior tests** to measure quality
3. **Review markdown report** to identify issues
4. **Compare before/after** training results
5. **See BEHAVIOR_EVALUATION_GUIDE.md** for detailed scoring

---

## Examples by Use Case

### "I just trained a model and want to test it"
```bash
python -m Evaluator.cli \
  --model my-new-model \
  --prompt-set Evaluator/prompts/behavior_prompts.json \
  --markdown results/my_test.md
```

### "I want to see if my model uses tools correctly"
```bash
python -m Evaluator.cli \
  --model my-model \
  --prompt-set Evaluator/prompts/tool_prompts.json \
  --limit 20
```

### "I want to test one specific behavior"
```bash
python -m Evaluator.cli \
  --model my-model \
  --prompt-set Evaluator/prompts/behavior_prompts.json \
  --tags intellectual_humility \
  --markdown results/humility_test.md
```

### "I want to compare two models"
```bash
python -m Evaluator.cli --model model-a --prompt-set Evaluator/prompts/behavior_prompts.json --output results/model_a.json
python -m Evaluator.cli --model model-b --prompt-set Evaluator/prompts/behavior_prompts.json --output results/model_b.json
# Then manually compare the JSON files
```

---

**Quick Command Reference:**

```bash
# Minimal
python -m Evaluator.cli --model <name> --prompt-set <file>

# Typical
python -m Evaluator.cli --model <name> --prompt-set <file> --output <json> --markdown <md>

# Full
python -m Evaluator.cli \
  --backend ollama \
  --model <name> \
  --prompt-set <file> \
  --tags <comma-separated> \
  --limit <number> \
  --temperature 0.2 \
  --output <json> \
  --markdown <md>
```
