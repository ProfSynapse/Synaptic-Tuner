# Evaluator Summary

Quick reference for the evaluation system.

---

## Three Ways to Run Tests

### 1. Interactive Mode (Easiest) ⭐

```bash
python -m Evaluator
```

**You choose:**
1. Test suite (Behavior/Coverage/Baseline/Workflows/All)
2. Model (from LM Studio)
3. Number of runs

**Best for:** Quick testing, trying different test suites

---

### 2. Direct CLI (Most Control)

```bash
python -m Evaluator.cli \
  --model your-model \
  --prompt-set Evaluator/prompts/behavior_rubric.json
```

**Features:**
- Works with Ollama or LM Studio (`--backend`)
- Filter by tags (`--tags intellectual_humility`)
- Limit prompts (`--limit 10`)
- Custom output paths

**Best for:** Automation, specific testing needs

---

### 3. Batch Mode (Run Everything)

```bash
# In interactive mode, select option [5] Run All Tests
python -m Evaluator
# Then choose [5]
```

**Runs all 4 test suites:** 101 total prompts
- Behavior Rubric (43)
- Full Coverage (45)
- Baseline (6)
- Tool Combos (7)

**Best for:** Comprehensive model evaluation

---

## Test Suites

| Suite | Prompts | Purpose | When to Use |
|-------|---------|---------|-------------|
| **Behavior Rubric** | 43 | Tests all 6 behavior patterns | After training with rubric dataset |
| **Full Coverage** | 45 | One test per tool | Verify all tools work |
| **Baseline** | 6 | Quick functionality check | Smoke tests |
| **Tool Combos** | 7 | Multi-step workflows | Test complex sequences |
| **Run All** | 101 | All above combined | Complete evaluation |

---

## Quick Commands

```bash
# Interactive with behavior tests (recommended)
python -m Evaluator
# Select [1] Behavior Rubric Tests

# Run all tests
python -m Evaluator
# Select [5] Run All Tests

# CLI with behavior tests
python -m Evaluator.cli \
  --model your-model \
  --prompt-set Evaluator/prompts/behavior_rubric.json \
  --markdown results/report.md

# Test specific behavior only
python -m Evaluator.cli \
  --model your-model \
  --prompt-set Evaluator/prompts/behavior_rubric.json \
  --tags intellectual_humility

# Compare models
python -m Evaluator.cli --model base-model --prompt-set Evaluator/prompts/behavior_rubric.json --output results/base.json
python -m Evaluator.cli --model trained-model --prompt-set Evaluator/prompts/behavior_rubric.json --output results/trained.json
```

---

## Documentation

| Guide | Purpose |
|-------|---------|
| **QUICKSTART.md** | Quick CLI reference |
| **INTERACTIVE_MODE.md** | Interactive mode details |
| **BEHAVIOR_EVALUATION_GUIDE.md** | Behavior scoring & interpretation |
| **README.md** | Full technical details |

---

## Expected Results

### Base Model (Untrained)
- Tool calls: ~60-80% correct
- Context quality: Low
- Verification: Rare (~10%)

### After SFT Training
- Tool calls: 85-95% correct
- Context quality: Moderate
- Verification: Inconsistent (30-60%)

### After Behavior KTO Training
- Tool calls: 90-98% correct
- Context quality: High (80+ chars)
- Verification: Consistent (85-95%)
- Batch usage: 70-85%
- Escalation: 80-90%

---

## Output Files

```
Evaluator/results/
├── model_behavior_rubric_20251122_160000.json  # Full results
├── model_behavior_rubric_20251122_160000.md    # Human-readable report
├── model_full_coverage_20251122_160500.json
└── model_full_coverage_20251122_160500.md
```

When running "All Tests", each suite gets its own file:
```
model_behavior_rubric_<timestamp>.json
model_full_coverage_<timestamp>.json
model_baseline_<timestamp>.json
model_tool_combos_<timestamp>.json
```

---

## Tips

1. **Always test with behavior rubric** after training
2. **Run multiple times** (3-5) for consistency
3. **Compare before/after** to measure improvement
4. **Check markdown report** for easy review
5. **Use tags** to focus on specific behaviors

---

## Getting Started

```bash
# 1. Start LM Studio and load your model
# 2. Run evaluator
python -m Evaluator

# 3. Select test suite
# Recommended: [1] Behavior Rubric Tests

# 4. Let it run and check results
# Results saved to Evaluator/results/
```

---

**Need more details?**
- Quick start: See `QUICKSTART.md`
- Interactive mode: See `INTERACTIVE_MODE.md`
- Behavior scoring: See `BEHAVIOR_EVALUATION_GUIDE.md`
