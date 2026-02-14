# Presets & Tags Reference

Evaluation presets and tag-based filtering.

---

## Presets

Defined in `Evaluator/config/eval_run.yaml`:

| Preset | Description | Use Case |
|--------|-------------|----------|
| `quick` | Quick smoke test (subset of tests) | Fast iteration during development |
| `full` | All tests, all scenarios | Comprehensive pre-release testing |
| `behavior_only` | Behavior tests only | Focus on reasoning/clarification |
| `strict` | All tests with 95% pass threshold | Quality gate for releases |

### Using Presets

```bash
# Quick smoke test
python -m Evaluator.cli --backend lmstudio --model MODEL --preset quick

# Full suite
python -m Evaluator.cli --backend lmstudio --model MODEL --preset full

# Behavior focus
python -m Evaluator.cli --backend lmstudio --model MODEL --preset behavior_only
```

Presets override `--scenario` and other flags. For custom runs, use flags directly.

---

## Tags

Tags categorize tests for filtered evaluation runs.

### Available Tags

**Behavioral:**
| Tag | Tests | What It Tests |
|-----|-------|---------------|
| `intellectual_humility` | ~10 | Asking for clarification on ambiguous requests |
| `clarification` | ~8 | Seeking more info before acting |
| `destructive` | ~6 | Caution with delete/overwrite operations |
| `delegation` | ~4 | Using promptManager for complex tasks |

**Tool-Specific:**
| Tag | Tests | What It Tests |
|-----|-------|---------------|
| `storageManager` | ~15 | File operations (move, delete, create, rename) |
| `contentManager` | ~8 | Content editing (write, append, replace) |
| `vaultLibrarian` | ~5 | Search operations |
| `memoryManager` | ~4 | Session/workspace management |
| `agentManager` | ~3 | Agent CRUD operations |

### Tag Filtering

```bash
# Single tag
python -m Evaluator.cli --backend lmstudio --model MODEL \
  --tags intellectual_humility

# Multiple tags (comma-separated)
python -m Evaluator.cli --backend lmstudio --model MODEL \
  --tags intellectual_humility,clarification,destructive

# Combine with scenario
python -m Evaluator.cli --backend lmstudio --model MODEL \
  --scenario behavior_prompts.yaml \
  --tags clarification
```

---

## Custom Run Configuration

For runs that don't fit presets:

```bash
# Run only 5 tests
python -m Evaluator.cli --backend lmstudio --model MODEL \
  --scenario behavior_prompts.yaml --limit 5

# Run specific tags with JSON output
python -m Evaluator.cli --backend lmstudio --model MODEL \
  --scenario tool_prompts.yaml \
  --tags storageManager \
  --output Evaluator/results/storage_tests.json

# Dry run to verify config
python -m Evaluator.cli --backend lmstudio --model MODEL \
  --scenario behavior_prompts.yaml --dry-run
```

---

## Recommended Evaluation Strategy

### During Development
```bash
--preset quick              # Fast feedback loop
```

### After Training
```bash
--preset full               # Comprehensive check
```

### Before Release
```bash
--preset strict             # Quality gate (95% threshold)
```

### Investigating Specific Capability
```bash
--tags intellectual_humility  # Focus on weakness
```

### A/B Comparison
```bash
# Same preset, different models
--preset full --output results/model_a.json
--preset full --output results/model_b.json
```
