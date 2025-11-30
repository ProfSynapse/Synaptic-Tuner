# Interactive Evaluation Mode

Quick guide for using the interactive evaluator (`python -m Evaluator`).

---

## Quick Start

```bash
python -m Evaluator
```

This will guide you through:
1. **Select test suite** - Choose which tests to run
2. **Select model** - Pick from models available in LM Studio
3. **Choose run count** - How many times to run the tests
4. **Auto-run** - Tests execute and save results automatically

---

## Test Suite Options

When you run `python -m Evaluator`, you'll be prompted to choose a test suite:

### 1. Behavior Rubric Tests (Recommended) ⭐
- **43 prompts** testing all 6 behavior patterns
- Tests: Intellectual Humility, Verification, Context Continuity, Strategic Tool Selection, Error Recovery, Workspace Awareness
- **Best for:** Evaluating trained models after SFT/KTO training
- **Use when:** You want to measure behavior quality

### 2. Full Tool Coverage
- **45 prompts** - one test per tool
- Tests: Every vaultManager, contentManager, memoryManager, vaultLibrarian, agentManager tool
- **Best for:** Sanity checking tool functionality
- **Use when:** You want to verify all tools work

### 3. Baseline Tests
- **6 prompts** - general functionality with behavior expectations
- Tests: Mixed scenarios with behavior awareness
- **Best for:** Quick smoke tests
- **Use when:** You want a fast check

### 4. Multi-Step Workflows
- **7 prompts** - complex tool sequences
- Tests: Multi-tool chaining and workflow patterns
- **Best for:** Testing complex scenarios
- **Use when:** You want to verify multi-step operations

---

## Interactive Flow

### Example Session

```
+====================================+
|      LM Studio Evaluator           |
|        Interactive CLI             |
+====================================+

Select test suite:
[1] Behavior Rubric Tests
     43 prompts testing all 6 behavior patterns (Recommended)
[2] Full Tool Coverage
     45 prompts - one test per tool
[3] Baseline Tests
     6 general prompts with behavior expectations
[4] Multi-Step Workflows
     7 prompts testing complex tool sequences

Enter a number (default 1): 1
Selected: Behavior Rubric Tests

Using only available model: nexus-toolbox_v0.1.6

How many runs? (default 1): 1

Running evaluation for: nexus-toolbox_v0.1.6
Test suite: Behavior Rubric (43 prompts)
Prompt file: F:\Code\Toolset-Training\Evaluator\prompts\behavior_prompts.json
Runs: 1

--- Run 1/1 ---
[PASS] IH_ambiguous_deletion
[PASS] IH_complex_organization
...
```

---

## Command-Line Shortcuts

You can skip interactive prompts by providing flags:

### Skip Model Selection
```bash
python -m Evaluator --model your-model-name
```

### Skip Test Suite Selection
```bash
python -m Evaluator --prompt-set Evaluator/prompts/behavior_prompts.json
```

### Skip Run Count Selection
```bash
python -m Evaluator --runs 3
```

### Skip All Prompts
```bash
python -m Evaluator \
  --model nexus-toolbox_v0.1.6 \
  --prompt-set Evaluator/prompts/behavior_prompts.json \
  --runs 1
```

---

## Output Files

Results are saved automatically:

### Default Location
```
Evaluator/results/
├── <model-name>_<test-suite>_<timestamp>.json
└── <model-name>_<test-suite>_<timestamp>.md
```

### Example Filenames
```
nexus-toolbox_v0_1_6_behavior_rubric_20251122_163000.json
nexus-toolbox_v0_1_6_behavior_rubric_20251122_163000.md
```

### Custom Output Directory
```bash
python -m Evaluator --output-dir results/my_tests
```

---

## Tips

### 1. **Always Use Behavior Tests for Trained Models**
If you've trained a model with the behavior rubric dataset, test it with:
```bash
python -m Evaluator
# Then select [1] Behavior Rubric Tests
```

### 2. **Run Multiple Times for Consistency**
Models can have variability. Run 3-5 times:
```bash
python -m Evaluator --runs 5
```

### 3. **Compare Before/After Training**
```bash
# Before training
python -m Evaluator --model base-model --prompt-set Evaluator/prompts/behavior_prompts.json --runs 3

# After training
python -m Evaluator --model trained-model --prompt-set Evaluator/prompts/behavior_prompts.json --runs 3
```

### 4. **Use Dry Run to Test Setup**
```bash
python -m Evaluator --dry-run
```

### 5. **Increase Timeout for Large Models**
```bash
python -m Evaluator --timeout 120
```

---

## Reading Results

### Console Output
Shows real-time progress and summary:
```
[PASS] IH_ambiguous_deletion
[FAIL] IH_complex_organization (2 issue(s))
[PASS] VBA_destructive_folder_delete
...

Evaluated 43 prompt(s): 38 passed, 5 failed.
Pass rate by tag:
  - intellectual_humility: 6/8 (75.0%)
  - verification_before_action: 5/6 (83.3%)
  - context_continuity: 7/7 (100.0%)
  ...
```

### JSON File
Complete results with:
- All prompts and responses
- Tool calls made
- Validation details
- Metadata (model, settings, timing)

### Markdown Report
Human-readable summary with:
- Pass/fail for each prompt
- Validation issues
- Tool usage statistics
- Response examples

---

## Troubleshooting

### "Unable to list models from LM Studio"
1. Make sure LM Studio is running
2. Load a model in LM Studio
3. Start the local server (Server tab)
4. Check it's at `http://127.0.0.1:1234`

### "Prompt set not found"
The prompt file doesn't exist. Check:
- File path is correct
- You're in the repository root
- Prompt files exist in `Evaluator/prompts/`

### Model Takes Too Long
Increase timeout:
```bash
python -m Evaluator --timeout 180
```

### Want to Use Ollama Instead
Use the full CLI:
```bash
python -m Evaluator.cli \
  --backend ollama \
  --model your-model \
  --prompt-set Evaluator/prompts/behavior_prompts.json
```

---

## When to Use Which Mode

### Use Interactive Mode (`python -m Evaluator`) When:
- ✅ Testing models in LM Studio
- ✅ You want guided prompts
- ✅ Quick evaluation sessions
- ✅ You're new to the evaluator

### Use CLI Mode (`python -m Evaluator.cli`) When:
- ✅ Using Ollama backend
- ✅ Scripting/automation
- ✅ Custom settings needed
- ✅ Running specific test subsets (with `--tags`)

---

## Quick Command Reference

```bash
# Interactive (easiest)
python -m Evaluator

# Interactive with model pre-selected
python -m Evaluator --model your-model

# Interactive with test suite pre-selected
python -m Evaluator --prompt-set Evaluator/prompts/behavior_prompts.json

# Fully automated (no prompts)
python -m Evaluator \
  --model your-model \
  --prompt-set Evaluator/prompts/behavior_prompts.json \
  --runs 3

# Dry run (test setup without API calls)
python -m Evaluator --dry-run

# Full CLI (maximum control)
python -m Evaluator.cli \
  --backend lmstudio \
  --model your-model \
  --prompt-set Evaluator/prompts/behavior_prompts.json \
  --tags intellectual_humility \
  --limit 10 \
  --output results/test.json \
  --markdown results/report.md
```

---

## Next Steps

1. **Run your first test:**
   ```bash
   python -m Evaluator
   ```

2. **Review the markdown report** to see detailed results

3. **Compare with baseline** to measure improvement

4. **See BEHAVIOR_EVALUATION_GUIDE.md** for scoring and interpretation

5. **See QUICKSTART.md** for more CLI examples
