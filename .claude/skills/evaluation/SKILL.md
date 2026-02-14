---
name: evaluation
description: Complete reference for the model evaluation system. Covers the Evaluator CLI, writing YAML test scenarios, backend configuration, presets, behavior/schema validation, scoring, model comparison, and HuggingFace integration. Use when evaluating models, writing test prompts, comparing training runs, or interpreting eval results. This skill is about USING the evaluation system via CLI and YAML — never modifying source code.
allowed-tools: Read, Bash, Write, Grep, Glob
---

# Model Evaluation

Config-driven evaluation framework for testing tool-calling models against YAML-defined test scenarios.

## Quick Reference

| Task | Command |
|------|---------|
| Interactive menu | `./run.sh` → Evaluate |
| Full evaluation | `python -m Evaluator.cli --backend lmstudio --model MODEL --scenario behavior_prompts.yaml` |
| Quick smoke test | `python -m Evaluator.cli --backend lmstudio --model MODEL --preset quick` |
| Tag filter | `python -m Evaluator.cli --backend lmstudio --model MODEL --tags intellectual_humility` |
| Dry run (no model calls) | `python -m Evaluator.cli --backend lmstudio --model MODEL --dry-run` |
| Eval + upload to HF | `python -m Evaluator.cli --backend unsloth --model PATH --upload-to-hf user/model` |

## Status System

| Status | Meaning | When |
|--------|---------|------|
| **PASS** | All good | Correct tool + behavioral expectations met |
| **WARN** | Tool correct, behavior off | Right tool called but reasoning/interaction suboptimal |
| **FAIL** | Wrong tool or error | Wrong tool called, missing tool, or backend error |

## Key Directories

- `Evaluator/` — Core evaluation code
- `Evaluator/config/scenarios/` — Test scenario YAMLs (behavior + tool)
- `Evaluator/config/` — Run config, display config, tool schema, response types
- `Evaluator/results/` — Evaluation output (JSON + Markdown)

## Progressive Reference

Load the specific reference you need:

| Reference | When to Load | Path |
|-----------|-------------|------|
| **CLI Commands** | Running evaluations, all flags and options | `reference/cli-commands.md` |
| **Scenario Authoring** | Writing or modifying test scenario YAMLs | `reference/scenario-authoring.md` |
| **Backends** | Configuring eval backends (Ollama, LM Studio, Unsloth, etc.) | `reference/backends.md` |
| **Results & Metrics** | Interpreting results, comparing models, metrics reference | `reference/results-metrics.md` |
| **Presets & Tags** | Using presets, filtering by tags, run configuration | `reference/presets-tags.md` |

## Available Test Scenarios

| File | Tests | Focus |
|------|-------|-------|
| `behavior_prompts.yaml` | 51 | Behavioral patterns (clarification, humility, delegation) |
| `tool_prompts.yaml` | 40 | Tool calling correctness (right tool, right args) |

## Common Patterns

**Evaluate LoRA adapter directly:**
```bash
python -m Evaluator.cli \
  --backend unsloth \
  --model ./sft_output_rtx3090/TIMESTAMP/final_model \
  --scenario behavior_prompts.yaml tool_prompts.yaml \
  --output Evaluator/results/my_eval.json \
  --markdown Evaluator/results/my_eval.md
```

**Compare base vs fine-tuned:**
```bash
# Base model via LM Studio
python -m Evaluator.cli --backend lmstudio --model qwen2.5-7b-instruct \
  --output Evaluator/results/base.json

# Fine-tuned via Unsloth
python -m Evaluator.cli --backend unsloth --model path/to/lora \
  --output Evaluator/results/finetuned.json
```

**Focus on specific capability:**
```bash
python -m Evaluator.cli --backend lmstudio --model MODEL \
  --tags intellectual_humility,clarification \
  --output Evaluator/results/ih_tests.json
```

## Environment Variables

```bash
OLLAMA_HOST=127.0.0.1                 # Ollama host
OLLAMA_PORT=11434                     # Ollama port
LMSTUDIO_HOST=localhost               # LM Studio host (auto-detects WSL)
LMSTUDIO_PORT=1234                    # LM Studio port
OPENROUTER_API_KEY=sk-or-...          # OpenRouter API key
HF_TOKEN=hf_...                       # HuggingFace (for uploads)
```

## Tips

- Use `--preset quick` for fast iteration, `--preset full` for release testing
- Behavior tests (WARN) catch "correct but suboptimal" — useful for KTO training signals
- Failed behavior tests make great KTO negative examples
- Use `--validate-context` to check sessionId/workspaceId consistency
- Results JSON has `by_tag` breakdown — find weak areas fast
- `--lineage` creates structured provenance for model cards
- All test logic lives in YAML — add tests by editing YAML, not Python
