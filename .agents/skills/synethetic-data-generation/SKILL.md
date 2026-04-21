---
name: synthetic-data-generation
description: Complete reference for the SynthChat synthetic dataset generation system. Covers CLI commands (generate, improve, validate), scenario YAML authoring, rubric YAML authoring, settings configuration, evaluation, and full workflow. Use when generating datasets, writing rubrics/scenarios, configuring models/workers, improving dataset quality, or running evaluations. This skill is about USING the system via CLI and YAML — never modifying source code.
allowed-tools: Read, Bash, Write, Grep, Glob
---

# SynthChat: Synthetic Data Generation

Generate, improve, validate, and evaluate synthetic training datasets via CLI and YAML configuration.

## Quick Reference

| Task | Command |
|------|---------|
| Generate dataset | `python -m SynthChat.run generate [options]` |
| Generate with environment runtime checks | `python -m SynthChat.run generate --env-backend local [options]` |
| Generate with custom tool schema/rules | `python -m SynthChat.run generate --env-backend local --env-tool-schema path/to/tool_schema.yaml --env-exec-config path/to/environment_execution.yaml [options]` |
| Improve dataset | `python -m SynthChat.run improve -i FILE [options]` |
| Validate dataset | `python -m SynthChat.run validate -i FILE [options]` |
| Evaluate model | `python -m Evaluator.cli --model NAME [options]` |
| Structural check | `python3 scripts/validate_syngen.py FILE` |
| JSONL → Markdown | `./scripts/jsonl_to_markdown.sh data.jsonl` |
| Combine datasets | `./scripts/combine_datasets.sh -o out.jsonl FILE1 FILE2` |
| Interactive menu | `./run.sh` |

## Key Directories

- `SynthChat/scenarios/` — Generation templates (6 files, ~30 scenarios)
- `SynthChat/rubrics/` — Quality rubrics (17 files)
- `SynthChat/config/` — `settings.yaml`, `validation.yaml`
- `Evaluator/config/environment_execution.yaml` — Runtime action inference rules (config-driven)
- `Datasets/synthchat/` — Generated datasets go here (dry-runs and full runs)
- `SynthChat/interactions/` — Judge/improve logs

## Progressive Reference

Load the specific reference you need:

| Reference | When to Load | Path |
|-----------|-------------|------|
| **CLI Commands** | Running generate/improve/validate/eval | `reference/cli-commands.md` |
| **Settings Config** | Configuring providers, models, workers, targets | `reference/settings-config.md` |
| **Scenario Authoring** | Writing or modifying scenario YAMLs | `reference/scenario-authoring.md` |
| **Rubric Authoring** | Writing or modifying rubric YAMLs | `reference/rubric-authoring.md` |
| **Testing Protocol** | After creating/modifying scenarios or rubrics — MUST dry-run before full generation | `reference/testing-protocol.md` |
| **Manual Editing** | Hand-crafting individual dataset lines | `reference/manual-editing.md` |

## MANDATORY: Dry-Run Before Full Generation

**NEVER go straight from writing a scenario/rubric to a full generation run.**

After creating or modifying any scenario or rubric YAML:

1. **Dry-run 3-5 examples** → show user → get feedback
2. **Iterate** on YAML based on feedback
3. **Only after user approves** → run full generation

See `reference/testing-protocol.md` for the full protocol and dry-run script.

## Common Patterns

**Generate with parallel workers:**
```bash
python -m SynthChat.run generate --workers 4
```

**Improve specific rubrics on a line range:**
```bash
python -m SynthChat.run improve -i data.jsonl --rubrics thinking_quality,factuality --start-line 1 --end-line 50
```

**Switch provider/model at CLI:**
```bash
python -m SynthChat.run generate --provider openrouter --model google/gemini-2.0-flash-001
```

**Generate from docs:**
```bash
python -m SynthChat.run generate --docs "path/to/essays/" --scenarios essay_outline --per-doc 1
```

**Dry-run a checked-in smoke target manifest:**
```bash
python -m SynthChat.run generate \
  --targets-file SynthChat/config/targets_cli_existing_tools_quickcheck.json \
  --max-iterations 3 \
  --output Datasets/synthchat/dryrun_cli_existing_tools_quickcheck.jsonl
```

**Validate then fix:**
```bash
python -m SynthChat.run validate -i Datasets/synthchat/data.jsonl --rubrics system_prompt_format
python -m SynthChat.run improve -i Datasets/synthchat/data.jsonl --rubrics system_prompt_format
```

**Always save outputs to `SynthChat/outputs/`:**
```bash
python -m SynthChat.run generate -o Datasets/synthchat/my_dataset.jsonl
```

## Environment Variables

```bash
OPENROUTER_API_KEY=sk-or-...          # Required for OpenRouter
LMSTUDIO_HOST=localhost               # LM Studio host
LMSTUDIO_PORT=1234                    # LM Studio port
OLLAMA_HOST=http://localhost:11434    # Ollama endpoint
HF_TOKEN=hf_...                       # HuggingFace uploads
```

## Config-Driven Architecture

SynthChat is fully config-driven — all tool-call formats, workspace structures, and label mappings are defined in YAML under `SynthChat/config/`. The included formats (e.g., `useTools` wrapper) are **example demonstrations**, not canonical formats. Users define their own formats without touching code.

Key config files:
- `SynthChat/config/tool_call_formats.yaml` — Tool-call response schemas (wrapper name, context fields, call structure)
- `SynthChat/config/workspace_formats.yaml` — System prompt sections and structure
- `SynthChat/config/label_mappings.yaml` — Issue classification and label rollups
- `SynthChat/config/settings.yaml` — Generation settings, model config, output paths

To add a new tool-call format, add a named entry to `tool_call_formats.yaml` and reference it from your scenario YAML. No code changes needed.

## Tips

- Use `--workers 4` for parallel generation (each worker gets its own LLM client)
- Set `save_failures: true` in settings to keep failed examples as KTO negatives
- Interactions log in `SynthChat/interactions/` shows judge/improve exchanges
- Progress checkpoints save to `.synthchat_checkpoint.json` on interruption
- Be greedy to stop on errors — kill early, fix, retest
- Environment traces are stored under `example.metadata.environment` when enabled
- For non-default tool names, provide `--env-tool-schema` and `--env-exec-config`
- Prefer checked-in `SynthChat/config/targets_*.json` manifests over ad hoc inline JSON when running smoke tests or repeatable generation slices
