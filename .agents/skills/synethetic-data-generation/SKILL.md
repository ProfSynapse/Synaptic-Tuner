---
name: synthetic-data-generation
description: Complete reference for the SynthChat synthetic dataset generation system. Covers CLI commands (generate, improve, validate), scenario YAML authoring, rubric YAML authoring, settings configuration, evaluation, and full workflow. Use when generating datasets, writing rubrics/scenarios, configuring models/workers, improving dataset quality, or running evaluations. This skill is about USING the system via CLI and YAML — never modifying source code.
allowed-tools: Read, Bash, Write, Grep, Glob
---

# SynthChat: Synthetic Data Generation

Generate, improve, validate, sanitize, and evaluate synthetic training datasets via CLI and YAML configuration.

## Quick Reference

| Task | Command |
|------|---------|
| Generate dataset | `python -m SynthChat.run generate [options]` |
| Generate with environment runtime checks | `python -m SynthChat.run generate --env-backend local [options]` |
| Generate with custom tool schema/rules | `python -m SynthChat.run generate --env-backend local --env-tool-schema path/to/tool_schema.yaml --env-exec-config path/to/environment_execution.yaml [options]` |
| Improve dataset | `python -m SynthChat.run improve -i FILE [options]` |
| Validate dataset | `python -m SynthChat.run validate -i FILE [options]` |
| Sanitize docs or JSONL | `python -m SynthChat.run sanitize -i PATH --privacy-profile PROFILE [options]` |
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
| **CLI Commands** | Running generate/improve/validate/sanitize/eval | `reference/cli-commands.md` |
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

## Default Quality Gates

In this repo, the default assumption for scenario authoring is:
- scenarios should include rubrics
- scenarios should use judge/final_judge where appropriate
- tool scenarios should use environment execution when practical

Do not assume `generate` is automatically running a judge just because
`--max-iterations` is set. That only matters when the scenario actually defines
rubrics/judge config. If a scenario has no `rubrics`, `judge`, or
`final_judge`, generation is just raw sampling plus any enabled environment
checks.

Treat judge-less scenarios as explicit smoke/plumbing exceptions, not the
default production pattern.

## Common Patterns

**Generate with parallel workers:**
```bash
python -m SynthChat.run generate --workers 4
```

**Improve specific rubrics on a line range:**
```bash
python -m SynthChat.run improve -i data.jsonl --rubrics thinking_quality,factuality --start-line 1 --end-line 50
```

**Regenerate arbitrary rows from a JSONL file:**
```bash
python -m SynthChat.run improve \
  -i data.jsonl \
  --rubrics promptManager_tools \
  --lines 7,12,20-25 \
  --workers 8 \
  -o Datasets/synthchat/regen_slice.jsonl
```

**Use a checked-in line manifest for targeted regeneration:**
```bash
python -m SynthChat.run improve \
  -i data.jsonl \
  --rubrics contentManager_tools \
  --line-file Datasets/tools_datasets/reports/cli_schema/regen_lines.txt \
  --workers 12 \
  -o Datasets/synthchat/regen_slice.jsonl
```

**Switch provider/model at CLI:**
```bash
python -m SynthChat.run generate --provider openrouter --model openai/gpt-oss-120b
```

**Generate from docs:**
```bash
python -m SynthChat.run generate --docs "path/to/essays/" --scenarios essay_outline --per-doc 1
```

**Generate from raw docs with privacy preprocessing:**
```bash
python -m SynthChat.run generate --docs "tests/fixtures/privacy/raw_seed_docs" --targets-file SynthChat/config/targets_privacy_docs_smoke.json --privacy-profile realistic_pseudonyms
```

**Sanitize a docs folder or JSONL dataset:**
```bash
python -m SynthChat.run sanitize -i tests/fixtures/privacy/raw_seed_docs --privacy-profile realistic_pseudonyms -o tmp/privacy_docs
python -m SynthChat.run sanitize -i tests/fixtures/privacy/raw_seed_dataset.jsonl --privacy-profile mask_only -o tmp/privacy_dataset.jsonl
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
OPF_CHECKPOINT=/path/to/privacy-filter-checkpoint   # Local OpenAI Privacy Filter checkpoint
TIKTOKEN_CACHE_DIR=/path/to/tiktoken-cache          # Local tiktoken cache for OPF
VLLM_HOST=127.0.0.1                  # Optional OpenAI-compatible vLLM polish endpoint
VLLM_PORT=8000
```

## Privacy Setup

Use the privacy preprocess path when raw docs or JSONL may contain PII or secrets and you want SynthChat to sanitize that content before it reaches the generation/improvement model.

Runtime split:
- `openai/privacy-filter` is the local span-detection/redaction model
- `opf` is the runtime wrapper used to load and run that model
- `vllm` is only for the optional post-sanitize `llm_polish` step, not for OPF itself

Profiles live in:
- `SynthChat/config/privacy_profiles.yaml`

Global defaults live in:
- `SynthChat/config/settings.yaml` under `privacy_preprocess`

Scenario-level opt-in lives in:
- `seed_data.preprocess_profile`

Recommended first-use setup when auto-download is unreliable:

```powershell
python - <<'PY'
from pathlib import Path
import shutil
from huggingface_hub import snapshot_download

target = Path(r"F:\Code\Toolset-Training\tmp\opf_privacy_filter")
if not target.exists():
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="openai/privacy-filter",
        local_dir=str(target),
        allow_patterns=["original/*"],
    )
    original = target / "original"
    for path in original.iterdir():
        shutil.move(str(path), str(target / path.name))
    original.rmdir()
print(target)
PY
```

OPF also needs the `o200k_base.tiktoken` encoding file. If that does not auto-download cleanly, cache it locally:

```powershell
New-Item -ItemType Directory -Force -Path F:\Code\Toolset-Training\tmp\tiktoken_cache | Out-Null
Invoke-WebRequest `
  -Uri "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken" `
  -OutFile "F:\Code\Toolset-Training\tmp\tiktoken_cache\fb374d419588a4632f3f557e76b4b70aebbca790"
```

Then set:

```powershell
$env:OPF_CHECKPOINT="F:\Code\Toolset-Training\tmp\opf_privacy_filter"
$env:TIKTOKEN_CACHE_DIR="F:\Code\Toolset-Training\tmp\tiktoken_cache"
```

Once those are set, the real OPF-backed sanitize flow can run fully from local files.

## Config-Driven Architecture

SynthChat is fully config-driven — all tool-call formats, workspace structures, label mappings, and dataset-specific wrapper assumptions must be defined in YAML/config, not hardcoded in code.

Important discipline for this repo:
- the current CLI/tool wrapper is only one example dataset format, not a runtime truth
- do not encode wrapper names, top-level fields, or command assumptions in parser/executor/judge code unless that behavior is driven from config
- if a generation/eval issue seems specific to the current toy/example format, fix config, scenarios, rubrics, or format definitions first
- environment/runtime failures should be lifted into judge/improver context as structured payload data, not handled primarily with ad hoc format-specific code repairs

Key config files:
- `SynthChat/config/tool_call_formats.yaml` — Tool-call response schemas (wrapper name, context fields, call structure)
- `SynthChat/config/workspace_formats.yaml` — System prompt sections and structure
- `SynthChat/config/label_mappings.yaml` — Issue classification and label rollups
- `SynthChat/config/settings.yaml` — Generation settings, model config, output paths

To add a new tool-call format, add a named entry to `tool_call_formats.yaml` and reference it from your scenario YAML. No code changes needed.

## Tips

- Use `--workers 4` for parallel generation (each worker gets its own LLM client)
- Use `improve --lines 3,8,10-15` or `--line-file path.txt` for targeted regeneration of arbitrary dataset rows
- Improve reports preserve original input `line_number` values even when you select a subset, so merge workflows can patch the source file deterministically
- Set `save_failures: true` in settings to keep failed examples as KTO negatives
- Interactions log in `SynthChat/interactions/` shows judge/improve exchanges
- Progress checkpoints save to `.synthchat_checkpoint.json` on interruption
- Be greedy to stop on errors — kill early, fix, retest
- Environment traces are stored under `example.metadata.environment` when enabled
- When environment validation is enabled, make sure the active judge/improver path receives those errors through prompt variables/config rather than relying on format-specific code patches
- If a response-stage retry is driven by environment feedback, each retry round must be judged against a freshly rerun environment result. Do not carry a prior round's environment failure forward into later judgments after the response has changed.
- For non-default tool names, provide `--env-tool-schema` and `--env-exec-config`
- Prefer checked-in `SynthChat/config/targets_*.json` manifests over ad hoc inline JSON when running smoke tests or repeatable generation slices
- For privacy smoke tests, prefer the checked-in fixtures under `tests/fixtures/privacy/`, the target manifest `SynthChat/config/targets_privacy_docs_smoke.json`, and the runbook `docs/plans/synthchat-privacy-smoke-runbook.md`
