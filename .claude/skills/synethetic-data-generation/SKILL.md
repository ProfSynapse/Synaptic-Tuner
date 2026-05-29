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
| Debug environment generation only | `python -m SynthChat.run env-generate --scenario SCENARIO --debug-artifacts [path] [options]` |
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

For environment-backed multi-turn tool data, also load:
- `reference/scenario-authoring.md` for config-first scenario structure
- `reference/testing-protocol.md` for dry-run, raw artifact inspection, and failure triage

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
- environment-backed tool scenarios should default to a full repair loop:
  initial assistant response -> response/schema judge -> response improver ->
  response re-judge -> rerun environment with structured errors -> final judge

Do not assume `generate` is automatically running a judge just because
`--max-iterations` is set. That only matters when the scenario actually defines
rubrics/judge config. If a scenario has no `rubrics`, `judge`, or
`final_judge`, generation is just raw sampling plus any enabled environment
checks.

Treat judge-less scenarios as explicit smoke/plumbing exceptions, not the
default production pattern.

For environment-backed multi-turn tool data, response rubrics are not optional.
If an assistant turn fails response schema validation, the in-loop turn judge
will not run because the environment loop stops before executing that malformed
turn. The response rubric is the repair path that receives validation errors,
judge feedback, and environment issues, then produces the corrected assistant
message. The `final_judge` is a terminal acceptance gate, not the improver.

Environment-backed rows should persist the complete replay bundle by default:
generated fixture, assertions, resolved environment config, task context,
stage reviews, and enough source metadata to reconstruct the episode later.
When projecting rollouts into per-turn training rows, keep a pointer to the
canonical replay row or carry the replay bundle forward. A tool-turn-only
projection is useful for static supervised examples, but it is not enough for
later live environment replay unless the original environment artifact is still
available.

Use at least 3 retries/iterations for the response repair stages by default:
set CLI `--max-iterations 3`, keep scenario judge/final_judge `max_retries: 3`,
and make response rubrics strict enough to fail runtime misses such as missing
expected tools, malformed wrapper JSON, unsupported shell commands, or required
CLI arguments omitted by the model. If any stage fails, the default behavior
should be retry/repair with the structured failure context before accepting or
saving the example.

For multi-turn environment loops, keep single-response repair paths separate
from agentic rollouts. The normal path is:
- generate assistant turn
- if configured, return schema/format validation feedback to the model instead
  of ending the episode before any environment step
- run the environment step
- show model-facing tool feedback
- run in-loop judge feedback when configured
- continue only when a tool call, recoverable runtime error, or failed judge
  feedback requires another assistant action
- run final gates and final judge on the whole trajectory

Do not let a post-environment single-response improver flatten a multi-turn
episode unless the scenario explicitly opts into that behavior. A failed
multi-turn episode should usually be debugged from the episode trace, not
rewritten as one assistant message that tries to complete every step at once.

Likewise, do not run post-loop response validation/improvement on successful
agentic rollouts unless the scenario explicitly opts in. The in-loop schema
validation, environment execution, in-loop judge, final gates, and final judge
should be the normal acceptance path. A response-stage improver after the loop
can accidentally rewrite the saved terminal message and corrupt an otherwise
valid trajectory.

For config-driven tool schemas, distinguish internal executor identifiers from
model-facing command names. The model prompt, tool feedback, judges, rubrics,
and scenario prose should use the configured user-facing tool surface. Internal
executor names may still appear in runtime records, labels, or diagnostics, but
they should not leak into generated conversations unless that is the configured
surface being trained.

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
  --rubrics prompt_tools \
  --lines 7,12,20-25 \
  --workers 8 \
  -o Datasets/synthchat/regen_slice.jsonl
```

**Use a checked-in line manifest for targeted regeneration:**
```bash
python -m SynthChat.run improve \
  -i data.jsonl \
  --rubrics content_tools \
  --line-file Datasets/tools_datasets/reports/cli_schema/regen_lines.txt \
  --workers 12 \
  -o Datasets/synthchat/regen_slice.jsonl
```

**Switch provider/model at CLI:**
```bash
python -m SynthChat.run generate --provider openrouter --model MODEL_ID
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

**Isolate generated environment setup before a full rollout:**
```bash
python -m SynthChat.run env-generate \
  --scenario scenario_key \
  --provider openrouter \
  --model MODEL_ID \
  --llm-timeout 30 \
  --max-retries 1 \
  --max-tokens 2048 \
  --output SynthChat/output/env_generation_debug.json \
  --debug-artifacts SynthChat/output/env_generation_debug.debug_events.jsonl
```

Use this when a full multi-turn generation smoke appears stuck before the
first assistant turn. It exercises only the configured `environment_generation`
stage, writes the generated environment/resolved context bundle, and streams
raw debug events so request starts, retries, errors, reviews, and generated
keys are inspectable without running the agent loop. For hang triage, set
`--max-retries 1` and a short `--llm-timeout` first; otherwise the configured
retry envelope can make one failing environment request look like a long stall.
Use `--max-tokens` when isolating whether latency is caused by a large
structured environment response.
If a provider-routed request stalls but minimal requests work, rerun with
`--disable-provider-routing` to distinguish scenario/schema problems from a
specific hosted provider route.

For generated environments, prefer deterministic stage gates before relying on
an LLM judge. Useful generic gates include `json_schema` with
`schema: canonical_environment`, `no_placeholder_strings`,
`required_mapping_keys`, and `min_fixture_items`. The judge should explain
semantic quality, but gates should reject malformed schemas, placeholders,
missing hidden anchors, and too-small fixtures.

Treat environment generation, assistant turn generation, in-loop judging, and
final judging as separate model-selection surfaces. It can be correct to use a
more reliable structured-output model for fixture/answer-key authoring and a
different model for the actual rollout being trained or evaluated. Keep that
split in scenario YAML (`environment_generation.provider/model`,
`assistant_generation.provider/model`, `judge.*.provider/model`,
`final_judge.provider/model`) rather than code. When changing stage models,
rerun env-generation-only first, then a small full smoke, because a pass in one
stage does not prove the other stages are healthy.

For OpenRouter or other routed providers, choose the environment-generation
response format per stage. Use `response_format: json_object` for loose dynamic
maps. Use `response_format: json_schema` when the scenario supplies an inline
schema that fully constrains required fields, allowed extra fields, command
counts, and ASCII/path rules; this can prevent malformed JSON and corrupted
hidden anchors before the agent loop starts. Response-healing plugins and
fallback models are useful diagnostics, but do not rely on them instead of
deterministic gates and retryable stage reviews.

Use the same format decision for assistant turns and in-loop judges. If strict
provider schema mode returns provider-internal artifacts, malformed nested
arrays, or long routed stalls, set `assistant_generation.response_format:
json_object` or `judge.in_loop.response_format: json_object` in scenario YAML
and let the local schema validator, environment executor, and judge feedback
drive retries. Keep this as config, not scenario-specific parser logic.

When a tool trajectory fails, inspect the raw debug artifact before changing
scenario prose. Check the exact `tool` string emitted by the model, the
executor's parsed arguments, the `tool_results`, and the in-loop judge feedback.
Executor normalization can hide model mistakes such as non-ASCII whitespace or
markdown backticks unless those are rejected through config-driven validation
rules such as `invalid_cli_patterns` in the environment execution config.
Likewise, generated `task_context.expected_command_sequence` should be gated
against shell syntax or stale command examples when the trained surface is a
configured CLI/tool wrapper. Reject those through scenario gates/config, not
runtime string repairs.

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

To add a new tool-call format, add a named entry to `tool_call_formats.yaml`
and reference it from your scenario YAML. No code changes should be needed
unless you are adding a genuinely reusable runtime capability that cannot be
expressed in config.

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
- For multi-turn tool scenarios, inspect raw `conversation_trace`, `metadata.environment.episode_trace`, and `SynthChat/interactions/*.jsonl` before changing prompts. These usually reveal whether the problem is prompt rendering, tool syntax, environment execution, response repair, or final judging.
- If a model correctly answers after a successful environment pass, successful in-loop judge feedback should not force another assistant turn. Another turn should be requested only when the latest assistant action still needs correction.
- When feeding an invalid assistant tool response back to the model for repair, render prior tool calls as JSON, never Python `repr` or pseudo-JSON. Single-quoted dict/list strings in validation feedback teach the model to repeat malformed tool-call arguments.
- Local environment runtimes should use a controlled runtime temp root rather than relying on Python `tempfile` behavior if the host creates unwritable temp directories. Keep this generic and configurable; do not special-case a scenario.
- For non-default tool names, provide `--env-tool-schema` and `--env-exec-config`
- Prefer checked-in `SynthChat/config/targets_*.json` manifests over ad hoc inline JSON when running smoke tests or repeatable generation slices
- For privacy smoke tests, prefer the checked-in fixtures under `tests/fixtures/privacy/`, the target manifest `SynthChat/config/targets_privacy_docs_smoke.json`, and the runbook `docs/plans/synthchat-privacy-smoke-runbook.md`
