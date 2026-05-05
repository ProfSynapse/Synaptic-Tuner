---
name: evaluation
description: Complete reference for the config-first model evaluation system. Covers the Evaluator CLI, assertion-driven YAML scenarios, response views, backend configuration, presets, scoring, LLM-as-judge, model comparison, and HuggingFace integration. Use when evaluating models, writing test prompts, comparing training runs, or interpreting eval results. This skill is about USING the evaluation system via CLI and YAML.
allowed-tools: Read, Bash, Write, Grep, Glob
---

# Model Evaluation

Config-first evaluation framework for testing model responses against YAML-defined correctness assertions.

The evaluator does not hardcode a specific tool family, manager id, wrapper name, or behavior rule as correctness. Scenarios define the prompt and the acceptable response shape directly under `correct`.

## Quick Reference

| Task | Command |
|------|---------|
| Interactive menu | `./run.sh` then Evaluate |
| Local Docker eval job | `python tuner.py local-run --job-config Trainers/local/jobs/<eval-job>.yaml --yes` |
| Tool eval | `python -m Evaluator.cli --backend vllm --model MODEL --scenario tool_prompts.yaml --host 127.0.0.1 --port 8011` |
| Full configured eval | `python -m Evaluator.cli --backend lmstudio --model MODEL --preset full` |
| Quick smoke test | `python -m Evaluator.cli --backend lmstudio --model MODEL --preset quick` |
| Tag filter | `python -m Evaluator.cli --backend lmstudio --model MODEL --scenario tool_prompts.yaml --tags TAG_NAME` |
| Dry run config load | `python -m Evaluator.cli --backend lmstudio --model MODEL --scenario tool_prompts.yaml --dry-run` |
| Eval with environment runtime | `python -m Evaluator.cli --backend lmstudio --model MODEL --scenario tool_prompts.yaml --env-backend local` |
| Eval with LLM judge | `python -m Evaluator.cli --backend lmstudio --model MODEL --scenario tool_prompts.yaml --judge --judge-rubrics tool_call_quality` |
| Eval + upload to HF | `python -m Evaluator.cli --backend unsloth --model PATH --upload-to-hf user/model` |

## Status System

| Status | Meaning | When |
|--------|---------|------|
| **PASS** | Configured checks passed | `correct` assertions passed, and optional environment/judge checks passed |
| **FAIL** | Configured checks failed or request errored | No `correct.any` path matched, required environment checks failed, judge failed, or backend errored |

Schema/structural validation may still be reported for debugging, but it is not the source of task correctness. Correctness belongs in scenario YAML.

## Key Directories

- `Evaluator/` - Core evaluation code
- `Evaluator/config/scenarios/` - YAML test scenarios
- `Evaluator/config/tool_schema.yaml` - Configured tool schema metadata
- `Evaluator/config/rubrics/` - LLM-as-judge rubrics
- `Evaluator/results/` - Evaluation output JSON and Markdown

## Progressive Reference

| Reference | When to Load | Path |
|-----------|-------------|------|
| CLI Commands | Running evaluations, all flags and examples | `reference/cli-commands.md` |
| Scenario Authoring | Writing or modifying YAML test scenarios | `reference/scenario-authoring.md` |
| Backends | Configuring vLLM, LM Studio, Ollama, Unsloth, and others | `reference/backends.md` |
| Results & Metrics | Interpreting JSON/Markdown output and failures | `reference/results-metrics.md` |
| Presets & Tags | Using presets and tag filters | `reference/presets-tags.md` |

## Active Scenario Pattern

Every test should define what counts as correct:

```yaml
tests:
  - id: configured_tool_case
    question: Use the configured tool interface to complete the requested operation.
    tags: [tool-call, single-tool]
    system: |
      <runtime_context>
      Include any context fields required by this scenario's configured schema.
      </runtime_context>
    correct:
      any:
        - name: configured_tool_call
          assertions:
            - type: jsonpath_equals
              path: $.tool_calls[0].name
              value: CONFIGURED_WRAPPER_NAME
            - type: jsonpath_regex
              path: $.tool_calls[0].arguments.ACTION_FIELD
              pattern: 'expected configured action pattern'
```

Use `correct.any` for multiple valid answers, such as command by id or by name. Use `correct.all` or nested `all`/`any`/`not` assertions for stricter structures.

## Response View

Assertions query a generic response view. This is syntax normalization only:

- `$.raw` preserves the raw assistant response.
- `$.content` is assistant text.
- `$.content_json` is parsed JSON content when content is JSON.
- `$.tool_calls` is a normalized list of emitted tool calls.
- OpenAI-style `function.arguments` JSON strings are parsed into objects.
- Plain text blocks like `tool_call: CONFIGURED_WRAPPER_NAME` plus `arguments: {...}` are parsed into the same view.

The response view must not map CLI commands to old manager tool ids or decide correctness. Scenario YAML decides what is correct.

## Tips

- Keep all task-specific expectations in YAML under `correct`.
- Do not add evaluator code for a specific tool, wrapper, or use case.
- Prefer regex or JSONPath assertions for configured action payloads, because quoting, field order, and equivalent forms can vary.
- If a schema allows equivalent forms, represent them as separate `correct.any` paths.
- Use `--limit` and `--tags` for fast iteration.
- Use `--validate-context` only when the scenario includes context fields that should be structurally checked.
- Use `--env-backend local` or `e2b` only when you need runtime execution checks beyond response correctness.
- For local LoRA adapter eval in this repo, prefer `python tuner.py local-run --job-config ...` with `run.method: eval` and `evaluation.runtime: vllm`. That path serves the base model in the local vLLM Docker container and overlays the adapter for parity with the intended local serving runtime.
- On Windows, set `job.persist: true` for repeat local vLLM eval jobs so `local-run` reuses a named copy-mode container instead of creating a fresh stopped container each time. Set the same `job.container_name` across related eval jobs when they should share the same reused container.
- Run long Docker/vLLM evals as background jobs with stdout/stderr redirected to files under `Evaluator/results/`, then poll logs and `stage_summary.json` with short commands. Avoid foreground waits that hide failures until a long timeout.
- Use a fresh `artifacts.host_path` for each eval attempt, or copy completed in-container artifacts to a fresh host folder after a nonzero eval exit. Some evals intentionally return nonzero when cases fail even though results were produced.
- For persistent copy-mode containers, avoid recopied large model directories once the model is already present. If a `docker cp` is interrupted, verify the container model directory before rerunning; partial model copies can cause misleading vLLM startup failures.
- For long workspace/system prompts, keep `evaluation.max_tokens` lower than the served model window leaves room for, and raise `--max-model-len` via `server_extra_args` when the model and hardware support it.
