# Evaluator

Evaluation harness for Claudesidian-MCP models served via [Ollama](https://ollama.com/). The Evaluator issues curated prompts, captures model replies, and validates tool-call structure using the same schema rules enforced in `tools/validate_syngen.py`.

## Components

| File | Purpose |
| --- | --- |
| `config.py` | Runtime configuration dataclasses + helpers (model name, host, prompt sets, limits). |
| `ollama_client.py` | Minimal HTTP client for Ollama's `/api/chat` endpoint with retries and streaming hooks. |
| `prompt_sets.py` | Loader for JSON / JSONL prompt collections → `PromptCase` objects with tags and metadata. |
| `schema_validator.py` | Thin wrapper that reuses the dataset validator to score a single assistant reply. |
| `reporting.py` | Aggregates per-case results and produces JSON + Markdown summaries. |
| `runner.py` | Core orchestration loop (load prompts, call Ollama, validate response). |
| `cli.py` | Command-line entry point (`python -m Evaluator.cli --prompt-set prompts/baseline.json`). |
| `prompts/` | Baseline prompt sets (JSON arrays today, more formats later). |
| `results/` | Run artifacts (`.json`, `.md`); `.gitkeep` placeholder retains the folder in git. |

## Workflow

> Requirements: Python 3.10+ and `requests` installed in your environment (`pip install requests`).

1. **Serve the model via Ollama**
   ```bash
   OLLAMA_HOST=127.0.0.1 OLLAMA_PORT=11434 ollama run claudesidian-mcp:latest
   ```
2. **Run the evaluator**
   ```bash
   python3 -m Evaluator.cli \
     --model claudesidian-mcp \
     --prompt-set Evaluator/prompts/baseline.json \
     --output Evaluator/results/run_$(date +%s).json
   ```
3. **Inspect results**
   - Console summary groups successes/failures by tag.
   - JSON artifact contains raw prompts, responses, validator issues, runtime metadata.
   - Optional Markdown export surfaces quick human-readable diffs.

## Prompt Set Format

`Evaluator/prompts/baseline.json` follows this schema (more formats supported later):

```json
[
  {
    "id": "search_workspace_references",
    "question": "Can you find every note mentioning the delayed workspace migration?",
    "tags": ["vaultLibrarian", "search"],
    "expected_tools": ["vaultLibrarian_searchContent"],
    "notes": "New scenario, not part of the dataset"
  }
]
```

- **`full_coverage.json`** — one single-tool scenario for every tool defined in `tools/tool_schemas.json` (47 prompts total). Helpful for sanity-checking individual tool schemas.
- **`tool_combos.json`** — curated multi-step workflows that require chaining two or more tools; the order of entries in `expected_tools` reflects the preferred call sequence.

> `expected_tools` is always an ordered list. Single-entry lists imply a dedicated tool, while multi-entry lists indicate the recommended execution order for chained evaluations.

## Extending

- Add new prompt files under `Evaluator/prompts/`.
- Implement richer reporters (CSV, HTML) by extending `reporting.py`.
- Plug different backends (OpenAI, vLLM) by implementing the same interface as `OllamaClient`.
