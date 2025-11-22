# Evaluator

Evaluation harness for tool-calling models served via [Ollama](https://ollama.com/) or [LM Studio](https://lmstudio.ai/). The Evaluator issues curated prompts, captures model replies, and validates tool-call structure using the same schema rules enforced in `tools/validate_syngen.py`.

## Quick Start (Platform-Specific)

### macOS / Linux
```bash
# Interactive mode (recommended - auto-detects models from LM Studio)
python3 evaluator

# Or create an alias to use 'python' instead of 'python3'
echo 'alias python=python3' >> ~/.zshrc  # macOS
echo 'alias python=python3' >> ~/.bashrc # Linux
source ~/.zshrc  # or ~/.bashrc
```

### Windows
```powershell
# Interactive mode
python evaluator

# Or if you have multiple Python versions:
py -3 evaluator
```

### What Interactive Mode Does
1. Connects to LM Studio and lists available models
2. Lets you select which model to test
3. Asks how many evaluation runs to perform
4. Runs full coverage tests (all 47 tools)
5. Generates JSON and Markdown reports automatically

**Note:** All examples below use `python3` for clarity. On Windows, use `python` or `py -3` instead.

## Components

| File | Purpose |
| --- | --- |
| `config.py` | Runtime configuration dataclasses + helpers (model name, host, prompt sets, limits) for both Ollama and LM Studio backends. |
| `ollama_client.py` | Minimal HTTP client for Ollama's `/api/chat` endpoint with retries and streaming hooks. |
| `lmstudio_client.py` | Minimal HTTP client for LM Studio's OpenAI-compatible `/v1/chat/completions` endpoint. |
| `prompt_sets.py` | Loader for JSON / JSONL prompt collections → `PromptCase` objects with tags and metadata. |
| `schema_validator.py` | Thin wrapper that reuses the dataset validator to score a single assistant reply. |
| `reporting.py` | Aggregates per-case results and produces JSON + Markdown summaries. |
| `runner.py` | Core orchestration loop (load prompts, call backend, validate response). |
| `cli.py` | Command-line entry point (`python -m Evaluator.cli --prompt-set prompts/baseline.json`). |
| `prompts/` | Baseline prompt sets (JSON arrays today, more formats later). |
| `results/` | Run artifacts (`.json`, `.md`); `.gitkeep` placeholder retains the folder in git. |

## Workflow

> Requirements: Python 3.10+ and `requests` installed in your environment (`pip install requests`).

### Option 1: Using Ollama (default)

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

### Option 2: Using LM Studio

1. **Load your model in LM Studio** and start the local server (default: `http://127.0.0.1:1234`)

2. **Run the evaluator with LM Studio backend**
   ```bash
   python3 -m Evaluator.cli \
     --backend lmstudio \
     --model your-model-name \
     --prompt-set Evaluator/prompts/baseline.json \
     --output Evaluator/results/run_lmstudio_$(date +%s).json
   ```

### Backend Configuration

Both backends support environment variables and CLI overrides:

**Ollama:**
- Environment: `OLLAMA_HOST` (default: 127.0.0.1), `OLLAMA_PORT` (default: 11434)
- CLI overrides: `--host`, `--port`

**LM Studio:**
- Environment: `LMSTUDIO_HOST` (default: 127.0.0.1), `LMSTUDIO_PORT` (default: 1234)
- CLI overrides: `--host`, `--port`

### Inspecting Results

- Console summary groups successes/failures by tag.
- JSON artifact contains raw prompts, responses, validator issues, runtime metadata.
- Optional Markdown export surfaces quick human-readable diffs.
- Use `--markdown <path>` to generate a Markdown summary.

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
- Add additional backends by implementing the same `.chat()` interface as `OllamaClient` and `LMStudioClient`:
  - The client must accept a `Sequence[Mapping[str, str]]` of messages
  - Return an object with `.message` (str), `.raw` (dict), and `.latency_s` (float) attributes
  - Update `runner.py` type hints to include the new client type

## LM Studio helper CLI

Prefer a purpose-built helper for LM Studio runs:

**macOS/Linux:**
```bash
# List hosted models
python3 -m Evaluator.lmstudio_cli list-models

# Run the full coverage suite (defaults to prompts/full_coverage.json)
python3 -m Evaluator.lmstudio_cli run --model <model-id>
```

**Windows:**
```powershell
# List hosted models
python -m Evaluator.lmstudio_cli list-models

# Run the full coverage suite
python -m Evaluator.lmstudio_cli run --model <model-id>
```

Artifacts are written to `Evaluator/results/<model>_full_coverage_<timestamp>.json` and `.md`. If `--model` is omitted, the CLI will prompt you to choose from the models reported by LM Studio.

## One-command interactive flow

For the simplest path, just run:

**macOS/Linux:**
```bash
python3 evaluator
```

**Windows:**
```powershell
python evaluator
```

You'll get a short prompt to pick a model (auto-listed from LM Studio) and how many times to run the full-coverage suite. Everything else is automatic—the run(s) will save JSON and Markdown artifacts under `Evaluator/results/` with the model name and timestamp.

## Troubleshooting

### "command not found: python" (macOS/Linux)
Use `python3` instead of `python`, or create an alias:
```bash
echo 'alias python=python3' >> ~/.zshrc  # macOS with zsh
echo 'alias python=python3' >> ~/.bashrc # Linux with bash
source ~/.zshrc  # reload shell config
```

### "Unable to list models from LM Studio"
1. Make sure LM Studio is running
2. Load a model in LM Studio (click "Load Model" in the UI)
3. Start the local server (Server tab → Start Server)
4. Verify it's running at `http://127.0.0.1:1234`

### "Module not found: requests"
Install the required dependency:
```bash
pip3 install requests  # macOS/Linux
pip install requests   # Windows
```

### Wrong Python version
Check your Python version (must be 3.10+):
```bash
python3 --version  # macOS/Linux
python --version   # Windows
```
