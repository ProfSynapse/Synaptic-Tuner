# Synaptic Tuner

Synthetic data generation and fine-tuning toolkit for training local LLMs on custom datasets.

<div align="center">
  <img src="https://picoshare-production-7223.up.railway.app/-JRwnJvYt5S" alt="Synaptic Tuner Banner" width="800"/>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CUDA 12+](https://img.shields.io/badge/CUDA-12+-76B900.svg?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![W&B Optional](https://img.shields.io/badge/W%26B-optional-FFBE00.svg?logo=weightsandbiases&logoColor=black)](https://wandb.ai/)

</div>

<div align="center">
  <a href="https://github.com/unslothai/unsloth">
    <img src="https://raw.githubusercontent.com/unslothai/unsloth/main/images/unsloth%20logo%20white%20text.png" alt="Unsloth logo" width="360">
  </a>
  <br/>
  <sub><i>Training is powered by <a href="https://github.com/unslothai/unsloth">Unsloth</a> — huge thanks to their team.</i></sub>
</div>

## Choose a path
- **Beginner (no setup):** Run the Colab notebook `Trainers/notebooks/sft_colab_beginner.ipynb`. It walks through SFT, exports checkpoints, and keeps you on a free GPU (unless you want to pay for it).
- **Local/production:** Use the unified CLI (`./run.sh` on Linux/WSL, `.\run.ps1` on PowerShell, or `python tuner.py` if your env is already active). It covers training, uploads, evaluation, and the full pipeline.

## Quick start

### Configure credentials
- Copy `.env.example` to `.env` in the repo root, then add your Hugging Face write token (`HF_TOKEN=hf_...`). A write-scoped token is needed for downloading private models and uploading checkpoints; `HF_API_KEY` works as an alias if you prefer.
- (Optional) Set `HF_USERNAME` to prefill upload prompts, and `WANDB_API_KEY` if you want Weights & Biases logging during training.
- The CLI now auto-loads the root `.env` when you run `./run.sh`, `.\run.ps1`, or `python tuner.py`—no manual `export` needed.
- In Google Colab (beginner notebook), click the 🔑 **Secrets** icon, add a secret named `HF_TOKEN`, and paste the same write token there; the notebook reads it securely at runtime.

### Beginner notebook
1. Open `Trainers/notebooks/sft_colab_beginner.ipynb` in Google Colab.
2. Choose your GPU (recommend L4 for 8B models and below)
3. Fill out the forms in various cells to choose your model, datasets, hyperparameters, etc.
4. Run all cells; the notebook handles dataset download, training, and optional export (Hugging Face or Drive).
3. Bring the exported model into LM Studio/Ollama or continue with the CLI for evaluation.

### Unified CLI
Requirements: CUDA-capable GPU for training and a Python env (the setup scripts create or activate the `unsloth_latest` conda env).

```bash
git clone <repo-url> && cd Toolset-Training
./run.sh            # Linux/WSL interactive menu
.\run.ps1           # PowerShell wrapper (uses WSL for GPU flows)
python tuner.py     # If your Python env is already active
```

Pick a subcommand when prompted: `train`, `upload`, `eval`, or `pipeline` (train -> upload -> eval).

### Evaluation with LM Studio (WSL users)

If running evaluations from WSL against LM Studio on Windows, you need to configure network access:

1. **In LM Studio (Windows):**
   - Click **Developer** in the left sidebar
   - Go to **Server** settings
   - Toggle ON **Serve on Local Network**
   - Note the IP address shown (e.g., `192.168.1.104`)

2. **Add to your `.env` file:**
   ```bash
   LMSTUDIO_HOST=192.168.1.104
   ```

3. **Run evaluation:**
   ```bash
   ./run.sh eval
   ```

The IP address may change when your network changes—check LM Studio's server panel for the current address.

### Evaluate an existing model (Beginner Colab Notebook)
Use the beginner Colab notebook to run evaluations (training optional). Direct link:
**Colab Notebook:** [`sft_colab_beginner.ipynb`](https://github.com/ProfSynapse/Toolset-Training/blob/cli-refactor/Trainers/notebooks/sft_colab_beginner.ipynb)

Open it in Colab via "Open Notebook" > GitHub tab, paste the repo URL, then select the file, or upload it directly.

**In the Colab notebook (evaluation section near end):**
1. Scroll to the evaluation section after the training/export blocks.
2. Enter one or more model identifiers (matching what your local LM Studio or Ollama exposes).
3. Select a prompt set (start with `Evaluator/prompts/tool_prompts.json`).
4. Run the evaluation cells; they will generate JSON + Markdown outputs under `Evaluator/results/`.
5. Download those result files from the Colab file browser.
6. Open a PR including:
   - The JSON + Markdown files.
   - A short qualitative note: tool selection accuracy, context retention, hallucinations, typical failure cases.

Results always land in `Evaluator/results/` inside the repo workspace.

This flow keeps contribution friction low—no separate scripts—while helping converge on the strongest local model for your use case.

## Bring your own data
- The stack ships with our synthetic tool-calling and behavior datasets under `Datasets/`, but you can point the CLI to any JSONL that matches the expected format (set `local_file` in configs or pass the path when prompted).
- Keep datasets and metadata together; each file is self-describing.
- Validate before training: `python tools/validate_syngen.py <path-to-dataset>`.
- Update training params or swap datasets locally via `Trainers/rtx3090_sft/configs/config.yaml` (SFT) and `Trainers/rtx3090_kto/configs/config.yaml` (KTO).

### Expected format (SFT and KTO)
- JSONL with a `conversations` (or `messages`) array in OpenAI tool-calling style.
- Roles: `system` (optional), `user`, `assistant`.
- Assistant tool calls use `tool_calls` entries; the trainer will render these into text for chat templates.
- Each tool call must include a `context` argument with:
  - `sessionId`, `workspaceId`
  - `sessionDescription`, `sessionMemory`
  - `toolContext`, `primaryGoal`, `subgoal`
- For KTO/preference data, include `label` (true for preferred, false for dispreferred). For SFT, labels are optional but ignored.

Example record:

```json
{
  "conversations": [
    { "role": "system", "content": "<session_context>...embed context here...</session_context>" },
    { "role": "user", "content": "Request from user" },
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [
        {
          "type": "function",
          "function": {
            "name": "vaultManager_createFolder",
            "arguments": "{\"context\": {\"sessionId\": \"session_...\", \"workspaceId\": \"default\", \"sessionDescription\": \"...\", \"sessionMemory\": \"...\", \"toolContext\": \"...\", \"primaryGoal\": \"...\", \"subgoal\": \"...\"}, \"path\": \"/target/path\"}"
          }
        }
      ]
    }
  ],
  "label": true
}
```

## Improvement Engine

The improvement engine automatically improves dataset quality using LLM-based judging and iterative refinement. It validates and fixes common issues like hallucinated content, malformed thinking blocks, and inconsistent formatting.

### Quick start

```bash
# Via CLI menu
./run.sh
# Select: [6] Improvement Engine

# Direct command
python -m improvement_engine.services.rubric_runner \
  --file Datasets/tools_datasets/thinking/agentManager/tools_v1.7.jsonl \
  --output Datasets/tools_datasets/thinking/agentManager/tools_v1.8.jsonl \
  --rubrics factuality,thinking_quality \
  --max-iterations 3

# List available rubrics
python -m improvement_engine.services.rubric_runner --list
```

### Available rubrics

| Rubric | Scope | Description |
|--------|-------|-------------|
| `factuality` | thinking, response | Ensures all details are grounded in system prompt or user request |
| `thinking_quality` | thinking | Validates thinking block structure and content quality |
| `system_prompt_format` | system_prompt | Validates XML tag structure and required fields |
| `response_quality` | response | Checks response formatting and appropriateness |
| `tool_alignment` | response | Validates tool calls match user intent |
| `destructive_safety` | thinking | Ensures destructive operations have proper safeguards |
| `confidence_calibration` | thinking | Checks confidence scores are well-calibrated |
| `context_alignment` | thinking | Validates thinking aligns with provided context |

Run `--list` to see all available rubrics with descriptions.

### Unified validation architecture

The improvement engine uses a shared validation infrastructure (`shared/validation/`) that's also used by the Evaluator and training pipelines. This provides:

- **Format-agnostic parsing**: Auto-detects Qwen, Mistral, ChatML, and OpenAI tool call formats
- **Config-driven validation**: Same YAML rubric format works across all systems
- **Cross-scope validation**: Validate content across different parts of the response

### Cross-scope validation

Rubrics can validate content across different scopes. For example, the factuality rubric extracts dates and file paths from the thinking block and validates they exist in the system prompt or user request:

```yaml
# From improvement_engine/rubrics/factuality.yaml
validations:
  - cross_scope:
      from: thinking
      to: [system_prompt, user]
      extract:
        fields: [memory, goal, requirements]
        pattern: '\b(?:20\d{2}[-/]\d{1,2}[-/]\d{1,2}|...)\b'
      validate_in_content: true
    error: "HALLUCINATED DATE: '{value}' not found in inputs"
```

**Example transformation:**
- **Before:** `"memory": "User has been working on this project since 2025-12-07"`
- **After:** `"memory": "User wants to create a checkpoint for their current work"`

Dates that exist in inputs are preserved; only hallucinated dates are removed.

### Backend options

The improvement engine supports multiple LLM backends:

```bash
# LM Studio (default)
--backend lmstudio --host localhost --port 1234

# Ollama
--backend ollama

# OpenRouter (requires OPENROUTER_API_KEY in .env)
--backend openrouter
```

### Interaction logging

All judge/improver interactions are logged for KTO training data generation:

```bash
# Logs are saved to:
improvement_engine/interactions/interactions_<dataset>_<timestamp>.jsonl

# View latest interactions
ls -lt improvement_engine/interactions/ | head -5
```

## Repository map
- `Trainers/notebooks/` - notebooks (start with `sft_colab_beginner.ipynb`; others cover KTO, Nebius, evaluation).
- `tuner/` - unified CLI used by `run.sh` and `run.ps1`.
- `Trainers/rtx3090_sft` and `Trainers/rtx3090_kto` - local configs and scripts for SFT and KTO.
- `Evaluator/` - evaluation CLIs, prompt sets, and result reports.
- `Datasets/` - datasets and metadata; validation utilities in `tools/`.
- `improvement_engine/` - dataset quality improvement with LLM-based judging and rubrics.
- `shared/` - shared infrastructure used across all modules:
  - `validation/` - unified validation (format-agnostic parsing, config-driven validators, rubric loading)
  - `llm/` - unified LLM client (OpenRouter, LMStudio, Ollama)
  - `utilities/` - common utilities (paths, env, YAML loading)
- `docs/` and `finetuning-strategy.md` - architecture and deep-dive notes.
- `CLAUDE.md` - project-wide development guide and FAQs.

## Contributing and support
- File issues or PRs with logs and dataset info when relevant.

## License
MIT.
