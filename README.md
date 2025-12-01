# Synaptic Tuner

Synthetic data and training stack for teaching local LLMs to run the Obsidian Nexus toolset.

<div align="center">
  <img src="https://picoshare-production-7223.up.railway.app/-JRwnJvYt5S" alt="Synaptic Tuner Banner" width="800"/>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CUDA 12+](https://img.shields.io/badge/CUDA-12+-76B900.svg?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![W&B Optional](https://img.shields.io/badge/W%26B-optional-FFBE00.svg?logo=weightsandbiases&logoColor=black)](https://wandb.ai/)

</div>

## Choose a path
- **Beginner (no setup):** Run the Colab notebook `Trainers/notebooks/sft_colab_beginner.ipynb`. It walks through SFT, exports checkpoints, and keeps you on a free GPU (unless you want to pay for it).
- **Local/production:** Use the unified CLI (`./run.sh` on Linux/WSL, `.\run.ps1` on PowerShell, or `python tuner.py` if your env is already active). It covers training, uploads, evaluation, and the full pipeline.

## Quick start

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

This flow keeps contribution friction low—no separate scripts—while helping us converge on the strongest local model for the Obsidian Nexus plugin.

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

## Repository map
- `Trainers/notebooks/` - notebooks (start with `sft_colab_beginner.ipynb`; others cover KTO, Nebius, evaluation).
- `tuner/` - unified CLI used by `run.sh` and `run.ps1`.
- `Trainers/rtx3090_sft` and `Trainers/rtx3090_kto` - local configs and scripts for SFT and KTO.
- `Evaluator/` - evaluation CLIs, prompt sets, and result reports.
- `Datasets/` - datasets and metadata; validation utilities in `tools/`.
- `docs/` and `finetuning-strategy.md` - architecture and deep-dive notes.
- `CLAUDE.md` - project-wide development guide and FAQs.

## Contributing and support
- File issues or PRs with logs and dataset info when relevant.

## License
MIT.
