# Synaptic Tuner

**An agentic-first toolkit for building custom LLMs.** Generate synthetic training data, fine-tune models, evaluate quality, and deploy — all driven by AI agents that know the system end-to-end.

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

## The Problem

Fine-tuning a local LLM is a multi-step pipeline that most people never finish. You need to generate quality training data, format it correctly, pick the right training method, configure dozens of hyperparameters, evaluate the results, then figure out GGUF quantization and deployment. Each step has its own tools, formats, and gotchas. Most tutorials cover one piece — you're left stitching the rest together yourself.

## The Solution

Synaptic Tuner handles the full pipeline in one repo, and it's designed to be **operated by AI coding agents**. Instead of memorizing CLI flags and YAML schemas, you describe what you want in plain English. Built-in skills give your agent deep knowledge of every component — it generates data, trains models, runs evaluations, and deploys, enforcing best practices at each step.

**Agentic-first means:**
- The repo ships with 4 agent skills covering the entire workflow
- Skills use progressive disclosure — your agent loads only what it needs, when it needs it
- Best practices are encoded as protocols (dry-run before generation, interleave KTO datasets, etc.)
- You describe intent, your agent handles execution — *"train a 7B model on my dataset"* just works

Skills are written as Markdown — they work with any AI coding tool that supports project-level instructions. Claude Code is the reference integration, but the knowledge transfers to Cursor, Windsurf, Cline, Roo Code, and others. There's also a full interactive CLI and Colab notebooks if you prefer working without an agent.

## Quick Start

| Path | How |
|------|-----|
| **Claude Code (recommended)** | Open repo in [Claude Code](https://docs.anthropic.com/en/docs/claude-code) and tell it what you want |
| **Interactive CLI** | `./run.sh` (Linux/WSL) or `.\run.ps1` (PowerShell) |
| **Beginner (no GPU)** | `Trainers/notebooks/sft_colab_beginner.ipynb` in Google Colab |

## Using with Claude Code

This repo is built to be operated by Claude Code. It has skills covering the entire pipeline — just describe what you want.

**Setup:** *"Set this repo up for me"* — Claude checks your platform, runs environment setup, helps create `.env` with credentials, and verifies with a dry run.

**What you can ask:**

| Task | Example | Skill |
|------|---------|-------|
| Generate training data | *"Generate 50 examples of the search scenario"* | `synthetic-data-generation` |
| Write scenarios/rubrics | *"Write a new scenario for content operations"* | `synthetic-data-generation` |
| Train a model | *"Train a 7B model on my dataset"* | `fine-tuning` |
| Evaluate a model | *"Compare my fine-tuned model against the base"* | `evaluation` |
| Upload to HuggingFace | *"Upload with GGUF quantizations"* | `upload-deployment` |
| Full pipeline | *"Train, evaluate, and upload if it looks good"* | All skills |

Skills use progressive disclosure — lean SKILL.md files auto-load, detailed reference docs in `reference/` load on demand. Best practices are enforced automatically (dry-run before generation, interleave KTO datasets, use merged_16bit for GGUF, etc.).

## Using with Other AI Coding Tools

The skills in `.claude/skills/` are plain Markdown — they work with any AI coding tool. Most platforms use `AGENTS.md` as their entrypoint (Claude Code uses `CLAUDE.md`). Copy the skill files to your platform's rules directory, or use the universal `.skills/` folder at your project root:

| Platform | Where to put skills |
|----------|-------------------|
| **Cursor** | `.cursor/rules/` (rename `.md` → `.mdc`) |
| **Windsurf** | `.windsurf/rules/` |
| **Cline** | `.clinerules/` |
| **Roo Code** | `.roo/rules/` |
| **Amazon Q** | `.amazonq/rules/` |
| **JetBrains AI** | `.aiassistant/rules/` |
| **Augment** | `.augment/rules/` |
| **Kilo Code** | `.kilocode/rules/` |
| **Tabnine** | `.tabnine/guidelines/` |
| **Zed, Aider, GitHub Copilot, others** | `.skills/` at project root |

Most platforms auto-discover Markdown in their rules directory. For tools that use `AGENTS.md`, point it at the skills folder or reference the skill files directly.

## The Pipeline

```
Generate Data (SynthChat)  →  Train (SFT → KTO → GRPO)  →  Evaluate  →  Upload to HF
```

| Stage | Tool | Key Config |
|-------|------|------------|
| **Generate** | `python -m SynthChat.run generate` | `SynthChat/scenarios/`, `SynthChat/rubrics/`, `SynthChat/config/settings.yaml` |
| **Improve** | `python -m SynthChat.run improve` | Rubric YAMLs define judge/improver prompts |
| **Train SFT** | `python train_sft.py --model-size 7b` | `Trainers/rtx3090_sft/configs/config.yaml` |
| **Train KTO** | `python train_kto.py --model-size 7b` | `Trainers/rtx3090_kto/configs/config.yaml` |
| **Train GRPO** | `python train_grpo.py` | `Trainers/rtx3090_grpo/configs/config.yaml` |
| **Evaluate** | `python -m Evaluator.cli --backend lmstudio --model MODEL` | `Evaluator/config/scenarios/` |
| **Upload** | `python src/upload_to_hf.py MODEL user/repo --save-method merged_16bit` | Supports LoRA, merged 16-bit, GGUF |

## Dataset Format

JSONL with `conversations` array. Tool call structure is fully configurable.

```json
{
  "conversations": [
    {"role": "user", "content": "Create a new folder called Projects"},
    {"role": "assistant", "content": null, "tool_calls": [{"type": "function", "function": {"name": "createFolder", "arguments": "{\"path\": \"/Projects\"}"}}]}
  ],
  "label": true
}
```

- **SFT**: Positive examples only, `label` ignored
- **KTO**: Interleaved `true`/`false` labels required
- **GRPO**: Prompts + ground truth for reward scoring

## Repository Map

```
Synaptic-Tuner/
├── SynthChat/              # Synthetic data generation (scenarios, rubrics, config)
├── Trainers/
│   ├── rtx3090_sft/        # SFT training
│   ├── rtx3090_kto/        # KTO training
│   ├── rtx3090_grpo/       # GRPO training
│   └── notebooks/          # Colab notebooks (beginner + advanced)
├── Evaluator/              # Model evaluation (scenarios, backends, results)
├── Datasets/               # Training data (JSONL)
├── shared/                 # Shared infra (LLM client, upload, validation, UI)
├── tuner/                  # Unified CLI (used by run.sh)
├── .claude/skills/         # Agent skills (4 skills, 22 reference docs — works with any AI coding tool)
└── CLAUDE.md               # Project-wide dev guide
```

## Environment

```bash
# .env in repo root (auto-loaded by CLI)
HF_TOKEN=hf_...                       # HuggingFace (required for uploads)
OPENROUTER_API_KEY=sk-or-...          # OpenRouter (for generation/improvement)
WANDB_API_KEY=...                     # Weights & Biases (optional)
LMSTUDIO_HOST=localhost               # LM Studio host
OLLAMA_HOST=http://localhost:11434    # Ollama endpoint
```

## License

MIT.
