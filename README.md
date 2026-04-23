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
- The repo ships with project skills covering the entire workflow
- Skills use progressive disclosure — your agent loads only what it needs, when it needs it
- Best practices are encoded as protocols (dry-run before generation, interleave KTO datasets, etc.)
- You describe intent, your agent handles execution — *"train a 7B model on my dataset"* just works

Skills are written as Markdown — they work with any AI coding tool that supports project-level instructions. Claude Code is the reference integration, but the knowledge transfers to Cursor, Windsurf, Cline, Roo Code, and others. There's also a full interactive CLI and Colab notebooks if you prefer working without an agent.

## Cloud Workflow

Cloud training is now a first-class path, not an afterthought. The canonical Hugging Face Jobs flow is:

```bash
python tuner.py cloud-pipeline --method sft --preset full
```

That launches training on HF Jobs, saves artifacts to provider-native storage, and hands the exact finished run into cloud evaluation automatically. For full experiment orchestration, use:

```bash
python tuner.py run-experiment --experiment-spec Trainers/cloud/experiments/<spec>.yaml --yes
```

That path can run train -> evaluation -> exact loss -> analysis -> recommendation from one checked-in experiment spec.

**Cloud surfaces built into the repo:**
- `cloud-pipeline` for the standard HF Jobs train-then-evaluate workflow
- `run-experiment` for full cloud experiment bundles and comparisons
- `cloud-eval` to re-evaluate an existing bucket-backed run on remote GPU
- `cloud-jobs` to inspect live HF Jobs status and logs
- `bucket` to analyze, read, list, pull, and push bucket-backed artifacts
- `plan-hardware` for blind hardware planning using live HF Jobs flavors and pricing
- `cloud-gym` to run the vault gym against a trained cloud run

## Recent Updates

- HF Jobs is now the canonical cloud path for train + evaluate, with `cloud-pipeline` handling the common workflow end-to-end.
- Cloud evaluation writes structured artifacts back into the source run, including `evaluation_results.json`, `evaluation_results.md`, and `evaluation_lineage.json`.
- Bucket-backed progress is a first-class UX: training and evaluation stream JSONL progress that the local dashboard can replay.
- `run-experiment` now supports fuller cloud orchestration, including post-training evaluation and exact-loss stages as separate sibling jobs by default.
- `plan-hardware` and `scripts/hf_jobs_hardware.py` make hardware selection less guessy by using the live HF Jobs hardware surface.
- Evolutionary SFT is now supported in the cloud experiment path through checked-in specs and `cloud-pipeline --train-*` overrides.

## Local Docker Training

If you have a local GPU and want Docker-isolated Unsloth training without the usual UID/GID permission headaches, use `local-run`:

```bash
python tuner.py local-run --job-config Trainers/local/jobs/qwen35_2b_sft_smoke.yaml
```

This runs the training stack inside a container with the asciimatics dashboard visible in your terminal, and writes artifacts back to the host with your own user's ownership. Opting into `job.persist: true` keeps a long-lived container around so repeat runs skip pip install and model download. Checked-in starter configs live under `Trainers/local/jobs/` (smoke + 2-epoch SFT).

Cloud remains the primary path; `local-run` is the supporting path when you want to iterate on a local GPU with Docker isolation.

## Quick Start

| Path | How |
|------|-----|
| **Claude Code (recommended)** | Open repo in [Claude Code](https://docs.anthropic.com/en/docs/claude-code) and tell it what you want |
| **HF Jobs cloud train + eval** | `python tuner.py cloud-pipeline --method sft --preset full` |
| **Full cloud experiment bundle** | `python tuner.py run-experiment --experiment-spec Trainers/cloud/experiments/<spec>.yaml --yes` |
| **Local Docker training** | `python tuner.py local-run --job-config Trainers/local/jobs/qwen35_2b_sft_smoke.yaml` |
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
| Run HF Jobs train + eval | *"Launch the canonical cloud pipeline for this SFT run"* | `fine-tuning` |
| Run a full cloud experiment | *"Run this experiment spec with eval and exact loss"* | `fine-tuning` |
| Inspect live cloud jobs | *"Show me the latest HF Jobs status and logs"* | `fine-tuning` |
| Evaluate a model | *"Compare my fine-tuned model against the base"* | `evaluation` |
| Upload to HuggingFace | *"Upload with GGUF quantizations"* | `upload-deployment` |
| Full pipeline | *"Train, evaluate, and upload if it looks good"* | All skills |

Skills use progressive disclosure — lean SKILL.md files auto-load, detailed reference docs in `reference/` load on demand. Best practices are enforced automatically (dry-run before generation, interleave KTO datasets, use merged_16bit for GGUF, etc.).

## Using with Other AI Coding Tools

The canonical skills live in `.skills/`, with synced copies in `.agents/skills/` for agent tooling. They are plain Markdown, so they work with any AI coding tool. Most platforms use `AGENTS.md` as their entrypoint (Claude Code uses `CLAUDE.md`). Copy the skill files to your platform's rules directory, or use the universal `.skills/` folder at your project root:

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
SynthChat (env-backed data)  →  SFT (local or HF Jobs)  →  cloud eval / exact loss  →  merge/publish Nexus model  →  KTO  →  env-GRPO  →  Upload/Deploy
```

| Stage | Tool | Key Config |
|-------|------|------------|
| **Generate env-backed data** | `python3 -m SynthChat.run generate` | `SynthChat/scenarios/`, `SynthChat/config/settings.yaml`, `SynthChat/config/targets_*.json` |
| **Project datasets** | `python3 SynthChat/scripts/project_rollout_datasets.py` | `Datasets/environment_rollouts/`, `Datasets/kto/`, `Datasets/grpo/` |
| **Train SFT** | `python tuner.py train` → `sft` | `Trainers/sft/configs/config.yaml` |
| **Canonical HF SFT + eval** | `python tuner.py cloud-pipeline --method sft --preset full` | HF Jobs + bucket-backed run artifacts |
| **Full cloud experiment** | `python tuner.py run-experiment --experiment-spec Trainers/cloud/experiments/<spec>.yaml --yes` | Experiment spec + parallel post-training stages |
| **Inspect live HF jobs** | `python tuner.py cloud-jobs list` / `python tuner.py cloud-jobs logs --job professorsynapse/<job-id> --tail 200` | Live HF Jobs status + raw logs |
| **Inspect bucket-backed artifacts** | `python tuner.py bucket analyze --path runs/hf_jobs/sft/<run-prefix>/` | `training_lineage.json`, eval artifacts, loss artifacts |
| **Blind hardware planning** | `python tuner.py plan-hardware --experiment-spec Trainers/cloud/experiments/<spec>.yaml` | Live HF Jobs hardware + pricing |
| **Train KTO** | `python tuner.py train` → `kto` | `Trainers/kto/configs/config.yaml` |
| **Train env-GRPO** | `python tuner.py train` → `grpo` | `Trainers/grpo/configs/env_config.yaml` |
| **Cloud env-GRPO** | `python tuner.py cloud-run --job-config Trainers/cloud/jobs/nexus_quark_l25_28_env_grpo.yaml` | HF Jobs config + `Trainers/grpo/configs/env_config.yaml` |
| **Evaluate** | `python -m Evaluator.cli --backend lmstudio --model MODEL` | `Evaluator/config/scenarios/` |
| **Upload / merge** | `python tuner.py modelops` or upload scripts | Hugging Face / Nexus model repos |

### Current GRPO split

- `Trainers/grpo/configs/config.yaml` is the older static projected-dataset GRPO path.
- `Trainers/grpo/configs/env_config.yaml` is the current environment-backed multi-step GRPO path.
- Local NVIDIA `train -> grpo` now routes to the env-backed trainer and will bootstrap the isolated GRPO runtime if it is missing.
- The canonical alignment flow is documented in [.skills/fine-tuning/protocols/environment-backed-alignment-pipeline.md](/Users/jrosenbaum/Documents/Code/Synthetic%20Conversations/.skills/fine-tuning/protocols/environment-backed-alignment-pipeline.md).

## Config-Driven Architecture

SynthChat is fully config-driven. **Nothing is hardcoded for a specific use case.** The included example formats (`useTools`, `getTools`, workspace structures, label mappings) are **toy demonstrations** showing how the system works — they are not canonical formats and should not be treated as ground truth.

Everything is configurable via YAML in `SynthChat/config/`:
- **`tool_call_formats.yaml`** — Define any tool-call response schema (wrapper name, context fields, call structure)
- **`workspace_formats.yaml`** — Define system prompt sections, tag names, default values
- **`label_mappings.yaml`** — Define issue classification rules and label rollup groups
- **`settings.yaml`** — Generation settings, model config, output paths

To add a new tool-call format (e.g., for your own agent framework), add a named entry to `tool_call_formats.yaml` and reference it from your scenario YAML. No code changes required.

## Dataset Format

JSONL with `conversations` array. Tool call structure is fully configurable — the example below uses the included `useTools` wrapper format, but you can define any format via `tool_call_formats.yaml`.

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
- **GRPO (static)**: Prompts + ground truth for reward scoring
- **env-GRPO**: Canonical SynthChat rollout records with environment config, stage reviews, and replayable prompts

## Repository Map

```
Synaptic-Tuner/
├── SynthChat/              # Synthetic data generation (scenarios, rubrics, config)
├── Trainers/
│   ├── sft/                # SFT training
│   ├── kto/                # KTO training
│   ├── grpo/               # Static GRPO + env-backed GRPO
│   └── notebooks/          # Colab notebooks (beginner + advanced)
├── Evaluator/              # Model evaluation (scenarios, backends, results)
├── Datasets/               # Training data + canonical environment rollouts
├── shared/                 # Shared infra (LLM client, upload, validation, UI)
├── tuner/                  # Unified CLI (used by run.sh)
├── .skills/                # Canonical project skills and protocols
├── .agents/skills/         # Synced skill copy for agent tooling
└── CLAUDE.md               # Project-wide dev guide
```

## Environment

```bash
# .env in repo root (auto-loaded by CLI)
HF_TOKEN=hf_...                       # HuggingFace (required for cloud jobs, buckets, uploads)
OPENROUTER_API_KEY=sk-or-...          # OpenRouter (for generation/improvement)
WANDB_API_KEY=...                     # Weights & Biases (optional)
LMSTUDIO_HOST=localhost               # LM Studio host
OLLAMA_HOST=http://localhost:11434    # Ollama endpoint
```

## License

MIT.
