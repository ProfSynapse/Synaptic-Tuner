# CLAUDE.md

Quick reference for Claude Code when working in this repository.

## Repository Purpose

Synthetic dataset generation and LLM fine-tuning system for training models to use the Claudesidian-MCP toolset (Obsidian vault operations). Teacher models generate training data, which is then used for SFT/KTO fine-tuning.

## Quick Start

```bash
# Interactive CLI (recommended)
./run.sh              # Linux/WSL
.\run.ps1             # Windows

# Or directly
python tuner.py       # Auto-detects conda environment
```

## Project Structure

```
Toolset-Training/
├── tuner.py                    # Main CLI entry point
├── run.sh / run.ps1            # Platform wrappers (auto-activate conda)
├── setup_env.sh / setup_env.ps1 # Environment setup
│
├── Datasets/                   # Training data (JSONL format)
│   ├── behavior_datasets/      # Behavioral training (thinking + non-thinking)
│   └── tools_datasets/         # Tool-specific training (thinking + non-thinking)
│
├── Trainers/
│   ├── rtx3090_sft/           # SFT training (initial training)
│   │   ├── setup.sh           # Full environment setup
│   │   ├── train_sft.py       # Training entry point
│   │   └── configs/           # Training configuration
│   │
│   ├── rtx3090_kto/           # KTO training (refinement)
│   │   ├── setup.sh           # Full environment setup
│   │   ├── train_kto.py       # Training entry point
│   │   └── configs/           # Training configuration
│   │
│   └── shared/                # Shared code (upload, model loading, utilities)
│
├── improvement_engine/        # Dataset quality improvement
│   ├── batch_improve.py       # Batch improvement script
│   ├── parallel_batch.py      # Parallel batch processing
│   └── services/              # LLM service, validators
│
├── synth_chat/                # Synthetic chat generation
│   ├── run_generation.py      # Main generator
│   └── configs/               # Generation configs
│
├── Evaluator/                 # Model testing harness
│   └── cli.py                 # Evaluation CLI
│
├── Tools/                     # Dataset utilities
│   ├── validate_syngen.py     # Dataset validator
│   ├── run_synth_chat.sh/ps1  # Synthetic chat wrapper
│   └── analyze_tool_coverage.py
│
├── shared/                    # Shared infrastructure
│   ├── llm/                   # Unified LLM client (OpenRouter, LMStudio, Ollama)
│   ├── upload/                # Upload framework
│   └── utilities/             # Path, env utilities
│
└── web-ui/                    # Next.js dataset editor
    └── npm run dev            # Start dev server
```

## Common Tasks

### 1. Training a Model

**Via CLI (Recommended):**
```bash
./run.sh
# Select: Train -> NVIDIA GPU -> SFT (for initial) or KTO (for refinement)
```

**Direct Python:**
```bash
# SFT (initial training)
cd Trainers/rtx3090_sft
python train_sft.py --model-size 7b

# KTO (refinement)
cd Trainers/rtx3090_kto
python train_kto.py --model-size 7b
```

**Key Difference:**
- **SFT**: Teaches tool-calling from scratch (positive examples only)
- **KTO**: Refines existing model (needs interleaved True/False examples)

### 2. Uploading to HuggingFace

**Via CLI (Recommended):**
```bash
./run.sh
# Select: Upload -> Choose training run -> Configure save method
```

**Direct Python:**
```bash
cd Trainers/rtx3090_sft  # or rtx3090_kto
python src/upload_to_hf.py \
  ./sft_output_rtx3090/YYYYMMDD_HHMMSS/final_model \
  username/model-name \
  --save-method merged_16bit \
  --create-gguf
```

### 3. Generating Synthetic Data

```bash
# Interactive mode
./Tools/run_synth_chat.sh

# Quick test (100 examples)
./Tools/run_synth_chat.sh --quick
```

### 4. Improving Dataset Quality

```bash
# Test on single example
python test_improvement.py

# Batch improvement
cd improvement_engine
python batch_improve.py
```

### 5. Validating Datasets

```bash
python Tools/validate_syngen.py Datasets/your_dataset.jsonl
```

### 6. Evaluating Models

```bash
# Via Evaluator
python -m Evaluator.cli \
  --model your-model-name \
  --prompt-set Evaluator/prompts/tool_prompts.json
```

## Key Bash Scripts

**Root Level:**
- `run.sh` / `run.ps1` - Main CLI wrappers (auto-activate conda)
- `setup_env.sh` / `setup_env.ps1` - Environment setup

**Trainers:**
- `Trainers/rtx3090_sft/setup.sh` - Full SFT environment setup
- `Trainers/rtx3090_kto/setup.sh` - Full KTO environment setup

**Tools:**
- `Tools/run_synth_chat.sh` / `.ps1` - Synthetic chat generation wrapper

**Dataset Improvement:**
- `.claude/skills/synthetic-data-generation/scripts/improve_dataset.sh` - Dataset improvement skill

## Configuration Files

**Training (Python dataclasses):**
- `Trainers/rtx3090_sft/configs/training_config.py` - SFT config (LR: 2e-4, epochs: 3)
- `Trainers/rtx3090_kto/configs/training_config.py` - KTO config (LR: 2e-7, epochs: 1)

**Datasets:**
- SFT: `Datasets/syngen_tools_sft_11.18.25.jsonl` (2,676 positive examples)
- KTO: `Datasets/syngen_tools_11.18.25.jsonl` (4,649 interleaved examples)

**Improvement Engine:**
- `improvement_engine/config/config.yaml` - Main config
- `improvement_engine/rubrics/*.yaml` - Quality rubrics

**Synth Chat:**
- `synth_chat/config/config.yaml` - Generation config
- `synth_chat/configs/agents.yaml` - Agent configs
- `synth_chat/configs/behaviors.yaml` - Behavior configs

## Environment Variables

Create `.env` in repo root:

```bash
# HuggingFace (required for uploads)
HF_TOKEN=hf_your_token_here

# OpenRouter (for improvement engine)
OPENROUTER_API_KEY=sk-or-...

# LM Studio (if using local models)
LMSTUDIO_HOST=localhost  # or 192.168.x.x for WSL

# Ollama (if using local models)
OLLAMA_HOST=http://localhost:11434

# Weights & Biases (optional)
WANDB_API_KEY=your_wandb_key
```

## Common Patterns

### Dataset Format (ChatML)

```jsonl
{
  "conversations": [
    {"role": "user", "content": "User request"},
    {"role": "assistant", "content": "tool_call: toolName\narguments: {...}\n\nResult: {...}\n\nResponse"}
  ],
  "label": true
}
```

**Key Rules:**
- NO system message (starts with user role)
- `label`: `true` = positive, `false` = negative (for KTO only)
- Single-turn conversations preferred
- Context object must be first parameter in tool calls

### Training Output Structure

```
sft_output_rtx3090/YYYYMMDD_HHMMSS/
├── checkpoints/           # Training checkpoints
├── final_model/          # LoRA adapters
├── logs/                 # Training metrics (JSONL)
└── model-name/           # Created by upload (if uploaded)
    ├── lora/
    ├── merged-16bit/
    ├── merged-4bit/
    └── gguf/
```

### Monitoring Training

```bash
# Real-time log viewing
cd Trainers/rtx3090_sft
tail -f sft_output_rtx3090/YYYYMMDD_HHMMSS/logs/training_latest.jsonl
```

## Platform Notes

**WSL2 (Recommended):**
- Full compatibility
- Better performance
- All scripts work

**Windows PowerShell:**
- Use `.ps1` scripts
- Some multiprocessing limitations
- Prefer WSL2 if possible

## Troubleshooting

**CUDA OOM:**
```bash
# Reduce batch size
python train_sft.py --model-size 7b --batch-size 4
```

**Missing dependencies:**
```bash
./setup_env.sh  # Auto-installs everything
```

**Training logs not appearing:**
- Check `logs/training_latest.jsonl` exists
- Verify `run_dir` path in callbacks

## Key Documentation

- `README.md` - Project overview
- `Trainers/rtx3090_sft/README.md` - SFT training guide
- `Trainers/rtx3090_kto/README.md` - KTO training guide
- `improvement_engine/README.md` - Dataset improvement guide
- `KTO_TRAINING_REFERENCE.md` - KTO interleaving requirement
- `docs/` - Architecture specs

## Getting Help

- Check script help: `python script.py --help`
- Run dry runs: `python train_sft.py --dry-run`
- Validate first: `python Tools/validate_syngen.py dataset.jsonl`

---

**Key Principle:** Use the bash scripts (`./run.sh`, `setup.sh`, etc.) rather than direct Python when possible - they handle environment setup, dependency checks, and provide better UX.
