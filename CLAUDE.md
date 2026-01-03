# CLAUDE.md

Quick reference for Claude Code when working in this repository.

---

## AI Assistant Quick Reference

**Before starting any task:**
1. Check system state: `./run.sh status` (or `python tuner.py status`)
2. If issues detected: `./run.sh doctor` for full diagnostics
3. Discover resources: `./run.sh list [datasets|models|runs|rubrics]`

**Key Commands:**
| Command | Description |
|---------|-------------|
| `./run.sh` | Interactive menu (recommended entry point) |
| `./run.sh status` | Quick system health check |
| `./run.sh doctor` | Full diagnostics with fix suggestions |
| `./run.sh doctor --fix` | Auto-fix common issues |
| `./run.sh list datasets` | Show available training datasets |
| `./run.sh list runs` | Show completed training runs |
| `./run.sh list rubrics` | Show available improvement rubrics |

**Pre-flight Checklist:**
- [ ] Environment ready? (`./run.sh status`)
- [ ] Required tokens set? (HF_TOKEN, OPENROUTER_API_KEY)
- [ ] GPU available? (`nvidia-smi`)
- [ ] LLM backend running? (LM Studio, Ollama, or OpenRouter)

---

## Important Rules

- **Never save output files to /tmp** - Keep all generated files within the repository (e.g., `docs/`, `Datasets/`, or create a `scratch/` folder)
- Test outputs should go to `docs/test_*.jsonl` or similar
- **Be greedy to stop on errors** - When testing, monitor output and kill immediately if something looks wrong. Fix and retest quickly rather than waiting for long runs to complete. Early exit = faster iteration.

## Repository Purpose

Synthetic dataset generation and LLM fine-tuning system. Teacher models generate training data, which is then used for SFT/KTO fine-tuning of smaller models.

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
├── SynthChat/                 # Synthetic chat generation & dataset improvement
│   ├── run_generation.py      # Main generator
│   ├── services/              # Rubric runner, validators, improvement
│   │   └── rubric_runner.py   # Dataset quality improvement
│   ├── rubrics/               # Quality rubrics (YAML)
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
│   ├── utilities/             # Path, env, YAML loading utilities
│   └── validation/            # Unified validation (used by SynthChat, Evaluator, Trainer)
│       ├── parsing/           # Format-agnostic response parsing (Qwen/Mistral/ChatML)
│       ├── validators/        # Config-driven validators (XML, JSON, YAML, regex, code)
│       └── rubric/            # Rubric loading and caching
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

### 4. Improving Dataset Quality (LM Studio)

**Direct Command:**
```bash
python -m SynthChat.services.rubric_runner \
  --file Datasets/tools_datasets/thinking/agentManager/tools_v1.7.jsonl \
  --output Datasets/tools_datasets/thinking/agentManager/tools_v1.8.jsonl \
  --rubrics system_prompt_format \
  --backend lmstudio \
  --start-line 1 \
  --end-line 3 \
  --max-iterations 3
```

**Options:**
- `--file` - Input JSONL file
- `--output` - Output JSONL file
- `--rubrics` - Comma-separated rubric names (e.g., `system_prompt_format,thinking_quality`)
- `--backend` - `lmstudio`, `ollama`, or `openrouter`
- `--host` - LM Studio host (default: localhost)
- `--port` - LM Studio port (default: 1234)
- `--start-line` / `--end-line` - Line range to process
- `--max-iterations` - Max improvement loops per example
- `--no-interactions` - Disable interaction logging (enabled by default)

**List available rubrics:**
```bash
python -m SynthChat.services.rubric_runner --list
```

**Interactive Menu (Alternative):**
```bash
./run.sh
# Select: [6] Improvement Engine (clean datasets)
```

**How it works:**
1. Loads example from dataset
2. Runs **schema validation** (YAML-driven: xml, json, regex, yaml, code)
3. Passes validation results TO judge prompt
4. Judge sees errors and gives targeted feedback
5. Improver fixes based on feedback
6. Logs interaction to `SynthChat/interactions/` for KTO training

**Checking Interactions:**
```bash
# View latest interactions file
ls -lt SynthChat/interactions/ | head -5

# Inspect judge prompt (shows schema validation results)
cat SynthChat/interactions/interactions_LATEST.jsonl | head -1 | jq '.conversations[1].content'
```

**What Judge Sees:**
```
============================================================
SCHEMA VALIDATION RESULTS
============================================================

❌ system_prompt_format: Schema validation FAILED
   - Missing required XML tag: <vault_structure>
   - Missing field in <selected_workspace>: workflows
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

---

## Decision Trees

Use these decision trees to guide task execution and avoid common pitfalls.

### Training a Model

```
START: User wants to train a model
  |
  v
[1] Check environment ready?
    Run: ./run.sh status
    |
    +-- NOT READY --> Run: ./run.sh doctor --fix --> Retry status
    |
    +-- READY --> Continue
          |
          v
[2] What datasets are available?
    Run: ./run.sh list datasets
    |
    v
[3] Determine training type needed:
    |
    +-- NEW MODEL (learning from scratch)
    |     |
    |     v
    |   USE SFT:
    |   - Needs positive-only examples (label: true)
    |   - Higher learning rate: 2e-4
    |   - Epochs: 2-3 typical
    |   - Command: python Trainers/rtx3090_sft/train_sft.py
    |
    +-- REFINING EXISTING MODEL
          |
          v
        USE KTO:
        - Needs interleaved true/false examples
        - Lower learning rate: 1e-6 to 2e-7
        - Epochs: 1 typical
        - Command: python Trainers/rtx3090_kto/train_kto.py
          |
          v
[4] ALWAYS run with --dry-run first to validate configuration
    |
    v
[5] If dry-run passes, run actual training
```

### Evaluating a Model

```
START: User wants to evaluate a model
  |
  v
[1] Is LLM backend running?
    |
    +-- LM Studio: Check http://localhost:1234/v1/models
    +-- Ollama: Check http://localhost:11434/api/tags
    +-- OpenRouter: Check OPENROUTER_API_KEY is set
    |
    +-- NOT RUNNING --> Start backend or set API key
    |
    +-- RUNNING --> Continue
          |
          v
[2] Find the model to evaluate:
    Run: ./run.sh list runs
    |
    v
[3] Choose scenario set:
    |
    +-- behavior_prompts: Test general behavior/reasoning
    +-- tool_prompts: Test tool-calling capability
    |
    v
[4] Run evaluation:
    python -m Evaluator.cli --model <model> --prompt-set <scenarios>
```

### Improving Dataset Quality

```
START: User wants to improve dataset quality
  |
  v
[1] Is LLM backend available?
    Check LM Studio, Ollama, or OpenRouter
    |
    +-- NOT AVAILABLE --> Start backend or configure API
    |
    +-- AVAILABLE --> Continue
          |
          v
[2] List available rubrics:
    Run: python -m SynthChat.services.rubric_runner --list
    |
    v
[3] Validate dataset first:
    Run: python Tools/validate_syngen.py <dataset_file>
    |
    +-- VALIDATION FAILED --> Fix JSON/format errors first
    |
    +-- VALIDATION PASSED --> Continue
          |
          v
[4] Test with small batch first:
    python -m SynthChat.services.rubric_runner \
      --file <input> --output <output> \
      --rubrics <rubric_names> \
      --start-line 1 --end-line 5
    |
    v
[5] If test passes, run on full dataset
```

### Generating Synthetic Data

```
START: User wants to generate synthetic training data
  |
  v
[1] Check SynthChat configuration:
    Review: synth_chat/config/config.yaml
    |
    v
[2] Is teacher model backend running?
    (Usually needs high-quality model like GPT-4, Claude, etc.)
    |
    +-- NOT AVAILABLE --> Configure OpenRouter or local model
    |
    +-- AVAILABLE --> Continue
          |
          v
[3] Quick test first:
    ./Tools/run_synth_chat.sh --quick
    |
    +-- ERRORS --> Check config and backend connectivity
    |
    +-- SUCCESS --> Continue
          |
          v
[4] Run full generation:
    ./Tools/run_synth_chat.sh
```

---

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

**SynthChat (Dataset Improvement):**
- `SynthChat/config/config.yaml` - Main config
- `SynthChat/rubrics/*.yaml` - Quality rubrics

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

## Diagnostics Guide

### Quick Health Check

```bash
# Full system diagnostics
./run.sh doctor

# Auto-fix common issues
./run.sh doctor --fix
```

### Common Issues and Fixes

#### CUDA / GPU Issues

**"CUDA not available"**
```bash
# Check NVIDIA driver
nvidia-smi

# Check PyTorch CUDA version
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"

# Fix: Reinstall PyTorch with correct CUDA version
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**"CUDA out of memory" (OOM)**
```bash
# Option 1: Reduce batch size
python train_sft.py --model-size 7b --batch-size 4

# Option 2: Use smaller model
python train_sft.py --model-size 3b

# Option 3: Enable gradient checkpointing (in config)
# Option 4: Clear GPU memory
nvidia-smi --gpu-reset
```

#### LLM Backend Issues

**"LM Studio not reachable"**
```bash
# Check if LM Studio is running
curl http://localhost:1234/v1/models

# WSL users: Use Windows host IP instead of localhost
# Find Windows IP: cat /etc/resolv.conf | grep nameserver
# Update .env: LMSTUDIO_HOST=<windows_ip>

# Ensure "Serve on Local Network" is enabled in LM Studio settings
```

**"Ollama connection refused"**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama service
ollama serve

# Check available models
ollama list
```

**"OpenRouter API error"**
```bash
# Verify API key is set
echo $OPENROUTER_API_KEY

# Test API connectivity
curl https://openrouter.ai/api/v1/models \
  -H "Authorization: Bearer $OPENROUTER_API_KEY"
```

#### Dataset Issues

**"Dataset validation failed"**
```bash
# Run validation to see specific errors
python Tools/validate_syngen.py <file>

# Common fixes:
# - Check JSON syntax (missing commas, quotes)
# - Ensure "conversations" array exists
# - Verify role is "user" or "assistant"
# - Check "label" field for KTO datasets
```

**"No examples found in dataset"**
```bash
# Check file is not empty
wc -l <dataset_file>

# Check file format (should be JSONL)
head -1 <dataset_file> | python -m json.tool
```

#### Training Issues

**"Training logs not appearing"**
- Check `logs/training_latest.jsonl` exists in run directory
- Verify `run_dir` path in callbacks configuration
- Check disk space: `df -h`

**"Loss is NaN"**
- Learning rate too high - reduce by 10x
- Data contains invalid values - run validation
- Gradient explosion - enable gradient clipping

**"Training stuck / no progress"**
- Check GPU utilization: `watch nvidia-smi`
- Verify data loader is working
- Check for deadlocks in multi-GPU setup

#### Environment Issues

**"Missing dependencies"**
```bash
# Full environment setup
./setup_env.sh

# Or with auto-fix
./run.sh doctor --fix

# Manual pip install
pip install -r requirements.txt
```

**"Module not found"**
```bash
# Ensure conda environment is active
conda activate toolset

# Check PYTHONPATH includes repo root
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

---

## Capability Matrix

What can be fully automated vs. what needs user input:

| Task | Fully Auto | Needs User Input | Notes |
|------|:----------:|:----------------:|-------|
| Environment setup | X | | `./setup_env.sh` |
| Dependency install | X | | `./run.sh doctor --fix` |
| List resources | X | | `./run.sh list *` |
| Dataset validation | X | | `python Tools/validate_syngen.py` |
| System diagnostics | X | | `./run.sh doctor` |
| Training (SFT/KTO) | | X | Needs dataset choice, model size |
| Evaluation | | X | Needs model path, scenario set |
| Upload to HuggingFace | | X | Needs repo name, HF_TOKEN |
| Dataset improvement | | X | Needs rubrics, line range |
| Synthetic data gen | | X | Needs config, teacher model |

**Legend:** X = Supported

---

## Recovery Procedures

### Training Crashed Mid-Run

```bash
# 1. Find the last checkpoint
ls -la Trainers/rtx3090_sft/sft_output_rtx3090/<run_id>/checkpoints/

# 2. Check checkpoint integrity
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('<checkpoint_path>')"

# 3. Resume from checkpoint (if trainer supports it)
python train_sft.py --resume-from-checkpoint <checkpoint_path>

# 4. Or restart fresh with same config
python train_sft.py --config <original_config>
```

### Out of GPU Memory During Training

```bash
# Immediate fix: Kill process and clear memory
pkill -f train_sft
nvidia-smi --gpu-reset  # If needed

# Config fixes (in order of impact):
# 1. Reduce batch size (most effective)
# 2. Use gradient accumulation instead of large batch
# 3. Enable gradient checkpointing
# 4. Use smaller model variant (7B -> 3B)
# 5. Use 4-bit quantization for base model
```

### Evaluation Giving Weird Results

```bash
# 1. Verify model is fully loaded
python -c "from transformers import AutoModelForCausalLM; m = AutoModelForCausalLM.from_pretrained('<model_path>'); print(m)"

# 2. Check prompt format matches training format
# Compare: Evaluator/prompts/*.json with training data format

# 3. Verify sampling settings
# - Temperature: 0.0-0.3 for deterministic, 0.7-1.0 for creative
# - top_p: 0.9 typical
# - max_tokens: ensure sufficient for response

# 4. Test with a simple known-good prompt
python -c "
from transformers import pipeline
pipe = pipeline('text-generation', model='<model_path>')
print(pipe('Hello, how are you?', max_new_tokens=50))
"
```

### Dataset Improvement Not Working

```bash
# 1. Verify LLM backend is responding
curl http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"test"}],"max_tokens":10}'

# 2. Check rubric exists
python -m SynthChat.services.rubric_runner --list

# 3. Validate input file format
python Tools/validate_syngen.py <input_file>

# 4. Run with verbose logging
python -m SynthChat.services.rubric_runner \
  --file <input> --output <output> \
  --rubrics <rubric> --start-line 1 --end-line 1 \
  --verbose
```

### Upload to HuggingFace Failed

```bash
# 1. Verify HF_TOKEN is set and valid
python -c "from huggingface_hub import HfApi; HfApi().whoami()"

# 2. Check disk space for merge operation
df -h

# 3. Verify model path exists and is complete
ls -la <model_path>/

# 4. Try upload with smaller chunks
python src/upload_to_hf.py <model_path> <repo_name> \
  --save-method lora  # Upload just LoRA adapters first
```

### Synthetic Data Generation Errors

```bash
# 1. Check config is valid YAML
python -c "import yaml; yaml.safe_load(open('synth_chat/config/config.yaml'))"

# 2. Verify teacher model is accessible
# (depends on backend - LM Studio, OpenRouter, etc.)

# 3. Run with minimal config for testing
./Tools/run_synth_chat.sh --quick --dry-run

# 4. Check output directory is writable
touch Datasets/test_write && rm Datasets/test_write
```

---

## Troubleshooting Quick Reference

| Symptom | Likely Cause | Quick Fix |
|---------|--------------|-----------|
| "CUDA not available" | PyTorch/CUDA mismatch | Reinstall PyTorch |
| "Connection refused" | Backend not running | Start LM Studio/Ollama |
| "Module not found" | Wrong environment | `conda activate toolset` |
| "Permission denied" | File permissions | `chmod +x script.sh` |
| "Out of memory" | Batch too large | Reduce `--batch-size` |
| "Loss is NaN" | Learning rate too high | Reduce LR by 10x |
| "No examples found" | Wrong file format | Check JSONL format |
| "API key invalid" | Token expired/wrong | Update `.env` file |

## Key Documentation

- `README.md` - Project overview
- `Trainers/rtx3090_sft/README.md` - SFT training guide
- `Trainers/rtx3090_kto/README.md` - KTO training guide
- `SynthChat/README.md` - Dataset improvement guide
- `KTO_TRAINING_REFERENCE.md` - KTO interleaving requirement
- `docs/EVOLUTIONARY_FINETUNING.md` - Unified validation & evolutionary training design
- `shared/validation/README.md` - Shared validation module guide
- `docs/` - Architecture specs

## Getting Help

- Check script help: `python script.py --help`
- Run dry runs: `python train_sft.py --dry-run`
- Validate first: `python Tools/validate_syngen.py dataset.jsonl`

---

**Key Principle:** Use the bash scripts (`./run.sh`, `setup.sh`, etc.) rather than direct Python when possible - they handle environment setup, dependency checks, and provide better UX.
