# Project Reference

Scripts, configuration files, environment variables, data patterns, and platform notes.

---

## Key Bash Scripts

**Root Level:**
- `run.sh` / `run.ps1` - Main CLI wrappers (auto-activate conda)
- `setup_env.sh` / `setup_env.ps1` - Environment setup

**Trainers:**
- `Trainers/sft/setup.sh` - Full SFT environment setup
- `Trainers/kto/setup.sh` - Full KTO environment setup
- `Trainers/local/jobs/*.yaml` - Local Docker training job configs (used by `python tuner.py local-run`; UID-agnostic, persistent-container mode)

**Tools:**
- `Tools/run_synth_chat.sh` / `.ps1` - Synthetic chat generation wrapper

**Dataset Improvement:**
- `.claude/skills/synthetic-data-generation/scripts/improve_dataset.sh` - Dataset improvement skill

---

## Configuration Files

**Training (Python dataclasses):**
- `Trainers/sft/configs/training_config.py` - SFT config (LR: 2e-4, epochs: 3)
- `Trainers/kto/configs/training_config.py` - KTO config (LR: 2e-7, epochs: 1)

**Datasets:**
- SFT: `Datasets/syngen_tools_sft_11.18.25.jsonl` (2,676 positive examples)
- KTO: `Datasets/syngen_tools_11.18.25.jsonl` (4,649 interleaved examples)

**SynthChat (Dataset Improvement):**
- `SynthChat/config/config.yaml` - Main config
- `SynthChat/rubrics/*.yaml` - Quality rubrics

**Synth Chat (Generation):**
- `synth_chat/config/config.yaml` - Generation config
- `synth_chat/configs/agents.yaml` - Agent configs
- `synth_chat/configs/behaviors.yaml` - Behavior configs

---

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

---

## Data Patterns

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
sft_output/YYYYMMDD_HHMMSS/
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
cd Trainers/sft
tail -f sft_output/YYYYMMDD_HHMMSS/logs/training_latest.jsonl
```

---

## Platform Notes

**WSL2 (Recommended):**
- Full compatibility, better performance, all scripts work

**Windows PowerShell:**
- Use `.ps1` scripts
- Some multiprocessing limitations
- Prefer WSL2 if possible

---

## Capability Matrix

| Task | Fully Auto | Needs User Input | Notes |
|------|:----------:|:----------------:|-------|
| Environment setup | X | | `./setup_env.sh` |
| Dependency install | X | | `./run.sh doctor --fix` |
| List resources | X | | `./run.sh list *` |
| Dataset validation | X | | `python3 .skills/synethetic-data-generation/scripts/validate_syngen.py` |
| System diagnostics | X | | `./run.sh doctor` |
| Training (SFT/KTO) | | X | Needs dataset choice, model size |
| Local Docker training | | X | `python tuner.py local-run --job-config Trainers/local/jobs/<config>.yaml`; UID-agnostic, persistent-container mode |
| Evaluation | | X | Needs model path, scenario set |
| Upload to HuggingFace | | X | Needs repo name, HF_TOKEN |
| Dataset improvement | | X | Needs rubrics, line range |
| Synthetic data gen | | X | Needs config, teacher model |

---

## Key Documentation

- `README.md` - Project overview
- `Trainers/sft/README.md` - SFT training guide
- `Trainers/kto/README.md` - KTO training guide
- `SynthChat/README.md` - Dataset improvement guide
- `KTO_TRAINING_REFERENCE.md` - KTO interleaving requirement
- `docs/EVOLUTIONARY_FINETUNING.md` - Unified validation & evolutionary training design
- `shared/validation/README.md` - Shared validation module guide
- `docs/` - Architecture specs

---

## Getting Help

- Check script help: `python script.py --help`
- Run dry runs: `python train_sft.py --dry-run`
- Validate first: `python3 .skills/synethetic-data-generation/scripts/validate_syngen.py dataset.jsonl`

**Key Principle:** Use the bash scripts (`./run.sh`, `setup.sh`, etc.) rather than direct Python when possible - they handle environment setup, dependency checks, and provide better UX.
