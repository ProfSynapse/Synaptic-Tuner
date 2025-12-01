# Toolset-Training

<div align="center">
  <img src="https://picoshare-production-7223.up.railway.app/-JRwnJvYt5S" alt="Toolset-Training Banner" width="800"/>
  
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
  [![GitHub Stars](https://img.shields.io/github/stars/ProfSynapse/Synaptic-Tuner?style=social)](https://github.com/ProfSynapse/Synaptic-Tuner)
  [![GitHub Issues](https://img.shields.io/github/issues/ProfSynapse/Synaptic-Tuner)](https://github.com/ProfSynapse/Synaptic-Tuner/issues)
  [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/ProfSynapse/Synaptic-Tuner/pulls)
  [![CUDA 12.1+](https://img.shields.io/badge/CUDA-12.1+-76B900.svg?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
  [![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
  [![Weights & Biases](https://img.shields.io/badge/Weights%20&%20Biases-Integrated-FFBE00.svg?logo=weightsandbiases&logoColor=black)](https://wandb.ai/)
</div>

---

**Synthetic dataset generation and LLM fine-tuning system** for training local language models to reliably use the **Claudesidian-MCP toolset** for Obsidian vault operations.

> Train small, local models (3B-20B parameters) to call tools as reliably as Claude, ChatGPT, or Copilot using supervised fine-tuning (SFT) and preference learning (KTO).

---

## 🚀 Quick Start

### Option 1: Colab Notebooks (Easiest - No Setup Required)

**For beginners and quick experiments:**

1. **SFT Training** (Recommended for first-time users):
   - Open [`Trainers/notebooks/sft_colab_beginner.ipynb`](Trainers/notebooks/sft_colab_beginner.ipynb) in Google Colab
   - Free T4 GPU included
   - ~45 minutes training time
   - Produces a 7B model that understands tool calling

2. **Advanced SFT Training** (For tool-calling datasets):
   - Open [`Trainers/notebooks/sft_colab_tool_calling.ipynb`](Trainers/notebooks/sft_colab_tool_calling.ipynb)
   - Optimized for Claudesidian-MCP dataset format
   - Includes HuggingFace upload and GGUF export

3. **KTO Refinement** (After SFT):
   - Open [`Trainers/notebooks/kto_colab_notebook.ipynb`](Trainers/notebooks/kto_colab_notebook.ipynb)
   - Preference learning to refine model quality
   - Requires contrastive dataset (positive + negative examples)

**Why notebooks first?**
- ✅ No local setup required
- ✅ Free GPU access (Google Colab T4)
- ✅ Visual progress tracking
- ✅ Automatic checkpoint management
- ✅ One-click HuggingFace upload

### Option 2: Local Training (For Production)

**Prerequisites:**
- NVIDIA GPU with 10GB+ VRAM (RTX 3090 recommended)
- WSL2 (Ubuntu) or native Linux
- CUDA 12.1+

**Setup & Training:**

**Linux/WSL2:**
```bash
# 1. Clone repository
git clone https://github.com/ProfSynapse/Synaptic-Tuner.git
cd Synaptic-Tuner

# 2. Setup environment (one-time)
cd Trainers/rtx3090_sft
bash setup.sh

# 3. Interactive training CLI (recommended)
cd ../
./train.sh

# The CLI will let you choose:
#   1) SFT Only - Supervised Fine-Tuning
#   2) KTO Only - Preference Learning
#   3) SFT → KTO Pipeline - Full training (recommended)
```

**Windows PowerShell:**
```powershell
# 1. Clone repository
git clone https://github.com/ProfSynapse/Synaptic-Tuner.git
cd Synaptic-Tuner

# 2. Setup environment (one-time)
cd Trainers\rtx3090_sft
# Follow setup instructions in README

# 3. Interactive training CLI (recommended)
cd ..\
.\train.ps1

# The CLI will let you choose:
#   1) SFT Only - Supervised Fine-Tuning
#   2) KTO Only - Preference Learning
#   3) SFT → KTO Pipeline - Full training (recommended)
```

**Interactive CLI Features:**
- 📋 **Choose training mode** - SFT, KTO, or combined pipeline
- ⚙️ **Configure W&B logging** - Enable/disable with custom project names
- 📊 **Configuration preview** - See all settings before starting
- ✅ **Auto-detection** - Finds Python, verifies CUDA automatically
- 🔗 **Seamless chaining** - Pipeline mode auto-updates configs
- 💻 **Cross-platform** - Identical experience on Linux and Windows

**See full local setup guide:** [Trainers/rtx3090_sft/README.md](Trainers/rtx3090_sft/README.md)

### Option 3: Evaluate Trained Models

**Using LM Studio (Recommended - Visual Interface):**

```bash
# 1. Load your model in LM Studio (http://localhost:1234)

# 2. Run interactive evaluation
python evaluator.py

# OR use the CLI directly
python -m Evaluator.lmstudio_cli run --model your-model-name

# 3. Results saved to Evaluator/results/
```

**Using Ollama (CLI-focused):**

```bash
# 1. Serve model via Ollama
ollama run your-model-name

# 2. Run evaluation
python -m Evaluator.cli \
  --model your-model-name \
  --prompt-set Evaluator/prompts/tool_prompts.json \
  --output Evaluator/results/run_$(date +%s).json \
  --markdown Evaluator/results/report.md
```

**Evaluation outputs:**
- JSON with per-prompt results, validator scores, and latency
- Markdown report with success/failure breakdown
- Tool coverage analysis

**See evaluation guide:** [Evaluator/README.md](Evaluator/README.md)

---

## 📁 Repository Structure

```
Synaptic-Tuner/
├── Trainers/
│   ├── notebooks/              # 🎯 START HERE - Colab notebooks
│   │   ├── sft_colab_beginner.ipynb           # ⭐ Best for first-time users
│   │   ├── sft_colab_tool_calling.ipynb       # Advanced SFT training
│   │   ├── kto_colab_notebook.ipynb           # KTO preference learning
│   │   └── kto_tool_calling_notebook.ipynb    # KTO for tool calling
│   ├── rtx3090_sft/            # Local SFT training (initial learning)
│   │   ├── train.sh            # Training wrapper script
│   │   ├── train_sft.py        # Main training script
│   │   ├── setup.sh            # Environment setup
│   │   └── configs/            # Training configurations
│   ├── rtx3090_kto/            # Local KTO training (refinement)
│   │   ├── train.sh            # Training wrapper script
│   │   ├── train_kto.py        # Main training script
│   │   └── configs/            # Training configurations
│   └── mistral_lora_mac/       # Apple Silicon (M1/M2/M3) training
│
├── Datasets/                   # Training data (ChatML format)
│   ├── syngen_tools_sft_11.22.25.jsonl        # ⭐ Latest SFT dataset (2,889 examples)
│   ├── syngen_tools_11.18.25.jsonl            # KTO dataset (4,649 examples)
│   └── syngen_toolset_v1.0.0_*.jsonl          # Legacy datasets
│
├── Evaluator/                  # Model testing harness
│   ├── cli.py                  # Generic evaluation CLI
│   ├── lmstudio_cli.py         # LM Studio-specific CLI
│   ├── interactive_cli.py      # Interactive evaluation
│   ├── prompts/                # Test prompt sets
│   │   ├── tool_prompts.json  # One prompt per tool (47 prompts)
│   │   ├── baseline.json       # General scenarios
│   │   └── tool_combos.json    # Multi-step workflows
│   └── results/                # Evaluation outputs (JSON + Markdown)
│
├── tools/                      # Validation utilities
│   ├── validate_syngen.py      # Dataset validator
│   ├── analyze_tool_coverage.py # Tool coverage analysis
│   └── tool_schemas.json       # Tool definitions (47+ tools)
│
└── docs/                       # Architecture & guides
    ├── SCHEMA_VERIFICATION_REFERENCE.md
    └── WORKSPACE_*.md          # Workspace documentation
```

---

## 🎓 Training Guide

### SFT First, KTO Second

**Training Pipeline:**
1. **SFT (Supervised Fine-Tuning)** - Teaches WHAT tool calling is
   - Uses positive examples only
   - Higher learning rate (2e-4)
   - 3 epochs
   - Result: Model learns tool syntax and formatting

2. **KTO (Preference Learning)** - Teaches WHICH tool calls are better
   - Uses positive + negative examples
   - Very low learning rate (2e-7)
   - 1 epoch
   - Result: Model prefers high-quality tool calls

### Training Methods Comparison

| Aspect | SFT (rtx3090_sft) | KTO (rtx3090_kto) |
|--------|------------------|-------------------|
| **Purpose** | Initial training | Refinement |
| **Dataset** | Positive examples only | Positive + negative |
| **Learning Rate** | 2e-4 (high) | 2e-7 (very low) |
| **Epochs** | 3 | 1 |
| **Batch Size** | 6 | 4 |
| **Training Time** | ~45 min | ~15 min |
| **Use When** | Starting from scratch | Improving existing model |

### Notebook Training (Colab)

**Start here if you're new:**

1. **Open notebook:** [`Trainers/notebooks/sft_colab_beginner.ipynb`](Trainers/notebooks/sft_colab_beginner.ipynb)
2. **Connect to free GPU:** Click "Connect" → Runtime → Change runtime type → T4 GPU
3. **Run all cells:** Click Runtime → Run all
4. **Wait ~45 minutes:** Progress bars show training status
5. **Download model:** Final cell exports to HuggingFace or Google Drive

**Advanced users:**
- Use [`sft_colab_tool_calling.ipynb`](Trainers/notebooks/sft_colab_tool_calling.ipynb) for tool-calling datasets
- Use [`kto_colab_notebook.ipynb`](Trainers/notebooks/kto_colab_notebook.ipynb) for preference learning

### Local Training (RTX 3090)

**Full setup instructions:** See [Trainers/rtx3090_sft/README.md](Trainers/rtx3090_sft/README.md)

**Quick start:**

```bash
# SFT Training (initial learning)
cd Trainers/rtx3090_sft
./train.sh --model-size 7b

# KTO Training (refinement)
cd ../rtx3090_kto
./train.sh --model-size 7b --local-file ../../Datasets/syngen_tools_11.18.25.jsonl
```

**Configuration:**
- Edit `configs/training_config.py` for advanced settings
- Model sizes: 3b (fast), 7b (recommended), 13b (quality), 20b (specialized)
- Monitor logs: `tail -f sft_output_rtx3090/*/logs/training_latest.jsonl`

**VRAM Requirements:**
- 3B: ~8-10 GB
- 7B: ~9-11 GB
- 13B: ~14-16 GB
- 20B: ~18-20 GB

---

## 📊 Datasets

### Current Datasets

| Dataset | Examples | Type | Purpose |
|---------|----------|------|---------|
| `syngen_tools_sft_11.22.25.jsonl` | 2,889 | SFT | ⭐ **Latest SFT training** |
| `syngen_tools_11.18.25.jsonl` | 4,649 | KTO | Interleaved True/False |
| `syngen_toolset_v1.0.0_claude.jsonl` | 5,120 | Legacy | Original Claude dataset |
| `syngen_toolset_v1.0.0_chatgpt.jsonl` | 1,088 | Legacy | ChatGPT dataset |

### Dataset Format (ChatML)

```json
{
  "conversations": [
    {
      "role": "user",
      "content": "User request"
    },
    {
      "role": "assistant",
      "content": "tool_call: toolName\narguments: {...}\n\nResult: {...}\n\nResponse to user"
    }
  ],
  "label": true
}
```

**Key requirements:**
- ✅ NO system message (starts with user role)
- ✅ Context object as FIRST parameter in all tool calls
- ✅ All 7 context fields required (sessionId, workspaceId, sessionDescription, sessionMemory, toolContext, primaryGoal, subgoal)
- ✅ `sessionMemory` never empty
- ✅ Single-turn conversations (multi-turn removed in 11/18/25)

### Validate Dataset

```bash
python tools/validate_syngen.py Datasets/syngen_tools_sft_11.22.25.jsonl
```

---

## 🔍 Evaluation

### Quick Evaluation (Interactive)

```bash
# Interactive prompt - choose model from LM Studio
python evaluator.py
```

### LM Studio CLI (Recommended)

```bash
# List available models
python -m Evaluator.lmstudio_cli list-models

# Run full coverage evaluation (47 prompts, one per tool)
python -m Evaluator.lmstudio_cli run --model your-model-name

# Results saved to:
# - Evaluator/results/your-model-name_full_coverage_TIMESTAMP.json
# - Evaluator/results/your-model-name_full_coverage_TIMESTAMP.md
```

### Advanced Evaluation

```bash
# Custom prompt set
python -m Evaluator.cli \
  --backend lmstudio \
  --model your-model-name \
  --prompt-set Evaluator/prompts/tool_combos.json \
  --output Evaluator/results/combos_$(date +%s).json \
  --markdown Evaluator/results/combos_report.md

# Using Ollama instead
python -m Evaluator.cli \
  --backend ollama \
  --model your-model-name \
  --prompt-set Evaluator/prompts/baseline.json \
  --output Evaluator/results/run.json
```

### Prompt Sets

- **`tool_prompts.json`** - One prompt per tool (47 prompts)
- **`baseline.json`** - General scenarios
- **`tool_combos.json`** - Multi-step workflows

**See full evaluation guide:** [Evaluator/README.md](Evaluator/README.md)

---

## 🛠 Tool Coverage

**47+ tools across 5 agent categories:**

- **vaultManager** - File/folder operations (create, read, update, delete, move, copy)
- **contentManager** - CRUD operations for notes and content
- **memoryManager** - Session/state/workspace management
- **vaultLibrarian** - Advanced search, batch operations, metadata queries
- **agentManager** - Agent lifecycle, prompt execution, image generation

**Schema source:** `tools/tool_schemas.json`

---

## 📚 Documentation

### Getting Started
- **[This README]** - Overview and quick start
- **[Trainers/notebooks/sft_colab_beginner.ipynb]** - Interactive training tutorial
- **[Evaluator/README.md]** - Evaluation guide

### Advanced Training
- **[Trainers/rtx3090_sft/README.md]** - Local SFT training
- **[Trainers/rtx3090_kto/README.md]** - Local KTO training
- **[CLAUDE.md]** - Comprehensive development guide

### Reference
- **[docs/SCHEMA_VERIFICATION_REFERENCE.md]** - Tool schema reference
- **[finetuning-strategy.md]** - Master strategy document (203KB)
- **[KTO_TRAINING_REFERENCE.md]** - KTO-specific notes

---

## 🖥 Platform Support

| Platform | SFT | KTO | Notebooks | Status |
|----------|-----|-----|-----------|--------|
| **Google Colab** | ✅ | ✅ | ✅ | ⭐ Recommended for beginners |
| **WSL2 / Linux** | ✅ | ✅ | ✅ | ⭐ Best for production |
| **Windows PowerShell** | ⚠️ | ⚠️ | ❌ | Limited (use WSL2) |
| **macOS (Apple Silicon)** | ✅ | ❌ | ✅ | MLX-based (see mistral_lora_mac) |

**Windows users:** Strongly recommend WSL2 for full compatibility. PowerShell has known issues with multiprocessing.

---

## 🔧 Troubleshooting

### CUDA Out of Memory

**Reduce batch size:**
```bash
python train_sft.py --model-size 7b --batch-size 4 --gradient-accumulation 6
```

**Or reduce sequence length:**
```bash
python train_sft.py --model-size 7b --max-seq-length 1024
```

### Training Logs Not Appearing

**Fixed in 11/18/25.** Logs now correctly write to:
- `sft_output_rtx3090/YYYYMMDD_HHMMSS/logs/training_YYYYMMDD_HHMMSS.jsonl`
- `kto_output_rtx3090/YYYYMMDD_HHMMSS/logs/training_YYYYMMDD_HHMMSS.jsonl`

**Monitor in real-time:**
```bash
tail -f sft_output_rtx3090/*/logs/training_latest.jsonl
```

### Model Outputs Generic Text Instead of Tool Calls

**Diagnosis:** Wrong training method for initial training.

**Solution:** Use SFT first, then optionally refine with KTO.
1. Train with `rtx3090_sft` using positive examples (teaches tool syntax)
2. Optionally refine with `rtx3090_kto` using contrastive examples

### Dataset Validation Failures

**Common issues:**
- Missing context object (must be first parameter)
- Empty `sessionMemory` field (never allowed)
- Incorrect ID format (must match: `session_<13digits>_<9chars>`)
- Multi-turn conversations (removed in 11/18/25 update)

**Fix:**
```bash
python tools/validate_syngen.py your_dataset.jsonl
```

---

## 📦 Environment Setup

**Create `.env` file in repository root:**

```bash
# HuggingFace (required for uploads, optional for training)
HF_TOKEN=hf_your_token_here

# Weights & Biases (optional)
WANDB_API_KEY=your_wandb_key
```

**Getting tokens:**
- HuggingFace: https://huggingface.co/settings/tokens (create with "write" access)
- W&B: https://wandb.ai/authorize

---

## 🤝 Contributing

### Adding New Datasets

1. Follow ChatML format (no system message)
2. Include complete context objects (all 7 fields)
3. Never leave sessionMemory empty
4. Use proper ID formats (13 digits_9 chars)
5. Validate before merging:
   ```bash
   python tools/validate_syngen.py your_new_dataset.jsonl
   ```

### Reporting Issues

- GitHub Issues: https://github.com/ProfSynapse/Synaptic-Tuner/issues
- Include: dataset version, training logs, error messages

---

## 📜 License

Generated using Claude (Anthropic) for the Claudesidian-MCP project.

---

## 🎯 Next Steps

1. **New users:** Start with [`Trainers/notebooks/sft_colab_beginner.ipynb`](Trainers/notebooks/sft_colab_beginner.ipynb)
2. **Advanced users:** Setup local training with [Trainers/rtx3090_sft/README.md](Trainers/rtx3090_sft/README.md)
3. **Evaluation:** Use LM Studio CLI: `python -m Evaluator.lmstudio_cli run`
4. **Questions:** Check [CLAUDE.md](CLAUDE.md) for comprehensive development guide

**Last Updated:** 2025-11-22
