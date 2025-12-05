# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

This is a synthetic dataset generation and LLM fine-tuning system designed to train local language models to reliably use the **Claudesidian-MCP toolset** for Obsidian vault operations. Teacher models (Claude, ChatGPT, Copilot) generate synthetic training data, which is then used to fine-tune smaller models using KTO (Kahneman-Tversky Optimization) or LoRA methods.

## Project Structure

```
Toolset-Training/
├── tuner.py               # Unified CLI entry point
├── run.sh                 # Bash wrapper (auto-activates conda)
├── run.ps1                # PowerShell wrapper
├── Datasets/              # Synthetic training data in ChatML format (JSONL)
├── Tools/                 # Dataset validation and analysis utilities
├── Trainers/
│   ├── shared/            # Shared modules across trainers
│   │   ├── upload/        # Upload framework (SOLID architecture)
│   │   ├── model_loading/ # Model loader abstractions
│   │   └── utilities/     # Path, env utilities
│   ├── mistral_lora_mac/  # Apple Silicon (MLX) LoRA fine-tuning
│   ├── rtx3090_kto/       # NVIDIA GPU (Unsloth) KTO fine-tuning
│   └── rtx3090_sft/       # NVIDIA GPU (Unsloth) SFT fine-tuning
├── Evaluator/             # Model testing harness (Ollama/LM Studio)
└── docs/                  # Architecture specs and setup guides
```

## Common Development Commands

### Dataset Validation

```bash
# Validate a dataset file
python tools/validate_syngen.py Datasets/syngen_toolset_v1.0.0_claude.jsonl

# Analyze tool coverage
python tools/analyze_tool_coverage.py Datasets/syngen_toolset_v1.0.0_claude.jsonl
```

### Self-Play Data Generation

**NEW!** Generate synthetic training data using your fine-tuned model:

```bash
# Interactive mode (recommended)
./Tools/run_selfplay.sh

# Quick test (100 examples)
./Tools/run_selfplay.sh --quick

# Standard generation (1000 examples)
./Tools/run_selfplay.sh --standard

# Large generation (5000 examples)
./Tools/run_selfplay.sh --large

# PowerShell (Windows)
.\Tools\run_selfplay.ps1
```

**What it does:**
1. Sends prompts to your fine-tuned model via LM Studio
2. Validates responses automatically
3. Collects both correct and incorrect examples
4. Creates interleaved KTO datasets (True/False pattern)
5. Ready for immediate KTO training

**See:** [docs/SELF_PLAY_GENERATION.md](docs/SELF_PLAY_GENERATION.md) for complete guide.

### Setup & Installation

#### WSL2 / Linux (Recommended)

```bash
cd Trainers/rtx3090_kto

# Full setup with verification
bash setup.sh

# Quick setup (no verification tests)
bash setup.sh --quick

# Setup with Flash Attention (optional, takes 5-10 min)
bash setup.sh --with-flash-attn
```

#### Windows PowerShell

**Not recommended** - Use WSL2 for best compatibility. Windows has known issues with multiprocessing and some dependencies.

### Unified CLI (Recommended)

The project now has a unified CLI that handles all operations from the repo root:

```bash
# Interactive mode (recommended)
python tuner.py

# Direct commands
python tuner.py train     # Training submenu
python tuner.py upload    # Upload submenu
python tuner.py eval      # Evaluation submenu
python tuner.py pipeline  # Full pipeline (train -> upload -> eval)

# Platform wrappers (auto-detect conda)
./run.sh          # Bash/WSL
.\run.ps1         # PowerShell
```

**Features:**
- Cross-platform (WSL, Linux, Windows, Mac)
- Auto-detects conda environment
- Interactive menus with configuration preview
- Supports all platforms: NVIDIA RTX (SFT/KTO) and Apple Silicon (MLX)
- Full pipeline automation

### Training

> **Recommended:** Use `python tuner.py train` from repo root for interactive training setup.

#### SFT (Supervised Fine-Tuning) - RECOMMENDED FOR INITIAL TRAINING

Use SFT to teach the model tool-calling behavior from scratch. SFT uses direct supervision with only positive examples.

**Via Unified CLI (Recommended):**

```bash
# From repo root - interactive selection of platform, method, model size, dataset
python tuner.py train
```

**Direct Python (Advanced):**

```bash
cd Trainers/rtx3090_sft

# Recommended: 7B model training (uses default SFT dataset)
python train_sft.py --model-size 7b

# With custom dataset
python train_sft.py --model-size 7b --local-file ../../Datasets/my_data.jsonl

# With W&B logging
python train_sft.py --model-size 7b --wandb --wandb-project my-project

# Dry run (setup without training)
python train_sft.py --model-size 7b --dry-run
```

**SFT Configuration:**
- Dataset: `syngen_tools_sft_11.18.25.jsonl` (2,676 positive examples)
- Learning rate: `2e-4` (100x higher than KTO)
- Epochs: `3` (vs KTO's 1)
- Batch size: `6` (vs KTO's 4)
- Method: Direct supervision on tool-calling patterns
- **NO system prompt** - Model internalizes patterns naturally

#### KTO (Preference Learning) - USE AFTER SFT

Use KTO to refine an already-trained model by teaching it to prefer better tool calls over worse ones.

**Via Unified CLI (Recommended):**

```bash
# From repo root - interactive selection
python tuner.py train
# Then select: NVIDIA GPU -> KTO
```

**Direct Python (Advanced):**

```bash
cd Trainers/rtx3090_kto

# Recommended: 7B model training (uses default KTO dataset)
python train_kto.py --model-size 7b

# With custom dataset
python train_kto.py --model-size 7b --local-file ../../Datasets/my_data.jsonl

# With W&B logging
python train_kto.py --model-size 7b --wandb --wandb-project my-project

# Dry run (setup without training)
python train_kto.py --model-size 7b --dry-run
```

**KTO Configuration:**
- Dataset: `syngen_tools_11.18.25.jsonl` (4,649 interleaved examples)
- Learning rate: `2e-7` (very low for refinement)
- Epochs: `1` (preference learning is fast)
- Batch size: `4`
- Method: Preference learning (chosen vs rejected)
- **Requires interleaved dataset** (True/False/True/False pattern)

**Model size options:** `3b` (fast), `7b` (recommended), `13b` (quality), `20b` (specialized)

### Training (Mac / Apple Silicon)

**Via Unified CLI (Recommended):**

```bash
# From repo root - interactive selection
python tuner.py train
# Then select: Apple Silicon (M1/M2/M3)
```

**Direct Python (Advanced):**

```bash
cd Trainers/mistral_lora_mac

# Standard training
python main.py --config config/config.yaml

# Resume from checkpoint
python main.py --config config/config.yaml --resume checkpoints/checkpoint_step_500.npz

# Evaluation only
python main.py --config config/config.yaml --resume checkpoints/best_checkpoint.npz --eval-only
```

### Model Upload to HuggingFace

> **Recommended:** Use `python tuner.py upload` from repo root for interactive upload setup.

Upload organizes all artifacts within your training run directory:

```
sft_output_rtx3090/YYYYMMDD_HHMMSS/
├── checkpoints/
├── final_model/              # Original LoRA adapters
├── logs/
└── your-model-name/          # Created during upload
    ├── lora/                 # LoRA adapters
    ├── merged-16bit/         # Full merged model (if selected)
    ├── merged-4bit/          # 4-bit quantized (if selected)
    ├── gguf/                 # GGUF quantizations (if created)
    │   ├── your-model-name.gguf (f16)
    │   ├── your-model-name-Q4_K_M.gguf
    │   ├── your-model-name-Q5_K_M.gguf
    │   └── your-model-name-Q8_0.gguf
    ├── upload_manifest.json  # Upload metadata
    └── README.md             # Auto-generated docs
```

**Benefits:**
- All artifacts stay with their training run
- No more scattered `gguf_output/` directories
- Automatic cleanup of temporary files (~14GB saved per upload)
- Complete traceability via manifest

#### Via Unified CLI (Recommended)

```bash
# From repo root - interactive selection
python tuner.py upload

# The CLI will:
# 1. List available training runs
# 2. Prompt for model name
# 3. Select save method (16bit/4bit/lora)
# 4. Optional GGUF creation
# 5. Show organized output structure
# 6. Upload and create manifest/README
```

#### Direct Python (Advanced)

```bash
cd Trainers/rtx3090_sft  # or rtx3090_kto

python src/upload_to_hf.py \
  ./sft_output_rtx3090/20251122_143000/final_model \
  username/model-name \
  --save-method merged_16bit \
  --create-gguf

# Output will be auto-organized in:
# ./sft_output_rtx3090/20251122_143000/model-name/
```

**Save methods:**
- `merged_16bit` - Full quality, ~14GB (recommended)
- `merged_4bit` - Smaller size, ~3.5GB
- `lora` - LoRA adapters only, ~320MB

### GGUF Creation (Integrated)

GGUF creation is now fully integrated into the upload process. When you select "Create GGUF quantizations?" during upload:

**What happens:**
1. Model is merged to 16-bit in temporary WSL native filesystem (for performance)
2. llama.cpp is cloned and built automatically (first time only)
3. Base f16 GGUF is created
4. Q4_K_M, Q5_K_M, and Q8_0 quantizations are generated
5. All GGUF files are saved to `training-run/model-name/gguf/`
6. Files are uploaded to HuggingFace
7. Temporary files are automatically cleaned up (~14GB saved)

**Performance:**
- All GGUF operations happen in WSL native filesystem (`~/tmp_gguf`)
- No NTFS I/O issues
- Typical timing: ~3-5 minutes total for all quantizations

**Output location:**
```
sft_output_rtx3090/YYYYMMDD_HHMMSS/model-name/gguf/
├── model-name.gguf         (f16, ~14GB)
├── model-name-Q4_K_M.gguf  (~4GB)
├── model-name-Q5_K_M.gguf  (~5GB)
└── model-name-Q8_0.gguf    (~8GB)
```

**Why this is better:**
- No manual GGUF creation steps
- No scattered `gguf_output/` directories
- Automatic cleanup of large temporary files
- Everything organized with the training run
- Single command for full upload + GGUF

**Manual GGUF creation** (if needed):
```bash
# Standalone GGUF creation without upload
python src/upload_to_hf.py \
  ./sft_output_rtx3090/20251122_143000/final_model \
  username/model-name \
  --skip-standard \
  --create-gguf
```

### Evaluation

```bash
# Using Ollama
python -m Evaluator.cli \
  --model claudesidian-mcp \
  --prompt-set Evaluator/prompts/baseline.json \
  --output Evaluator/results/run_$(date +%s).json

# Using LM Studio
python -m Evaluator.cli \
  --backend lmstudio \
  --model your-model-name \
  --prompt-set Evaluator/prompts/tool_prompts.json \
  --output Evaluator/results/run_lmstudio.json \
  --markdown Evaluator/results/report.md
```

**LM Studio + WSL Setup:**

If running from WSL and LM Studio is on Windows:

1. In LM Studio: **Developer** → **Server** → Enable **Serve on Local Network**
2. Note the IP shown (e.g., `192.168.1.104`)
3. Add to `.env`: `LMSTUDIO_HOST=192.168.1.104`
4. Run: `./run.sh eval`

The `.env` file is auto-loaded by `run.sh` and `run.ps1`.

**Prompt sets:**
- `baseline.json` - General scenarios
- `tool_prompts.json` - One prompt per tool (47 prompts)
- `tool_combos.json` - Multi-step workflows

## Architecture Overview

### Multi-Platform Training Strategy

The codebase maintains **three parallel implementations** for different hardware and training methods:

**Mac (mistral_lora_mac):**
- Framework: MLX (Apple-optimized)
- Method: LoRA fine-tuning
- Config: YAML (`config/config.yaml`)
- Entry: `main.py`
- Model: Mistral-7B-Instruct-v0.3

**NVIDIA - SFT (rtx3090_sft):**
- Framework: Unsloth + TRL
- Method: SFT (Supervised Fine-Tuning) - Direct supervision
- Config: Python dataclasses (`configs/training_config.py`)
- Entry: `train_sft.py`
- Models: 3B, 7B, 13B, 20B (configurable)
- Purpose: **Initial training** - Teaches tool-calling from scratch
- Dataset: Positive examples only (no labels)

**NVIDIA - KTO (rtx3090_kto):**
- Framework: Unsloth + TRL
- Method: KTO (Preference learning) - Contrastive learning
- Config: Python dataclasses (`configs/training_config.py`)
- Entry: `train_kto.py`
- Models: 3B, 7B, 13B, 20B (configurable)
- Purpose: **Refinement** - Teaches preference between good/bad tool calls
- Dataset: Interleaved True/False labels required

**Training Pipeline Recommendation:**
1. **Start with SFT** (`rtx3090_sft`) - Teach the model WHAT tool calling is
2. **Refine with KTO** (`rtx3090_kto`) - Teach the model WHICH tool calls are better

All trainers consume datasets from `Datasets/` directory.

### Data Flow Pipeline

1. **Generation**: Teacher models create synthetic examples in ChatML format
2. **Validation**: `tools/validate_syngen.py` checks structure, context objects, tool schemas
3. **Preparation**: Trainer converts to platform-specific format (Mistral Instruct or KTO)
4. **Training**: LoRA or KTO fine-tuning with checkpointing
5. **Evaluation**: Test via `Evaluator/` with prompt sets

## Critical Patterns & Requirements

### Context Object Pattern

**Every tool call MUST include a complete context object as the FIRST parameter:**

```json
{
  "context": {
    "sessionId": "session_1731015400000_a1b2c3d4e",
    "workspaceId": "ws_1731015400000_f5g6h7i8j",
    "sessionDescription": "Brief summary of session",
    "sessionMemory": "Never empty - prior context",
    "toolContext": "Why calling this tool",
    "primaryGoal": "User's main objective",
    "subgoal": "What this call achieves"
  },
  "otherParams": "..."
}
```

**All 7 fields are required.** `sessionMemory` must never be empty.

### KTO Interleaved Dataset Requirement

**CRITICAL for rtx3090_kto trainer:**

TRL's KTOTrainer has a bug where it crashes on homogeneous batches (all True or all False labels). The workaround is to use **interleaved datasets** with strict True/False/True/False pattern.

```python
# Dataset must alternate labels:
[True, False, True, False, True, False, ...]
```

This guarantees mixed batches with sequential sampling and prevents CUDA errors.

**Reference:** `KTO_TRAINING_REFERENCE.md` for full details.

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

- **NO system message** (starts with user role)
- `label`: `true` = desirable example, `false` = undesirable (for contrastive learning)
- Tool calls show complete execution: call → result → response
- Single-turn conversations preferred (multi-turn removed in 11.18.25 update)

### Available Datasets

**SFT Datasets (Positive Examples Only):**
- `syngen_tools_sft_11.18.25.jsonl` - **2,676 examples**
  - Combined from 11.14.25 (2,324 examples) + ChatGPT (352 examples)
  - Only positive (label=true) examples
  - Single-turn conversations only
  - Shuffled for training diversity
  - Default for `rtx3090_sft` trainer

**KTO Datasets (Interleaved True/False):**
- `syngen_tools_11.18.25.jsonl` - **4,649 examples**
  - Cleaned version with multi-turn conversations removed
  - Perfectly interleaved True/False/True/False pattern
  - Single-turn conversations only
  - Default for `rtx3090_kto` trainer

**Legacy Datasets:**
- `syngen_tools_11.14.25.jsonl` - Original 4,652 examples (had 3 multi-turn)
- `syngen_toolset_v1.0.0_claude.jsonl` - Earlier version
- `syngen_toolset_v1.0.0_chatgpt.jsonl` - Earlier version

**Why Multi-Turn Removed (11/18/25):**
- Model was outputting generic conversations instead of tool calls
- Multi-turn examples added complexity without benefit
- Single-turn examples provide clearer tool-calling patterns
- Improved dataset focus on tool behavior learning

### SFT vs KTO: Key Differences

**When to Use Each:**

| Aspect | SFT (rtx3090_sft) | KTO (rtx3090_kto) |
|--------|------------------|-------------------|
| **Purpose** | Teach tool-calling from scratch | Refine existing tool-calling behavior |
| **Learning Type** | Direct supervision | Preference learning |
| **Dataset** | Positive examples only | Positive + negative examples |
| **Labels** | Ignored (or no labels) | Required (True/False) |
| **Interleaving** | Not needed | Mandatory (True/False/True/False) |
| **Learning Rate** | 2e-4 (high) | 2e-7 (very low) |
| **Epochs** | 3 (multiple passes) | 1 (single pass) |
| **Batch Size** | 6 (higher throughput) | 4 (memory for ref_model) |
| **Reference Model** | Not needed | Required (for KL penalty) |
| **Max Grad Norm** | 1.0 (allows larger updates) | 0.5 (tighter control) |
| **VRAM Usage** | ~7-9 GB | ~9-11 GB (ref_model adds ~2GB) |
| **Training Time** | ~45 min (3 epochs) | ~15 min (1 epoch) |
| **Use Case** | Initial training | Fine-tuning / refinement |

**Training Pipeline:**
1. **SFT first** - Model learns WHAT tool calling is and HOW to format tool calls
2. **KTO second** (optional) - Model learns WHICH tool calls are better quality
3. **Result** - Model that both understands tool calling AND prefers high-quality calls

**Why SFT Failed Initially (Resolved 11/18/25):**
- Used KTO for initial training (wrong - KTO assumes model already knows tool syntax)
- KTO learning rate too low (2e-7) for teaching new behavior
- Only 1 epoch insufficient for learning from scratch
- Solution: Use SFT first with higher LR (2e-4) and 3 epochs

## Configuration Entry Points

### RTX 3090 SFT Configuration
**File:** `Trainers/rtx3090_sft/configs/training_config.py`

The SFT config uses Python dataclasses optimized for supervised fine-tuning:

**ModelConfig:**
- `model_name`: Base model to use (default: `unsloth/mistral-7b-v0.3-bnb-4bit`)
- `max_seq_length`: Maximum sequence length (default: 2048)
- `load_in_4bit`: Use 4-bit quantization (default: True)

**LoRAConfig:**
- `r`: LoRA rank (default: 32)
- `lora_alpha`: LoRA alpha scaling (default: 64)
- `lora_dropout`: Dropout for LoRA layers (default: 0.05)
- `target_modules`: Which layers to apply LoRA (q/k/v/o/gate/up/down projections)

**SFTTrainingConfig:**
- `per_device_train_batch_size`: Batch size per GPU (default: 6, higher than KTO's 4)
- `gradient_accumulation_steps`: Accumulation steps (default: 4, effective batch = 24)
- `learning_rate`: Learning rate (default: 2e-4, 100x higher than KTO)
- `max_grad_norm`: Gradient clipping (default: 1.0, vs KTO's 0.5)
- `max_seq_length`: Max sequence length (default: 2048)
- `num_train_epochs`: Number of epochs (default: 3, vs KTO's 1)
- `packing`: Pack multiple examples per sequence (default: False)
- `completion_only_loss`: Train only on assistant completions (default: True)
- `logging_steps`: Log metrics every N steps (default: 5)

**DatasetConfig:**
- `dataset_name`: HuggingFace dataset name
- `local_file`: Path to local JSONL file (default: `../../Datasets/syngen_tools_sft_11.18.25.jsonl`)
- `filter_desirable`: Filter for positive examples (default: False, dataset pre-filtered)

**Preset configs:**
- `get_3b_config()` - Fast iteration (batch_size=12, ~12GB VRAM)
- `get_7b_config()` - Production quality (batch_size=6, ~9GB VRAM) ⭐ **Recommended**
- `get_13b_config()` - Maximum quality (batch_size=4, ~14GB VRAM)
- `get_20b_config()` - Specialized tasks (batch_size=4)

**CLI Overrides:**
```bash
python train_sft.py \
  --model-size 7b \
  --batch-size 6 \
  --gradient-accumulation 4 \
  --learning-rate 2e-4 \
  --num-epochs 3 \
  --max-seq-length 2048
```

### RTX 3090 KTO Configuration
**File:** `Trainers/rtx3090_kto/configs/training_config.py`

The KTO config uses Python dataclasses optimized for preference learning:

**ModelConfig:**
- `model_name`: Base model to use (default: `unsloth/mistral-7b-v0.3-bnb-4bit`)
- `max_seq_length`: Maximum sequence length (default: 2048)
- `load_in_4bit`: Use 4-bit quantization (default: True)

**LoRAConfig:**
- `r`: LoRA rank (default: 32)
- `lora_alpha`: LoRA alpha scaling (default: 64)
- `lora_dropout`: Dropout for LoRA layers (default: 0.05)
- `target_modules`: Which layers to apply LoRA (q/k/v/o/gate/up/down projections)

**KTOTrainingConfig:**
- `per_device_train_batch_size`: Batch size per GPU (default: 4)
- `gradient_accumulation_steps`: Accumulation steps (default: 6, effective batch = 24)
- `learning_rate`: Learning rate (default: 2e-7)
- `beta`: KTO beta parameter (default: 0.3)
- `max_length`: Max sequence length (default: 2048)
- `num_train_epochs`: Number of epochs (default: 1)
- `use_kto_s`: Use KTO-S SIGN correction (default: False)
- `use_two_stage_lr`: Use two-stage LR schedule (default: False)

**DatasetConfig:**
- `dataset_name`: HuggingFace dataset name
- `local_file`: Path to local JSONL file (default: `../../Datasets/syngen_tools_11.14.25.jsonl`)
- `split_dataset`: Create train/val split (default: False)

**To modify:** Edit `configs/training_config.py` directly or override via CLI.

**Preset configs:**
- `get_3b_config()` - Fast iteration (batch_size=8)
- `get_7b_config()` - Production quality (batch_size=4) ⭐ **Recommended**
- `get_13b_config()` - Maximum quality (batch_size=2)
- `get_20b_config()` - Specialized tasks (batch_size=4)

**CLI Overrides:**
```bash
python train_kto.py \
  --model-size 7b \
  --batch-size 4 \
  --gradient-accumulation 6 \
  --learning-rate 2e-7 \
  --num-epochs 1 \
  --max-seq-length 2048
```

### Mac Configuration
**File:** `Trainers/mistral_lora_mac/config/config.yaml`

Key parameters:
- `model.max_seq_length: 2048`
- `lora.rank: 16` (memory vs capacity tradeoff)
- `training.per_device_batch_size: 2`
- `training.gradient_accumulation_steps: 4`
- `data.dataset_path: "path/to/dataset.jsonl"`

## Tool Coverage

The system supports **47+ tools** across **5 agent categories:**
- **vaultManager** - File/folder operations
- **contentManager** - CRUD operations
- **memoryManager** - Session/state/workspace management
- **vaultLibrarian** - Advanced search, batch operations
- **agentManager** - Agent lifecycle, prompt execution

**Schema source:** `Tools/tool_schemas.json` (central schema definitions)

## Platform-Specific Notes

### Windows vs WSL2

**WSL2 (Strongly Recommended):**
- Full compatibility with all features
- Multiprocessing works correctly
- Better performance
- Native bash script support
- All setup/upload scripts available

**Windows PowerShell (Limited Support):**
- Known issues with multiprocessing (hangs on data loading)
- `dataloader_num_workers` must be 0 in config
- Requires Windows compatibility patches (auto-applied in `train_kto.py`)
- Flash Attention not supported
- Limited to PowerShell scripts (`.ps1`)

**Windows Compatibility Patches** (auto-applied in `train_kto.py:26-50` and `upload_to_hf.py:10-38`):
1. Dataclass `fields()` wrapper for non-dataclasses
2. Disable `torch.compile` (not supported on Windows)
3. Pre-patch `torch._inductor.runtime.hints`

**If using Windows PowerShell:**
- Set `dataloader_num_workers: 0` in `configs/training_config.py`
- Use `.ps1` scripts instead of `.sh` scripts
- Expect slower training (no multiprocessing)

**Switching to WSL2:**
```powershell
# Install WSL2 (Windows 10/11)
wsl --install

# Clone repo in WSL2 filesystem (better performance)
cd ~
git clone <repo-url>

# Follow Linux setup instructions
```

### Mac Training (mistral_lora_mac)
Verify Metal GPU is available:
```bash
python -c "import mlx.core as mx; print(mx.metal.is_available())"
```

### Memory Optimization
Both trainers use:
- 4-bit quantization (RTX) or float16 (Mac)
- 8-bit optimizers (RTX)
- Gradient checkpointing (optional, 13B+ models)
- LoRA for parameter efficiency

Expected VRAM (RTX 3090):
- 3B: ~8-10 GB
- 7B: ~9-11 GB
- 13B: ~14-16 GB

## Testing & Validation

### Unit Tests
```bash
# Test dataset validator
python tools/validate_syngen.py <dataset.jsonl>
```

### Integration Tests
```bash
# Dry run (setup without training)
cd Trainers/rtx3090_kto
python train_kto.py --model-size 7b --dry-run
```

### Model Evaluation
```bash
# Serve model and run evaluation suite
python -m Evaluator.cli --model <name> --prompt-set Evaluator/prompts/tool_prompts.json
```

## Troubleshooting

### CUDA Out of Memory (RTX)

**For SFT:**
```bash
# Reduce batch size
python train_sft.py --model-size 7b --batch-size 4 --gradient-accumulation 6

# Reduce sequence length
python train_sft.py --model-size 7b --max-seq-length 1024
```

**For KTO:**
```bash
# Reduce batch size
python train_kto.py --model-size 7b --batch-size 2 --gradient-accumulation 16

# Reduce sequence length
python train_kto.py --model-size 7b --max-seq-length 1024
```

### Mac Out of Memory
Edit `config/config.yaml`:
- Reduce `per_device_batch_size: 1`
- Reduce `max_seq_length: 1024`
- Reduce `lora.rank: 8`

### Dataset Validation Failures
Common issues:
- Missing context object (must be first parameter)
- Empty `sessionMemory` field (never allowed)
- Incorrect ID format (must match: `session_<13digits>_<9chars>`)
- Context not as first parameter in tool call

### KTO Training Crashes (TRL Bug)
If you see `CUDA error: invalid configuration argument`:
- Ensure dataset is **interleaved** (True/False/True/False pattern)
- Check `KTO_TRAINING_REFERENCE.md` for full workaround

### GGUF Quantization Hangs (WSL2)
If GGUF quantization hangs or file size stops growing:
- **Cause:** Windows NTFS drives (`/mnt/c`) have I/O buffering issues
- **Solution:** Always use WSL native filesystem (`~` or `/home`)
- **Steps:**
  1. Copy base GGUF to `~/tmp_gguf/`
  2. Run quantization in WSL native filesystem
  3. Copy results back to Windows if needed
- **Performance:** 10-100x faster on WSL native filesystem
- See "GGUF Creation (WSL2 / Linux)" section for full workflow

### Training Logs Not Being Created (Fixed 11/18/25)
If training logs aren't appearing in the logs directory:
- **Issue:** Recursive `logs/logs/logs` directories or missing log files
- **Cause:** Callback was receiving wrong directory path
- **Fixed in:** Both `train_sft.py` and `train_kto.py`
- **Solution:** Callbacks now receive `run_dir` instead of `logs_dir`
- **Verify:** Check `sft_output_rtx3090/YYYYMMDD_HHMMSS/logs/training_*.jsonl` exists
- **Monitor:** Use `tail -f logs/training_latest.jsonl` for real-time viewing

### Model Outputs Generic Text Instead of Tool Calls
If model generates conversation text rather than tool calls:
- **Diagnosis:** Wrong training method for initial training
- **Cause:** Using KTO (preference learning) before teaching tool syntax
- **Solution:** Use SFT first, then optionally refine with KTO
- **Steps:**
  1. Train with `rtx3090_sft` using positive examples (teaches tool syntax)
  2. Optionally refine with `rtx3090_kto` using contrastive examples
- **Dataset:** Use `syngen_tools_sft_11.18.25.jsonl` for SFT (2,676 examples)
- **Settings:** 2e-4 LR, 3 epochs, no system prompt

## Key Documentation

- `README.md` - Project overview and stats
- `KTO_TRAINING_REFERENCE.md` - **Critical:** TRL bug workaround for KTO training
- `finetuning-strategy.md` - Master strategy document (203KB, comprehensive)
- `Trainers/rtx3090_sft/README.md` - **SFT training guide (start here for initial training)**
- `Trainers/rtx3090_kto/README.md` - KTO training guide (for refinement)
- `Trainers/mistral_lora_mac/README.md` - Mac training guide
- `Evaluator/README.md` - Evaluation harness usage
- `docs/SCHEMA_VERIFICATION_REFERENCE.md` - Tool schema reference

## Development Workflow

1. **Validate dataset** before training (`python tools/validate_syngen.py <dataset>`)
2. **Choose training method:**
   - **Initial training** → Use SFT (`rtx3090_sft`) with positive examples
   - **Refinement** → Use KTO (`rtx3090_kto`) with contrastive examples
3. **Choose platform** (Mac vs RTX) based on available hardware
4. **Use appropriate config** (YAML for Mac, dataclass presets for RTX)
5. **Monitor logs** in real-time:
   - Console: Formatted table every 5 steps
   - File: `tail -f logs/training_latest.jsonl`
6. **Evaluate checkpoints** with Evaluator before final deployment
7. **For SFT:** Use positive examples only, no interleaving needed
8. **For KTO:** Always verify dataset is interleaved (True/False/True/False)

## Environment Variables

Create a `.env` file in the repository root:

```bash
# HuggingFace (required for uploads, optional for training)
HF_TOKEN=hf_your_token_here
# or
HF_API_KEY=hf_your_token_here

# Weights & Biases (optional)
WANDB_API_KEY=your_wandb_key

# Conda environment name (optional, for PowerShell scripts)
CONDA_ENV=unsloth_env

# Mac training (optional)
HF_HOME=/path/to/cache
```

**Getting tokens:**
- HuggingFace: https://huggingface.co/settings/tokens (create with "write" access)
- W&B: https://wandb.ai/authorize

**Note:**
- `.env` file is gitignored
- See the root `.env.example` for template values
- PowerShell scripts auto-load from root `.env` file

## Script Reference

### Unified CLI (Recommended)

**Main entry point:**
- `tuner.py` - Unified CLI for all operations (train, upload, eval, pipeline)
- `run.sh` - Bash wrapper (auto-activates conda)
- `run.ps1` - PowerShell wrapper (auto-finds conda python)

### Setup Scripts

- `Trainers/rtx3090_kto/setup.sh` - Full NVIDIA environment setup with verification
- `Trainers/rtx3090_sft/setup.sh` - Full NVIDIA environment setup with verification

### Python Scripts (Cross-platform)

**Training:**
- `Trainers/rtx3090_sft/train_sft.py` - SFT training entry point
- `Trainers/rtx3090_kto/train_kto.py` - KTO training entry point

**Data:**
- `Trainers/rtx3090_sft/src/data_loader.py` - SFT dataset loading (simplified, no interleaving)
- `Trainers/rtx3090_kto/src/data_loader.py` - KTO dataset loading (with interleaving)
- `Trainers/rtx3090_sft/src/model_loader.py` - Model loading with Unsloth (shared)
- `Trainers/rtx3090_kto/src/model_loader.py` - Model loading with Unsloth (shared)
- `Trainers/rtx3090_kto/src/kto_s_trainer.py` - Custom KTO trainer with SIGN correction

**Shared Upload Framework:**
- `Trainers/shared/upload/` - Shared upload module (SOLID architecture)
  - `core/` - Interfaces, types, config, exceptions
  - `strategies/` - Save strategies (LoRA, 16-bit, 4-bit)
  - `converters/` - GGUF converter
  - `uploaders/` - HuggingFace uploader
  - `documentation/` - Manifest, model card, README generators
  - `platform/` - GPU memory, Windows patches, filesystem utilities
  - `orchestrator.py` - Upload workflow orchestration
  - `cli/upload_cli.py` - CLI interface

**Upload Thin Wrappers:**
- `Trainers/rtx3090_sft/src/upload_to_hf.py` - SFT upload (delegates to shared)
- `Trainers/rtx3090_kto/src/upload_to_hf.py` - KTO upload (delegates to shared)

**Other Utilities:**
- `Trainers/rtx3090_kto/src/inference.py` - Inference utilities
- `Trainers/rtx3090_sft/src/training_callbacks.py` - Training callbacks
- `Trainers/rtx3090_kto/src/training_callbacks.py` - Training callbacks
- `Trainers/rtx3090_kto/src/adaptive_memory.py` - Adaptive memory management
- `Trainers/rtx3090_kto/check_config.py` - Config verification
- `Trainers/rtx3090_kto/check_gpu_setup.py` - GPU verification
- `Trainers/rtx3090_kto/diagnose_batch_size.py` - Batch size tuning
- `Trainers/rtx3090_kto/test_installation.py` - Installation verification

**Validation:**
- `tools/validate_syngen.py` - Dataset validator
- `tools/analyze_tool_coverage.py` - Coverage analysis

## Training Output Structure

Both SFT and KTO trainers create organized output directories with consistent structure:

**During training:**
```
sft_output_rtx3090/YYYYMMDD_HHMMSS/  (or kto_output_rtx3090/YYYYMMDD_HHMMSS/)
├── checkpoints/
│   ├── checkpoint-50/
│   ├── checkpoint-100/
│   ├── checkpoint-150/
│   └── ...
├── final_model/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── tokenizer.json
│   └── ...
└── logs/
    ├── training_YYYYMMDD_HHMMSS.jsonl  (detailed metrics)
    └── training_latest.jsonl           (symlink to current)
```

**After upload:**
```
sft_output_rtx3090/YYYYMMDD_HHMMSS/
├── checkpoints/
├── final_model/
├── logs/
└── your-model-name/          # Created by upload_to_hf.py
    ├── lora/                 # LoRA adapters
    ├── merged-16bit/         # Full merged model (if created)
    ├── merged-4bit/          # 4-bit quantized (if created)
    ├── gguf/                 # GGUF quantizations (if created)
    │   ├── your-model-name.gguf
    │   ├── your-model-name-Q4_K_M.gguf
    │   ├── your-model-name-Q5_K_M.gguf
    │   └── your-model-name-Q8_0.gguf
    ├── upload_manifest.json  # Metadata about upload
    └── README.md             # Auto-generated documentation
```

**Log Structure:**
- Metrics logged every 5 steps to JSONL file
- Console output shows formatted table every 5 steps
- Each JSONL entry contains: step, timestamp, loss, learning_rate, GPU memory, etc.
- Real-time monitoring: `tail -f logs/training_latest.jsonl`

**Logging System (Fixed in 11/18/25):**
- Both trainers use shared `MetricsTableCallback` from `src/training_callbacks.py`
- Callback receives `run_dir`, automatically creates `logs/` subdirectory
- Logs written every `logging_steps` (default: 5)
- Fixed issues:
  - Recursive `logs/logs/logs` directories (now correctly creates single `logs/`)
  - Log files not being created (now properly writes to `logs/training_*.jsonl`)
  - Logging every step instead of every N steps (now respects `logging_steps` parameter)

**Upload Manifest Structure:**
```json
{
  "upload_timestamp": "2025-11-22T14:30:00",
  "training_run": "20251122_143000",
  "model_name": "username/model-name",
  "huggingface_url": "https://huggingface.co/username/model-name",
  "formats_created": ["merged_16bit", "gguf"],
  "directory_structure": {
    "merged_16bit": "merged-16bit/",
    "gguf": "gguf/"
  },
  "gguf_quantizations": [
    "model-name.gguf",
    "model-name-Q4_K_M.gguf",
    "model-name-Q5_K_M.gguf",
    "model-name-Q8_0.gguf"
  ]
}
```

**Why This Structure:**
- Timestamped runs allow multiple training sessions without conflicts
- Checkpoints separate from final model for easy comparison
- Logs in dedicated directory for analysis and monitoring
- **Upload artifacts stay with their training run** (new!)
- Complete traceability via upload_manifest.json
- No scattered files in separate directories
- Symlink to latest log for convenient tailing

## Important Notes

- **Never commit** trained models or checkpoints (large files in `.gitignore`)
- **Always run** `validate_syngen.py` before training
- **Training order:** Start with SFT, then optionally refine with KTO
- **For KTO training:** Dataset interleaving is mandatory (not optional)
- **For SFT training:** Only positive examples needed (no interleaving)
- **Context objects:** All 7 fields required, sessionMemory never empty
- **Tool schemas:** Reference `Tools/tool_schemas.json` for validation
- **Windows users:** Strongly recommend WSL2 for rtx3090_sft/kto (better compatibility)
- **PowerShell scripts:** All auto-load HF_TOKEN from root `.env` file
- **Bash scripts:** Source conda and activate environment automatically
- **Logging frequency:** Both trainers log metrics every 5 steps (configurable)
