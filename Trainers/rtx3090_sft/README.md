# RTX 3090 SFT Trainer

Supervised Fine-Tuning (SFT) for tool-calling instruction learning using Unsloth + TRL on RTX 3090 (24GB VRAM).

## Quick Start

### Interactive CLI (Recommended)

**Linux/WSL2:**
```bash
# Setup environment (first time only)
bash setup.sh

# Run interactive training CLI
cd ../
./train.sh

# Choose option 1 (SFT Only) or 3 (SFT → KTO Pipeline)
```

**Windows PowerShell:**
```powershell
# Run interactive training CLI
cd ..\
.\train.ps1

# Choose option 1 (SFT Only) or 3 (SFT → KTO Pipeline)
```

### Direct Commands (Advanced)

```bash
# Train with default 7B model
./train.sh --model-size 7b

# Train with custom dataset
./train.sh --model-size 7b --local-file ../../Datasets/my_data.jsonl

# Dry run (setup without training)
python train_sft.py --model-size 7b --dry-run
```

## What is SFT?

**Supervised Fine-Tuning (SFT)** teaches the model new behaviors through direct examples. Unlike KTO (preference learning), SFT learns from positive examples only.

**For tool-calling:**
- Model learns: `tool_call: toolName` syntax
- Internalizes: Context object patterns
- Understands: Tool execution flow

## SFT vs KTO

| Feature | SFT (this trainer) | KTO (rtx3090_kto) |
|---------|-------------------|-------------------|
| **Method** | Direct supervision | Preference learning |
| **Dataset** | Positive examples only | Positive + negative pairs |
| **Reference model** | Not needed | Required (or implicit) |
| **Learning rate** | 2e-4 (higher) | 2e-7 (100x lower) |
| **Epochs** | 3 (default) | 1 (default) |
| **VRAM** | ~7-9 GB (7B model) | ~9-11 GB (7B model) |
| **Training speed** | ~15 min/epoch | ~25 min/epoch |
| **Use case** | Learn new behaviors | Refine existing knowledge |

**Recommendation:** Use SFT first to teach tool-calling, then optionally use KTO for refinement.

## Chaining SFT → KTO

### Automated Pipeline (Recommended)

**Linux/WSL2:**
```bash
# From Trainers/ directory
cd ..
./train_sft_to_kto_pipeline.sh --wandb --wandb-project my-project
```

**Windows PowerShell:**
```powershell
# From Trainers\ directory
cd ..
.\train_sft_to_kto_pipeline.ps1
```

**What it does:**
1. Finds Python from unsloth_env automatically
2. Runs SFT training with `configs/config.yaml`
3. Captures SFT output path
4. Updates KTO config to use SFT model as base
5. Runs KTO refinement automatically
6. Produces final refined model

**Configuration:**
- Edit `configs/config.yaml` to customize SFT settings
- Edit `../rtx3090_kto/configs/config.yaml` for KTO settings
- Pipeline uses YAML configs (no CLI overrides needed)

### Manual Chaining

If you prefer step-by-step control:

**Linux/WSL2:**
```bash
# Step 1: Train SFT
./train.sh --model-size 7b

# Note the output path (e.g., sft_output_rtx3090/20251123_143000/)

# Step 2: Train KTO with SFT output
cd ../rtx3090_kto
python train_kto.py --model-size 7b \
  --model-name ../rtx3090_sft/sft_output_rtx3090/20251123_143000/final_model
```

**Windows PowerShell:**
```powershell
# Step 1: Run SFT
cd Trainers\rtx3090_sft
.\train.ps1

# Note the output path from console

# Step 2: Run KTO with SFT output
cd ..\rtx3090_kto
python train_kto.py --model-size 7b `
  --model-name "..\rtx3090_sft\sft_output_rtx3090\20251123_143000\final_model"
```

## Model Sizes

| Size | Model | VRAM | Batch Size | Speed |
|------|-------|------|------------|-------|
| 3B | Qwen2.5-3B-Instruct | ~6-8 GB | 12 | Fast |
| **7B** | **Mistral-7B-v0.3** | **~7-9 GB** | **6** | **Recommended** |
| 13B | Llama-2-13B | ~12-14 GB | 4 | Quality |
| 20B | GPT-OSS-20B | ~16-18 GB | 4 | Specialized |

## Training Commands

### Basic Training

```bash
# Recommended: 7B model
./train.sh --model-size 7b

# Fast iteration: 3B model
./train.sh --model-size 3b

# Maximum quality: 13B model
./train.sh --model-size 13b
```

### With Options

```bash
# Custom learning rate and epochs
./train.sh --model-size 7b --learning-rate 1e-4 --num-epochs 5

# Custom batch size
./train.sh --model-size 7b --batch-size 8 --gradient-accumulation 3

# With W&B logging
./train.sh --model-size 7b --wandb --wandb-project my-project

# Short test run
python train_sft.py --model-size 7b --max-steps 10
```

### Dataset Options

```bash
# Use local file
./train.sh --model-size 7b --local-file ../../Datasets/syngen_tools_sft_11.18.25.jsonl

# Use HuggingFace dataset
./train.sh --model-size 7b --dataset-name professorsynapse/claudesidian-synthetic-dataset --dataset-file syngen_tools_sft_11.18.25.jsonl

# Create train/val split
./train.sh --model-size 7b --split-dataset
```

## Configuration

**File:** `configs/training_config.py`

### Default Settings (7B Model)

```python
per_device_train_batch_size = 6
gradient_accumulation_steps = 4  # Effective batch = 24
learning_rate = 2e-4
num_train_epochs = 3
max_seq_length = 2048
```

### Modify via CLI

All config params can be overridden via command line:

```bash
python train_sft.py \
  --model-size 7b \
  --batch-size 4 \
  --gradient-accumulation 6 \
  --learning-rate 2e-4 \
  --num-epochs 3 \
  --max-seq-length 2048
```

## Dataset Format

SFT uses ChatML conversational format natively:

```jsonl
{
  "conversations": [
    {"role": "user", "content": "Delete the file test.md"},
    {"role": "assistant", "content": "tool_call: vaultManager_deleteNote\narguments: {\"path\": \"test.md\"}\n\nResult: {\"success\": true}\n\nDeleted test.md successfully."}
  ]
}
```

**Key points:**
- Starts with user role (no system message)
- Tool calls embedded in assistant response
- SFTTrainer applies chat template automatically
- Label field ignored (SFT uses all examples equally)

## Output Structure

```
sft_output_rtx3090/
└── 20251118_183000/          # Timestamped run
    ├── final_model/           # Trained model
    ├── checkpoints/           # Periodic checkpoints
    │   ├── checkpoint-50/
    │   ├── checkpoint-100/
    │   └── checkpoint-150/
    └── logs/
        └── logs/
            └── training_20251118_183000.jsonl  # Metrics
```

## Upload to HuggingFace

```bash
# Upload merged 16-bit model (recommended)
./upload_model.sh username/model-name

# Upload with GGUF creation
./upload_model.sh username/model-name yes

# Direct Python
python src/upload_to_hf.py ./sft_output_rtx3090/20251118_183000/final_model \
  username/model-name \
  --save-method merged_16bit
```

## Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size
./train.sh --model-size 7b --batch-size 4 --gradient-accumulation 6

# Reduce sequence length
./train.sh --model-size 7b --max-seq-length 1024

# Use smaller model
./train.sh --model-size 3b
```

### WSL2 / Windows Issues

- **Recommended:** Use WSL2 (better compatibility)
- **Windows:** Multiprocessing disabled automatically
- **Dataloader workers:** Set to 0 (already configured)

### Training Loss Not Decreasing

- Increase learning rate: `--learning-rate 5e-4`
- Train for more epochs: `--num-epochs 5`
- Check dataset quality with dry run

## Environment Setup

```bash
# Full setup with verification
bash setup.sh

# Quick setup (skip tests)
bash setup.sh --quick

# With Flash Attention (optional, 5-10 min compile)
bash setup.sh --with-flash-attn
```

## Requirements

- RTX 3090 (24GB VRAM) or similar GPU
- CUDA 12.1+
- Python 3.10+
- 50GB+ free disk space

## Key Files

- `train_sft.py` - Main training script
- `configs/training_config.py` - Configuration presets
- `src/data_loader.py` - Dataset loading (simplified vs KTO)
- `src/model_loader.py` - Unsloth model loading
- `src/training_callbacks.py` - Metrics logging

## Performance

**7B Model (RTX 3090):**
- VRAM: ~7-9 GB
- Speed: ~15 min/epoch
- Dataset: 2,676 examples
- Steps/epoch: ~112 steps
- Total time (3 epochs): ~45 minutes

## Differences from KTO Trainer

**Removed:**
- Reference model creation (saves ~2GB VRAM)
- Dataset interleaving requirements
- KTO-specific parameters (beta, weights)
- Two-stage LR scheduling
- KTO-S SIGN correction

**Simplified:**
- Data preprocessing (no conversion needed)
- Configuration (fewer parameters)
- Training loop (standard supervised learning)

**Changed:**
- Higher learning rate (2e-4 vs 2e-7)
- More epochs (3 vs 1)
- Larger batch size possible (6 vs 4)

## Next Steps

1. **Train:** `./train.sh --model-size 7b`
2. **Evaluate:** Test with `Evaluator/` module
3. **Upload:** Share to HuggingFace
4. **Optional:** Refine with KTO (rtx3090_kto)

## Support

See `../rtx3090_kto/README.md` for detailed troubleshooting and advanced configuration options.
