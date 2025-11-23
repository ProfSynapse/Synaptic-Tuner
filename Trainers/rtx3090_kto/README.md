# KTO Fine-Tuning for NVIDIA RTX 3090 (24GB VRAM)

Production-ready implementation of KTO (Kahneman-Tversky Optimization) fine-tuning optimized for NVIDIA RTX 3090 GPUs with 24GB VRAM. Based on the comprehensive specification in `docs/prep/local-training/rtx3090-kto-finetuning.md`.

## Features

- **Optimized for RTX 3090 (24GB VRAM)**: Supports 3B to 13B parameter models
- **Unsloth Integration**: 2x faster training with 70% less VRAM usage
- **Multiple Model Tiers**: Pre-configured for 3B, 7B, 13B, and 20B models
- **Flexible Configuration**: Easy-to-use config system with CLI overrides
- **Production Ready**: Includes inference, upload utilities, and monitoring
- **Memory Efficient**: 4-bit quantization, 8-bit optimizers, gradient checkpointing

## Quick Start

### Interactive CLI (Recommended)

**KTO is designed for refinement**, not initial training. Use the interactive CLI for best results:

**Linux/WSL2:**
```bash
# Go to Trainers/ directory
cd ../
./train.sh

# Choose option:
#   2) KTO Only - If you already have an SFT model
#   3) SFT → KTO Pipeline - Complete training (recommended)
```

**Windows PowerShell:**
```powershell
# Go to Trainers\ directory
cd ..\
.\train.ps1

# Choose option:
#   2) KTO Only - If you already have an SFT model
#   3) SFT → KTO Pipeline - Complete training (recommended)
```

**Why SFT → KTO Pipeline?**
- 📚 SFT teaches tool-calling syntax (WHAT to do)
- ✨ KTO refines quality (WHICH calls are better)
- 🎯 Combined approach produces best results
- ⚙️ Interactive CLI configures everything automatically

### 1. Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python -c "from unsloth import FastLanguageModel; print('Unsloth: OK')"
```

### 2. Basic Training (7B Model - Recommended)

```bash
# Easiest method (wrapper script):
./train.sh --model-size 7b

# Or with Python directly:
venv/bin/python train_kto.py --model-size 7b

# With adaptive memory management (auto-adjusts batch size):
./train.sh --model-size 7b --adaptive-memory

# With custom dataset:
./train.sh --model-size 7b --local-file ./my_data.jsonl

# With W&B logging:
./train.sh --model-size 7b --wandb --wandb-project my-project
```

**Note**: The `train.sh` wrapper ensures the correct Python environment is used. You can also use `venv/bin/python` directly.

### 3. Advanced Training Options

```bash
# 3B model for fast iteration
python train_kto.py --model-size 3b --batch-size 8 --gradient-accumulation 4

# 13B model for maximum quality
python train_kto.py --model-size 13b --batch-size 2 --gradient-accumulation 16

# Custom model
python train_kto.py \
  --model-name unsloth/Qwen2.5-7B-Instruct-bnb-4bit \
  --batch-size 4 \
  --learning-rate 5e-7 \
  --num-epochs 2

# Dry run (setup without training)
python train_kto.py --model-size 7b --dry-run
```

## Dataset Format

KTO requires unpaired preference data with boolean labels:

```jsonl
{"prompt": "What is the capital of France?", "completion": "The capital of France is Paris.", "label": true}
{"prompt": "What is the capital of France?", "completion": "I don't know.", "label": false}
```

Or ChatML format (automatically converted):

```jsonl
{
  "conversations": [
    {"role": "user", "content": "What is AI?"},
    {"role": "assistant", "content": "AI is artificial intelligence..."}
  ],
  "label": true
}
```

## Model Configurations

### Tier 1: Fast Iteration (3B Models)

```bash
python train_kto.py --model-size 3b
```

- Models: Qwen2.5-3B, Llama-3.2-3B
- Batch size: 8 (effective: 32)
- Speed: 25-35 tokens/sec
- Use case: Rapid prototyping and testing

### Tier 2: Production Quality (7B Models) ⭐ **Recommended**

```bash
python train_kto.py --model-size 7b
```

- Models: Mistral-7B, Llama-3.1-8B, Qwen2.5-7B
- Batch size: 4 (effective: 32)
- Speed: 20-30 tokens/sec
- Use case: Production deployments

### Tier 3: Advanced (13B Models)

```bash
python train_kto.py --model-size 13b
```

- Models: Llama-2-13B
- Batch size: 2 (effective: 32)
- Speed: 15-25 tokens/sec
- Use case: Maximum quality requirements

### Tier 4: Large Models (20B)

```bash
python train_kto.py --model-size 20b
```

- Models: GPT-OSS-20B
- Batch size: 4 (effective: 32)
- Speed: 10-15 tokens/sec
- Use case: Specialized large-scale tasks

## Memory Usage

### Expected VRAM Usage (RTX 3090 24GB)

| Model Size | VRAM Usage | Batch Size | Headroom |
|------------|------------|------------|----------|
| 3B | ~8-10 GB | 8 | 14+ GB |
| 7B | ~9-11 GB | 4 | 13+ GB |
| 13B | ~14-16 GB | 2 | 8+ GB |
| 20B | ~18-20 GB | 4 | 4+ GB |

### Memory Optimization Techniques

The implementation uses:
- 4-bit NF4 quantization (75% memory reduction)
- 8-bit AdamW optimizer (saves ~2GB)
- Gradient checkpointing (optional, 40-50% activation memory reduction)
- Unsloth optimizations (70% VRAM reduction vs standard)

## Training Performance

### Expected Training Times (RTX 3090, 10k examples)

| Model | Tokens/Sec | Time per Epoch |
|-------|------------|----------------|
| 3B | 30-40 | ~1.5 hours |
| 7B | 20-30 | ~2.5 hours |
| 13B | 15-25 | ~3.5 hours |

### Throughput with Unsloth

Unsloth provides **1.5-2x speedup** over standard PyTorch implementations.

## Inference

### Interactive Chat

```bash
python src/inference.py ./kto_output_rtx3090/final_model
```

### Programmatic Inference

```python
from src.inference import KTOInference

# Load model
inference = KTOInference("./kto_output_rtx3090/final_model")

# Generate response
response = inference.generate(
    "What is machine learning?",
    temperature=0.7,
    max_new_tokens=512
)
print(response)

# Multi-turn chat
messages = [
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi! How can I help?"},
    {"role": "user", "content": "Explain KTO training"}
]
response = inference.chat(messages)
```

## Upload to HuggingFace

### Standard Upload

```bash
python src/upload_to_hf.py \
  ./kto_output_rtx3090/final_model \
  username/model-name \
  --token YOUR_HF_TOKEN \
  --save-method merged_16bit
```

### With GGUF Conversion

```bash
python src/upload_to_hf.py \
  ./kto_output_rtx3090/final_model \
  username/model-name \
  --token YOUR_HF_TOKEN \
  --create-gguf \
  --gguf-quantizations Q4_K_M Q5_K_M Q8_0
```

This will:
1. Upload the standard model (16-bit merged)
2. Create GGUF versions (Q4_K_M, Q5_K_M, Q8_0)
3. Upload all GGUF files to the same repo

## Configuration

### Using Config Presets

```python
from configs.training_config import get_7b_config

config = get_7b_config()
config.training.learning_rate = 1e-6
config.training.num_train_epochs = 2
```

### Custom Configuration

Edit `configs/training_config.py` or create your own:

```python
from configs.training_config import Config

config = Config()
config.model.model_name = "unsloth/custom-model"
config.lora.r = 32
config.training.per_device_train_batch_size = 2
```

## Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size
python train_kto.py --model-size 7b --batch-size 2 --gradient-accumulation 16

# Reduce sequence length
python train_kto.py --model-size 7b --max-seq-length 1024

# Enable gradient checkpointing (for 13B+)
# Edit configs/training_config.py: gradient_checkpointing = True
```

### Slow Training

```bash
# Ensure Unsloth is installed
pip install unsloth

# Check GPU utilization
nvidia-smi -l 1

# Reduce dataloader workers if on Windows
# Edit configs/training_config.py: dataloader_num_workers = 0
```

### Installation Issues (Windows)

1. Install Visual Studio C++ Build Tools
2. Install CUDA Toolkit matching PyTorch version
3. Use WSL2 for best compatibility

```bash
# Or use pre-built wheels
pip install bitsandbytes --prefer-binary
```

### NaN Loss

```bash
# Reduce learning rate
python train_kto.py --learning-rate 1e-7

# Increase warmup
# Edit configs/training_config.py: warmup_ratio = 0.2
```

## Project Structure

```
rtx3090_kto/
├── README.md
├── requirements.txt
├── train_kto.py              # Main training script
├── configs/
│   └── training_config.py    # Configuration presets
└── src/
    ├── data_loader.py        # Dataset loading and preprocessing
    ├── model_loader.py       # Model loading with Unsloth
    ├── inference.py          # Inference utilities
    └── upload_to_hf.py       # HuggingFace upload
```

## Command-Line Reference

### Training Arguments

```
--model-size {3b,7b,13b,20b}     Model size preset
--model-name MODEL               Override model name
--max-seq-length LENGTH          Override max sequence length
--dataset-name NAME              HuggingFace dataset name
--dataset-file FILE              Dataset file within HF dataset
--local-file PATH                Path to local JSONL file
--split-dataset                  Create train/validation split
--output-dir DIR                 Override output directory
--batch-size SIZE                Override batch size
--gradient-accumulation STEPS    Override gradient accumulation
--learning-rate LR               Override learning rate
--num-epochs N                   Override number of epochs
--max-steps N                    Override max training steps
--wandb                          Enable W&B logging
--wandb-project PROJECT          W&B project name
--wandb-run-name NAME            W&B run name
--hf-token TOKEN                 HuggingFace token
--dry-run                        Setup without training
```

## Performance Tips

1. **Use Unsloth**: 2x faster, 70% less VRAM
2. **Enable BF16**: RTX 3090 supports BFloat16 (enabled by default)
3. **Optimal Batch Size**: Use recommended presets (3B:8, 7B:4, 13B:2)
4. **Pin Memory**: Enabled by default for faster GPU transfers
5. **Flash Attention**: Install for 3-5x faster attention (optional)

```bash
pip install flash-attn --no-build-isolation
```

## Hardware Requirements

- **GPU**: NVIDIA RTX 3090 (24GB VRAM) or equivalent
- **CUDA**: 11.8, 12.1, or 12.4+
- **RAM**: 32GB+ recommended
- **Storage**: 50GB+ for models and checkpoints
- **OS**: Linux (recommended), Windows 10/11, or WSL2

## Citation

Based on:
- **KTO Paper**: [Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306)
- **QLoRA Paper**: [Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- **Unsloth**: [github.com/unslothai/unsloth](https://github.com/unslothai/unsloth)

## License

This implementation is provided as-is for research and educational purposes.

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review `docs/prep/local-training/rtx3090-kto-finetuning.md`
3. Consult [Unsloth docs](https://docs.unsloth.ai/)
4. Review [TRL KTO documentation](https://huggingface.co/docs/trl/main/en/kto_trainer)

---

**Last Updated**: January 2025
**Tested Hardware**: NVIDIA RTX 3090 24GB (Driver 535+, CUDA 12.1)
**Software Versions**: PyTorch 2.2.0, Transformers 4.40.0, TRL 0.8.0, Unsloth latest
