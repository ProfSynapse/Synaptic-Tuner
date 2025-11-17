# Quick Start Guide

Get up and running with KTO training on RTX 3090 in 5 minutes.

## Prerequisites

- NVIDIA RTX 3090 (24GB VRAM) or compatible GPU
- NVIDIA drivers installed (535+)
- CUDA 12.1+ installed
- Python 3.10+
- 50GB+ free disk space

## 1. Installation (5 minutes)

```bash
# Clone or navigate to the project
cd rtx3090_kto

# Run setup script
bash setup.sh

# Activate environment
source venv/bin/activate
```

**Manual installation** (if setup.sh fails):

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Verify Installation (30 seconds)

```bash
# Check CUDA
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
# Expected: CUDA: True

# Check Unsloth
python -c "from unsloth import FastLanguageModel; print('Unsloth: OK')"
# Expected: Unsloth: OK

# Check GPU
nvidia-smi
# Should show RTX 3090 with ~24GB memory
```

## 3. Test with Sample Data (2 minutes)

```bash
# Dry run to test setup (no training)
./train.sh --model-size 3b --dry-run

# Or use Python directly:
# venv/bin/python train_kto.py --model-size 3b --dry-run

# Expected: ✓ Dry run completed. Exiting without training.
```

## 4. First Real Training (5-10 minutes)

Train a small 3B model on the default dataset:

```bash
# Easiest method (uses wrapper script):
./train.sh --model-size 3b --num-epochs 1

# Or use Python directly:
# venv/bin/python train_kto.py --model-size 3b --num-epochs 1
```

This will:
- Load Qwen2.5-3B-Instruct (4-bit)
- Train for 1 epoch on 8 examples
- Save to `./test_output/`
- Take ~5 minutes

## 5. Production Training

### Option A: Train on Default Dataset (Recommended)

```bash
# 7B model on Claudesidian dataset (4,652 examples)
# Easiest method (uses wrapper script):
./train.sh --model-size 7b

# Or use Python directly:
# venv/bin/python train_kto.py --model-size 7b

# Expected time: 2-3 hours for 1 epoch
# Output: ./kto_output_rtx3090/
```

### Option B: Train on Your Dataset

Prepare your data in JSONL format:

```jsonl
{"prompt": "Your question", "completion": "Good answer", "label": true}
{"prompt": "Your question", "completion": "Bad answer", "label": false}
```

Then train:

```bash
# Using wrapper script:
./train.sh --model-size 7b --local-file ./my_data.jsonl

# Or using Python directly:
# venv/bin/python train_kto.py --model-size 7b --local-file ./my_data.jsonl
```

## 6. Test Your Model

```bash
# Interactive chat
python src/inference.py ./kto_output_rtx3090/final_model

# You: What is machine learning?
# Assistant: [model generates response]
```

## 7. Upload to HuggingFace (Optional)

```bash
# Get token from: https://huggingface.co/settings/tokens

python src/upload_to_hf.py \
  ./kto_output_rtx3090/final_model \
  your-username/your-model-name \
  --token YOUR_HF_TOKEN
```

## Common Commands

### Fast Iteration (3B Model)
```bash
./train.sh --model-size 3b --batch-size 8
# Or: venv/bin/python train_kto.py --model-size 3b --batch-size 8
```
- Fastest training
- Good for testing
- ~30-40 tokens/sec

### Production Quality (7B Model) ⭐
```bash
./train.sh --model-size 7b
# Or: venv/bin/python train_kto.py --model-size 7b
```
- Best quality/speed balance
- Recommended for production
- ~20-30 tokens/sec

### Maximum Quality (13B Model)
```bash
./train.sh --model-size 13b
# Or: venv/bin/python train_kto.py --model-size 13b
```
- Highest quality
- Slower training
- ~15-25 tokens/sec

### With Experiment Tracking
```bash
./train.sh --model-size 7b --wandb --wandb-project my-project
# Or: venv/bin/python train_kto.py --model-size 7b --wandb --wandb-project my-project
```

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
python train_kto.py --model-size 7b --batch-size 2

# Or use smaller model
python train_kto.py --model-size 3b
```

### Slow Training
```bash
# Check GPU utilization
watch -n 1 nvidia-smi

# Should show:
# - GPU utilization: 90-100%
# - Memory usage: 8-15GB for 7B model
```

### Import Errors
```bash
# Reinstall dependencies
pip install --force-reinstall -r requirements.txt

# Or try without flash-attention
pip install -r requirements.txt --no-deps
pip install torch transformers datasets unsloth trl peft bitsandbytes accelerate
```

## Next Steps

1. **Read the full README.md** for detailed documentation
2. **Check configs/training_config.py** to customize hyperparameters
3. **Review rtx3090-kto-finetuning.md** in docs for theory and optimization techniques
4. **Monitor training** with `nvidia-smi` and training logs
5. **Experiment** with different models, batch sizes, and learning rates

## Expected Performance (RTX 3090)

| Model | VRAM | Batch | Speed | Quality |
|-------|------|-------|-------|---------|
| 3B | ~8GB | 8 | 30-40 tok/s | Good |
| 7B | ~10GB | 4 | 20-30 tok/s | Excellent ⭐ |
| 13B | ~15GB | 2 | 15-25 tok/s | Best |

## Resources

- Full documentation: `README.md`
- Configuration guide: `configs/training_config.py`
- Spec document: `docs/prep/local-training/rtx3090-kto-finetuning.md`
- Unsloth docs: https://docs.unsloth.ai/
- TRL KTO docs: https://huggingface.co/docs/trl/main/en/kto_trainer

## Support

Issues? Check:
1. NVIDIA drivers (535+)
2. CUDA version (12.1+)
3. Python version (3.10+)
4. Disk space (50GB+)
5. GPU memory (24GB)

Happy training! 🚀
