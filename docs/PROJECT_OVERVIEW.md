# RTX 3090 KTO Training - Project Overview

## Summary

Complete production-ready implementation of KTO (Kahneman-Tversky Optimization) fine-tuning optimized for NVIDIA RTX 3090 GPUs with 24GB VRAM. This implementation is based on the comprehensive specification documented in `/docs/prep/local-training/rtx3090-kto-finetuning.md` and the Colab notebook `kto_colab_notebook.ipynb`.

## Key Features

✅ **Optimized for RTX 3090 (24GB VRAM)**
- Supports 3B to 13B parameter models
- Memory-efficient 4-bit quantization
- Smart batch size configurations per model tier

✅ **Production Ready**
- Complete training pipeline
- Inference utilities
- HuggingFace upload (including GGUF)
- Comprehensive error handling

✅ **Performance Optimized**
- Unsloth integration (2x speedup)
- Flash Attention support
- 8-bit optimizers
- Gradient checkpointing options

✅ **Easy to Use**
- Simple CLI interface
- Multiple configuration presets
- Dry-run mode for testing
- Interactive chat inference

## Project Structure

```
rtx3090_kto/
├── README.md                    # Full documentation
├── QUICKSTART.md               # 5-minute setup guide
├── PROJECT_OVERVIEW.md         # This file
├── requirements.txt            # Python dependencies
├── setup.sh                    # Automated setup script
├── test_installation.py        # Installation verification
├── train_kto.py               # Main training script
├── .gitignore                 # Git ignore rules
│
├── configs/
│   └── training_config.py     # Configuration presets (3B/7B/13B/20B)
│
├── src/
│   ├── data_loader.py         # Dataset loading & ChatML conversion
│   ├── model_loader.py        # Model loading with Unsloth
│   ├── inference.py           # Inference & interactive chat
│   └── upload_to_hf.py        # HuggingFace upload (standard + GGUF)
│
└── examples/
    └── sample_dataset.jsonl   # Sample training data
```

## Quick Reference

### Installation
```bash
bash setup.sh
source venv/bin/activate
python test_installation.py
```

### Training
```bash
# Fast (3B model)
python train_kto.py --model-size 3b

# Recommended (7B model)
python train_kto.py --model-size 7b

# Best quality (13B model)
python train_kto.py --model-size 13b
```

### Inference
```bash
python src/inference.py ./kto_output_rtx3090/final_model
```

### Upload
```bash
python src/upload_to_hf.py \
  ./kto_output_rtx3090/final_model \
  username/model-name \
  --token YOUR_HF_TOKEN \
  --create-gguf
```

## Configuration Presets

### 3B Models (Fast Iteration)
- **Batch size**: 8 (effective: 32)
- **Speed**: 30-40 tokens/sec
- **VRAM**: ~8-10 GB
- **Use**: Prototyping, testing

### 7B Models (Production) ⭐ **Recommended**
- **Batch size**: 4 (effective: 32)
- **Speed**: 20-30 tokens/sec
- **VRAM**: ~9-11 GB
- **Use**: Production deployments

### 13B Models (Advanced)
- **Batch size**: 2 (effective: 32)
- **Speed**: 15-25 tokens/sec
- **VRAM**: ~14-16 GB
- **Use**: Maximum quality

### 20B Models (Specialized)
- **Batch size**: 4 (effective: 32)
- **Speed**: 10-15 tokens/sec
- **VRAM**: ~18-20 GB
- **Use**: Large-scale tasks

## Memory Optimizations

The implementation uses multiple optimization techniques:

1. **4-bit NF4 Quantization**: 75% memory reduction
2. **8-bit AdamW Optimizer**: ~2GB VRAM savings
3. **Gradient Checkpointing**: 40-50% activation memory reduction (optional)
4. **Unsloth Optimizations**: 70% VRAM reduction vs standard PyTorch
5. **Flash Attention**: 3-5x faster attention computation (optional)

## Performance Benchmarks (RTX 3090)

### Training Speed (with Unsloth)
| Model | Tokens/Sec | Examples/Hour | Time per 10k Examples |
|-------|------------|---------------|----------------------|
| 3B | 30-40 | 2000+ | ~5 hours |
| 7B | 20-30 | 900+ | ~11 hours |
| 13B | 15-25 | 600+ | ~17 hours |

### VRAM Usage
| Model | Base | + Optimizer | + Activations (batch=4) | Total |
|-------|------|-------------|------------------------|-------|
| 3B | ~2GB | +1GB | +2GB | ~5-6GB |
| 7B | ~3.5GB | +1.5GB | +3GB | ~8-10GB |
| 13B | ~6.5GB | +3GB | +4GB | ~13-15GB |

## Dataset Format

### KTO Format (Unpaired Preferences)
```jsonl
{"prompt": "Question", "completion": "Good answer", "label": true}
{"prompt": "Question", "completion": "Bad answer", "label": false}
```

### ChatML Format (Auto-converted)
```jsonl
{
  "conversations": [
    {"role": "user", "content": "Question"},
    {"role": "assistant", "content": "Answer"}
  ],
  "label": true
}
```

## Common Use Cases

### 1. Quick Experimentation
```bash
python train_kto.py \
  --model-size 3b \
  --local-file examples/sample_dataset.jsonl \
  --num-epochs 1 \
  --dry-run
```

### 2. Production Training
```bash
python train_kto.py \
  --model-size 7b \
  --wandb \
  --wandb-project production-models
```

### 3. Custom Dataset
```bash
python train_kto.py \
  --model-size 7b \
  --local-file my_dataset.jsonl \
  --batch-size 4 \
  --learning-rate 5e-7 \
  --num-epochs 2
```

### 4. Resume from Checkpoint
```bash
python train_kto.py \
  --model-size 7b \
  --output-dir ./kto_output_rtx3090 \
  # Training will resume from last checkpoint if found
```

## Troubleshooting

### CUDA Out of Memory
1. Reduce batch size: `--batch-size 2`
2. Reduce sequence length: `--max-seq-length 1024`
3. Use smaller model: `--model-size 3b`
4. Enable gradient checkpointing (edit config)

### Slow Training
1. Verify Unsloth is installed
2. Check GPU utilization with `nvidia-smi`
3. Ensure FP16/BF16 is enabled
4. Reduce dataloader workers if on Windows

### Dataset Issues
1. Ensure labels are boolean (true/false)
2. Check for empty prompts/completions
3. Validate balanced True/False distribution
4. Run: `python src/data_loader.py` to test

## Technical Details

### Model Loading
- Uses Unsloth's `FastLanguageModel` for optimized loading
- 4-bit NF4 quantization with double quantization
- Automatic dtype detection (FP16/BF16)
- Pre-quantized models from Unsloth hub

### LoRA Configuration
- **3B models**: r=32, alpha=64
- **7B models**: r=64, alpha=128
- **13B models**: r=64, alpha=128
- **20B models**: r=128, alpha=256
- Target modules: All attention + FFN layers

### KTO Training
- **Beta**: 0.1 (default), 0.05 for 20B models
- **Learning rate**: 5e-7 (conservative for KTO)
- **Warmup**: 10% of training steps
- **Optimizer**: AdamW 8-bit
- **Scheduler**: Cosine with warmup

## Integration with Existing Workflow

This implementation is designed to work with:
- **Claudesidian Dataset**: `professorsynapse/claudesidian-synthetic-dataset`
- **Mac M4 Implementation**: Compatible training scripts
- **Existing Notebooks**: Can be adapted from Colab notebook

## Future Enhancements

Potential improvements:
- [ ] Multi-GPU support (DDP/FSDP)
- [ ] Automated hyperparameter tuning
- [ ] Advanced monitoring (TensorBoard)
- [ ] Model merging utilities
- [ ] Quantization experiments (3-bit, 2-bit)
- [ ] Model evaluation suite

## Resources

### Documentation
- Full README: `README.md`
- Quick start: `QUICKSTART.md`
- Spec document: `/docs/prep/local-training/rtx3090-kto-finetuning.md`
- Reference notebook: `kto_colab_notebook.ipynb`

### External Links
- [Unsloth Documentation](https://docs.unsloth.ai/)
- [TRL KTO Trainer](https://huggingface.co/docs/trl/main/en/kto_trainer)
- [KTO Paper](https://arxiv.org/abs/2402.01306)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)

## Version Information

**Created**: November 2025
**Based on**:
- rtx3090-kto-finetuning.md (January 2025)
- kto_colab_notebook.ipynb (November 2025)

**Tested with**:
- Hardware: NVIDIA RTX 3090 24GB
- CUDA: 12.1
- PyTorch: 2.2.0+
- Transformers: 4.40.0+
- TRL: 0.8.0+
- Unsloth: Latest

## License

This implementation is provided for research and educational purposes.

## Support

For issues:
1. Check `README.md` troubleshooting section
2. Run `python test_installation.py`
3. Review specification document
4. Check Unsloth/TRL documentation

---

**Ready to train? Start with:**
```bash
bash setup.sh
python train_kto.py --model-size 7b --dry-run
python train_kto.py --model-size 7b
```
