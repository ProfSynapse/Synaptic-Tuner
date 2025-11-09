# KTO Fine-Tuning on Mac M4 (24GB) - Comprehensive Guide

## Executive Summary

Running KTO (Kahneman-Tversky Optimization) fine-tuning on Apple Silicon M4 with 24GB unified memory requires a different approach than traditional CUDA-based workflows. While Unsloth does not currently support Apple Silicon/MPS, **Apple's MLX framework** provides an excellent alternative for local fine-tuning with LoRA on Mac hardware. However, as of January 2025, **KTO training is not natively supported in MLX** - the framework focuses on LoRA and QLoRA methods.

**Key Recommendation**: For KTO training specifically on M4, you have two primary options:
1. Use PyTorch with MPS backend and TRL's KTOTrainer (CPU fallback may be required for some operations)
2. Use MLX for LoRA fine-tuning as an alternative to KTO, which provides similar parameter-efficient training benefits

This guide covers both approaches, with emphasis on what works best on Apple Silicon hardware.

## Technology Overview

### Apple MLX Framework
- **What it is**: Apple's native machine learning framework specifically designed for Apple Silicon
- **Version**: Python 3.11+ required for latest features
- **GPU Support**: Full Metal Performance Shaders (MPS) integration for M-series chips
- **Primary Use**: Parameter-efficient fine-tuning with LoRA/QLoRA
- **Limitations**: No native KTO support as of January 2025

### PyTorch with MPS Backend
- **Version**: PyTorch 1.12+ includes MPS support
- **Compatibility**: Works with TRL library for KTO training
- **Performance**: Variable - some operations may be slower on MPS than CPU for certain workloads
- **Status**: MPS implementation still maturing compared to CUDA

### Unified Memory Architecture
- **24GB Shared Memory**: Used by both CPU and GPU (Metal)
- **Advantage**: No explicit GPU-CPU transfers needed
- **Model Capacity**: Can handle 7B models in full precision, 13B models with 4-bit quantization, larger models (30B+) with aggressive quantization

---

## Detailed Documentation

## Option 1: KTO Training with PyTorch + TRL (Limited Support)

### Installation Steps

```bash
# Create virtual environment
python3 -m venv kto_env
source kto_env/bin/activate

# Install PyTorch with MPS support
pip install torch torchvision torchaudio

# Install TRL and dependencies
pip install trl transformers datasets accelerate
pip install peft bitsandbytes

# Verify MPS availability
python -c "import torch; print('MPS Available:', torch.backends.mps.is_available())"
```

### Known Limitations

**CRITICAL**: As of January 2025:
- BitsAndBytes (used for 4-bit quantization) has **limited Apple Silicon support**
- Some TRL operations may fall back to CPU
- MPS performance can be slower than CPU for certain operations on M4
- Flash Attention is not available for MPS

### KTO Configuration for M4

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import KTOConfig, KTOTrainer
from datasets import load_dataset
import torch

# Load model - start with smaller models (3B-7B recommended)
model_name = "Qwen/Qwen2-0.5B-Instruct"  # Start small
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="mps" if torch.backends.mps.is_available() else "cpu",
    torch_dtype=torch.float16,  # Use FP16 for memory efficiency
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load KTO dataset
train_dataset = load_dataset("trl-lib/kto-mix-14k", split="train")

# Configure KTO training
training_args = KTOConfig(
    output_dir="./kto_output_m4",
    num_train_epochs=1,
    per_device_train_batch_size=2,  # Conservative for MPS
    gradient_accumulation_steps=8,   # Effective batch size = 16
    learning_rate=5e-7,              # Conservative learning rate
    max_length=512,                  # Shorter sequences for memory
    max_prompt_length=256,
    beta=0.1,                        # KTO beta parameter
    desirable_weight=1.0,
    undesirable_weight=1.0,
    gradient_checkpointing=True,     # Essential for memory
    logging_steps=10,
    save_steps=100,
    fp16=False,                      # MPS doesn't support fp16 training well
    bf16=False,                      # M4 doesn't support bf16
)

# Initialize trainer
trainer = KTOTrainer(
    model=model,
    args=training_args,
    processing_class=tokenizer,
    train_dataset=train_dataset,
)

# Train
trainer.train()
```

### Memory Optimization Techniques

1. **Reduce Batch Size**: Start with `per_device_train_batch_size=1` or `2`
2. **Gradient Accumulation**: Use `gradient_accumulation_steps=8-16` to maintain effective batch size
3. **Shorter Sequences**: Set `max_length=512` or `max_length=1024` instead of longer contexts
4. **Gradient Checkpointing**: Always enable with `gradient_checkpointing=True`
5. **Remove Reference Model**: KTO can precompute reference log probs to save memory

### Expected Performance

- **Training Speed**: Approximately 1-3 tokens/second for 7B models (highly variable)
- **Memory Usage**:
  - 3B model: ~6-8 GB
  - 7B model FP16: ~14-18 GB
  - 13B model FP16: May exceed 24GB - not recommended

### Recommended Models for M4 24GB (PyTorch/KTO)

| Model | Size | Memory Usage (FP16) | Training Feasibility |
|-------|------|---------------------|---------------------|
| Qwen2-0.5B-Instruct | 0.5B | ~2 GB | Excellent |
| Qwen2.5-3B-Instruct | 3B | ~6 GB | Excellent |
| Llama-3.2-3B-Instruct | 3B | ~6 GB | Excellent |
| Mistral-7B-Instruct-v0.3 | 7B | ~14 GB | Good (tight) |
| Llama-3.1-8B-Instruct | 8B | ~16 GB | Marginal |
| Qwen2.5-7B-Instruct | 7B | ~14 GB | Good (tight) |

---

## Option 2: MLX Fine-Tuning with LoRA (RECOMMENDED for M4)

### Why MLX Instead of KTO?

While MLX doesn't support KTO training, it offers:
- **Native Apple Silicon optimization** - 2-3x faster than PyTorch MPS
- **Efficient LoRA/QLoRA** - Similar parameter-efficient fine-tuning benefits
- **Better memory utilization** - Optimized for unified memory architecture
- **Active development** - Strong support from Apple
- **Proven track record** - Widely used for Mac fine-tuning workflows

### Installation Steps

```bash
# Create virtual environment
python3 -m venv mlx_env
source mlx_env/bin/activate

# Verify Python version (3.11+ required)
python --version

# Install MLX and mlx-lm
pip install mlx
pip install mlx-lm

# Install additional dependencies
pip install huggingface_hub datasets pandas

# Verify installation
python -c "import mlx_lm; print('MLX LM installed successfully')"
```

### Dataset Preparation for MLX LoRA

MLX expects datasets in JSONL format with specific structure:

```json
{"text": "User: What is the capital of France?\nAssistant: The capital of France is Paris."}
{"text": "User: Explain quantum computing.\nAssistant: Quantum computing uses quantum bits or qubits..."}
```

For instruction tuning, you need `train.jsonl` and `valid.jsonl` in your data folder.

**Converting KTO datasets to MLX format**:

```python
from datasets import load_dataset
import json

# Load KTO dataset
dataset = load_dataset("trl-lib/kto-mix-14k", split="train")

# Convert to MLX format (using only desirable examples)
with open("train.jsonl", "w") as f:
    for item in dataset:
        if item.get("label") == True:  # Use only positive examples
            text = f"User: {item['prompt']}\nAssistant: {item['completion']}"
            json.dump({"text": text}, f)
            f.write("\n")

# Create validation split (use last 10%)
# ... similar process for valid.jsonl
```

### MLX LoRA Fine-Tuning Configuration

```bash
# Basic fine-tuning command
python -m mlx_lm.lora \
    --model microsoft/Phi-3.5-mini-instruct \
    --train \
    --data ./data \
    --iters 1000 \
    --steps-per-eval 100 \
    --batch-size 4 \
    --lora-layers 16 \
    --learning-rate 1e-5
```

**Advanced configuration for M4 24GB**:

```bash
# For 7B models with optimal settings
python -m mlx_lm.lora \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --train \
    --data ./data \
    --iters 2000 \
    --steps-per-eval 200 \
    --steps-per-report 10 \
    --batch-size 8 \
    --lora-layers 32 \
    --lora-rank 8 \
    --learning-rate 5e-6 \
    --max-seq-length 1024 \
    --adapter-path ./adapters \
    --grad-checkpoint
```

### MLX Configuration Parameters Explained

| Parameter | Description | Recommended for 24GB M4 |
|-----------|-------------|-------------------------|
| `--batch-size` | Training batch size | 4-8 for 7B, 8-16 for 3B |
| `--lora-layers` | Number of layers to apply LoRA | 16-32 (more = better but slower) |
| `--lora-rank` | LoRA rank dimension | 8-16 (higher = more capacity) |
| `--learning-rate` | Learning rate | 1e-5 to 5e-6 |
| `--max-seq-length` | Maximum sequence length | 512-2048 |
| `--grad-checkpoint` | Enable gradient checkpointing | Always use for large models |
| `--iters` | Training iterations | 1000-5000 depending on dataset |

### Recommended Models for M4 24GB (MLX/LoRA)

| Model | Size | Quantization | Tokens/Sec | Memory Usage | Training Recommended |
|-------|------|-------------|------------|--------------|---------------------|
| Qwen2.5-3B-Instruct | 3B | None (FP16) | ~25-30 | ~6 GB | Excellent |
| Llama-3.2-3B-Instruct | 3B | None (FP16) | ~25-30 | ~6 GB | Excellent |
| Mistral-7B-Instruct-v0.3 | 7B | None (FP16) | ~12-15 | ~14 GB | Good |
| Qwen2.5-7B-Instruct | 7B | None (FP16) | ~12-15 | ~14 GB | Good |
| Llama-3.1-8B-Instruct | 8B | 4-bit MLX | ~15-20 | ~8 GB | Good |
| Qwen3-14B-Instruct | 14B | 4-bit MLX | ~8-10 | ~12 GB | Feasible |
| Qwen3-30B (MoE) | 30B | 4-bit MLX | ~5-8 | ~16 GB | Marginal |

**Note**: MLX community on HuggingFace provides pre-quantized models optimized for Apple Silicon.

### Memory Optimization for MLX

1. **Use Pre-Quantized Models**: Search for `mlx-community` models on HuggingFace
2. **Reduce LoRA Rank**: Start with `--lora-rank 4` or `8` instead of 16
3. **Smaller Batch Sizes**: Use `--batch-size 2` or `4` for larger models
4. **Shorter Sequences**: Set `--max-seq-length 512` for memory-constrained scenarios
5. **Fewer LoRA Layers**: Use `--lora-layers 16` instead of 32 for large models

### Expected Performance (MLX)

**Training Speed** (M4 24GB):
- 3B models: ~25-30 tokens/second
- 7B models: ~12-15 tokens/second
- 14B models (4-bit): ~8-10 tokens/second

**Training Time Examples**:
- 3B model, 1000 examples, 1 epoch: ~15-20 minutes
- 7B model, 1000 examples, 1 epoch: ~30-40 minutes

### Model Conversion and Inference

After training, merge LoRA adapters and convert for deployment:

```bash
# Fuse LoRA adapter with base model
python -m mlx_lm.fuse \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --adapter-path ./adapters \
    --save-path ./fused-model

# Test the fused model
python -m mlx_lm.generate \
    --model ./fused-model \
    --prompt "User: Explain machine learning.\nAssistant:" \
    --max-tokens 256
```

---

## Comparative Analysis: PyTorch KTO vs MLX LoRA on M4

| Aspect | PyTorch + TRL (KTO) | MLX + LoRA |
|--------|---------------------|------------|
| **KTO Support** | ✅ Yes (with limitations) | ❌ No (LoRA only) |
| **Performance** | ~1-3 tok/s (7B) | ~12-15 tok/s (7B) |
| **Memory Efficiency** | Moderate | Excellent |
| **Apple Silicon Optimization** | Limited (MPS still maturing) | Excellent (native) |
| **Setup Complexity** | Moderate (dependency issues) | Simple |
| **4-bit Quantization** | Limited (BitsAndBytes issues) | Excellent (native MLX quant) |
| **Flash Attention** | ❌ Not available | ✅ Optimized attention |
| **Gradient Checkpointing** | ✅ Yes | ✅ Yes |
| **Community Support** | Large (PyTorch/HF) | Growing (Apple/MLX) |
| **Production Readiness** | Mature for CUDA, experimental for MPS | Stable for Apple Silicon |

---

## KTO Dataset Format Requirements

KTO training requires datasets with binary labels indicating desirable/undesirable outputs.

### Standard Unpaired Format (Recommended for KTO)

```json
{"prompt": "What is the capital of France?", "completion": "Paris is the capital of France.", "label": true}
{"prompt": "What is the capital of France?", "completion": "London is the capital.", "label": false}
{"prompt": "Explain quantum computing.", "completion": "Quantum computing uses qubits to perform calculations faster than classical computers.", "label": true}
```

### Conversational Format

```json
{
  "prompt": [{"role": "user", "content": "What color is the sky?"}],
  "completion": [{"role": "assistant", "content": "The sky is blue."}],
  "label": true
}
{
  "prompt": [{"role": "user", "content": "What color is the sky?"}],
  "completion": [{"role": "assistant", "content": "The sky is green."}],
  "label": false
}
```

### Paired Preference Format (Alternative)

```json
{
  "prompt": "What is machine learning?",
  "chosen": "Machine learning is a subset of AI that enables systems to learn from data.",
  "rejected": "Machine learning is just programming."
}
```

**Field Descriptions**:
- `prompt`: The input question or instruction
- `completion`: The model's output (unpaired format)
- `label`: Boolean - `true` for desirable, `false` for undesirable
- `chosen`: Preferred response (paired format)
- `rejected`: Non-preferred response (paired format)

**Dataset Requirements**:
- Minimum: At least one desirable and one undesirable completion
- Recommended: Balanced dataset with ~50/50 ratio of desirable/undesirable
- For imbalanced data: Adjust `desirable_weight` and `undesirable_weight` in KTOConfig

---

## Troubleshooting Guide

### Issue: MPS device not available

**Solution**:
```bash
# Check PyTorch MPS support
python -c "import torch; print(torch.backends.mps.is_available())"

# If false, reinstall PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio
```

### Issue: BitsAndBytes not working on Mac

**Symptom**: Errors when using 4-bit quantization with `load_in_4bit=True`

**Solution**: BitsAndBytes has limited Mac support. Options:
1. Use FP16 instead of quantization
2. Switch to MLX which has native quantization support
3. Use pre-quantized models from HuggingFace

### Issue: Training very slow on MPS

**Symptom**: MPS slower than CPU for training

**Solution**: This is a known issue for some operations on M4:
1. Try CPU-only training: Set `device_map="cpu"`
2. Switch to MLX framework (recommended)
3. Reduce batch size and use gradient accumulation
4. Use smaller models (3B instead of 7B)

### Issue: Out of memory errors

**Solutions**:
1. Reduce `per_device_train_batch_size` to 1
2. Increase `gradient_accumulation_steps`
3. Reduce `max_length` and `max_prompt_length`
4. Enable `gradient_checkpointing=True`
5. Use smaller model (3B instead of 7B)
6. Switch to 4-bit quantized models in MLX

### Issue: MLX model not found

**Solution**: Use mlx-community pre-converted models:
```python
# Instead of: "meta-llama/Llama-3.1-8B"
# Use: "mlx-community/Llama-3.1-8B-Instruct-4bit"
```

### Issue: Slow dataset loading

**Solution**:
```python
# Disable multiprocessing on Mac
dataset = load_dataset("...", num_proc=1)
```

---

## Best Practices and Recommendations

### For KTO Training on M4

1. **Start Small**: Begin with 0.5B-3B models to test your pipeline
2. **Use CPU for Some Operations**: MPS may not be faster for all operations
3. **Monitor Memory**: Use Activity Monitor to track memory usage
4. **Conservative Settings**: Use small batch sizes and low learning rates
5. **Gradient Accumulation**: Maintain effective batch size of 16-64
6. **Short Sequences**: Start with 512 tokens, expand if memory allows
7. **Test on Sample**: Validate setup with 100-1000 examples before full training

### For LoRA Training with MLX (Recommended)

1. **Use Pre-Quantized Models**: mlx-community models are optimized for M4
2. **Optimal Batch Sizes**: 4-8 for 7B models, 8-16 for 3B models
3. **LoRA Configuration**: rank=8, layers=16-32 for good balance
4. **Monitor Training**: Use `--steps-per-report 10` to track progress
5. **Validation**: Always include validation dataset
6. **Checkpointing**: Save adapters frequently with `--steps-per-save`
7. **Test Inference**: Verify adapter quality before fusing

### Model Selection Guidelines

**Choose 3B models if**:
- You want fastest training
- You're experimenting with hyperparameters
- You have limited time
- You're training on smaller datasets (<10k examples)

**Choose 7B models if**:
- You need better quality
- You have larger datasets (>10k examples)
- You can accept 2-3x longer training time
- You're fine-tuning for production use

**Avoid 13B+ models unless**:
- Using aggressive 4-bit quantization in MLX
- Training on very small datasets
- You have patience for slow training speeds

---

## Security Considerations

### Model Download Security

```python
# Always verify model sources
from huggingface_hub import scan_cache_dir

# Check downloaded models
scan_info = scan_cache_dir()
for repo in scan_info.repos:
    print(f"Repo: {repo.repo_id}")
    print(f"Size: {repo.size_on_disk_str}")
```

### Data Privacy

- **Local Training**: All training happens on-device - data never leaves your Mac
- **Model Weights**: Downloaded from HuggingFace - verify trusted sources
- **API Keys**: If using HuggingFace Hub, store tokens securely:
  ```bash
  huggingface-cli login
  # Tokens stored in ~/.huggingface/token
  ```

### Secure Model Deployment

After training:
1. Scan model outputs for PII before deploying
2. Test model for harmful outputs
3. Implement content filtering if deploying publicly
4. Version control your adapters and training scripts

---

## Resource Links

### Official Documentation
- [Apple MLX GitHub](https://github.com/ml-explore/mlx) - Official MLX repository
- [MLX LM Documentation](https://github.com/ml-explore/mlx-examples/tree/main/llms) - MLX language model tools
- [PyTorch MPS Backend](https://pytorch.org/docs/stable/notes/mps.html) - PyTorch Metal support (Updated 2024)
- [TRL KTO Trainer](https://huggingface.co/docs/trl/main/en/kto_trainer) - Official KTO documentation (February 2024)
- [Hugging Face TRL](https://huggingface.co/docs/trl/index) - Transformer Reinforcement Learning library

### Research Papers
- [KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306) - Original KTO paper (February 2024)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) - QLoRA methodology (May 2023)

### Community Resources
- [MLX Community Models](https://huggingface.co/mlx-community) - Pre-quantized MLX models
- [TRL Examples](https://github.com/huggingface/trl/tree/main/examples) - KTO training examples
- [MLX Examples](https://github.com/ml-explore/mlx-examples) - Official MLX examples

### Tutorials and Guides
- [LoRA Fine-Tuning on Apple Silicon](https://towardsdatascience.com/lora-fine-tuning-on-your-apple-silicon-macbook-432c7dab614a) - Comprehensive guide (2024)
- [Fine-tuning LLMs with MLX locally](https://heidloff.net/article/apple-mlx-fine-tuning/) - Practical tutorial (2024)
- [Simple Guide to MLX Fine-tuning](https://apeatling.com/articles/simple-guide-to-local-llm-fine-tuning-on-a-mac-with-mlx/) - Beginner-friendly (2024)

---

## Recommendations

### Primary Recommendation for M4 24GB

**Use MLX with LoRA instead of KTO** for the following reasons:

1. **Performance**: 5-10x faster training than PyTorch MPS
2. **Reliability**: Native Apple Silicon support, mature and stable
3. **Memory Efficiency**: Better utilization of unified memory
4. **Community**: Strong ecosystem of pre-quantized models
5. **Simplicity**: Easier setup, fewer dependency issues

**Trade-off**: You lose KTO-specific training but gain LoRA, which provides similar parameter-efficient fine-tuning benefits with proven results.

### When to Use PyTorch KTO on M4

Only consider PyTorch + TRL KTO if:
- KTO training method is absolutely required for your research
- You're comparing KTO against other alignment methods
- You have time to work through MPS compatibility issues
- You're willing to accept slower training speeds

**Expect**: 3-5x longer training times compared to MLX, potential compatibility issues, and need for extensive troubleshooting.

### Workflow Recommendation

1. **Start with MLX**: Validate your dataset and workflow with LoRA
2. **Use 3B Models**: Test with Qwen2.5-3B or Llama-3.2-3B first
3. **Scale Up Gradually**: Move to 7B once pipeline is stable
4. **Consider Cloud**: For true KTO at scale, consider cloud GPU instances
5. **Hybrid Approach**: Use MLX for fast iteration, cloud for final KTO training

### Future Outlook

- **MLX KTO Support**: May be added in future MLX releases - monitor repository
- **PyTorch MPS Improvements**: MPS backend continues to improve with each release
- **M4 Optimization**: Expect better performance as libraries optimize for M4 architecture

---

**Last Updated**: January 2025
**Compatibility**: macOS 14.0+, M4/M3/M2/M1 chips, Python 3.11+, MLX 0.x, PyTorch 2.x
