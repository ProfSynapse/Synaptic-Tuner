# Unsloth Quick Reference Guide

## 1. Import Order (CRITICAL!)

```python
import unsloth  # MUST be first
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer, SFTConfig
from transformers import ...
from peft import ...
```

## 2. Model Loading One-Liners

### Llama 7B (RTX 3090 Optimized)
```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/Llama-2-7b-chat-hf",
    max_seq_length=1024,
    load_in_4bit=True,
    dtype=torch.bfloat16,
)
```

### Mistral 7B (Maximum Quality)
```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="mistralai/Mistral-7B-v0.1",
    max_seq_length=2048,
    load_in_8bit=True,  # Better accuracy
    dtype=torch.bfloat16,
)
```

### Small Model (1B-3B)
```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-1B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)
```

### Large Model (Aggressive Memory)
```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-20B-4bit",
    max_seq_length=256,  # Very small
    load_in_4bit=True,
)
```

## 3. LoRA Setup One-Liners

### Standard Configuration
```python
model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth",
)
```

### Maximum Parameters (Higher Learning Capacity)
```python
model = FastLanguageModel.get_peft_model(
    model,
    r=64,
    lora_alpha=64,
    target_modules="all-linear",  # All layers
    use_gradient_checkpointing="unsloth",
)
```

### Minimum Memory (Aggressive Optimization)
```python
model = FastLanguageModel.get_peft_model(
    model,
    r=8,
    lora_alpha=8,
    target_modules=["q_proj", "v_proj"],  # Attention only
    use_gradient_checkpointing="unsloth",
    lora_dropout=0.1,
)
```

## 4. Training Configurations by Model Size

### 1B Model (RTX 3090)
```python
SFTConfig(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    max_seq_length=2048,
    learning_rate=2e-4,
    num_train_epochs=1,
    bf16=True,
    optim="adamw_8bit",
)
# Expected VRAM: 8-10GB
```

### 7B Model (RTX 3090)
```python
SFTConfig(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    max_seq_length=1024,
    learning_rate=2e-4,
    warmup_ratio=0.1,
    bf16=True,
    optim="adamw_8bit",
)
# Expected VRAM: 18-22GB
```

### 13B Model (RTX 3090 - Minimal)
```python
SFTConfig(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    max_seq_length=512,
    learning_rate=2e-4,
    bf16=True,
    optim="adamw_8bit",
)
# Expected VRAM: 22-24GB (tight!)
```

### 20B+ Model (Challenging)
```python
SFTConfig(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    max_seq_length=256,  # Very aggressive
    learning_rate=2e-4,
    bf16=True,
    optim="adamw_8bit",
)
# Expected VRAM: 23-24GB (extremely tight)
```

## 5. Memory Reduction Techniques (Priority Order)

| Technique | Memory Saved | How |
|-----------|-------------|-----|
| 4-bit quantization | 75% | `load_in_4bit=True` |
| 8-bit optimizer | 75% | `optim="adamw_8bit"` |
| Gradient checkpointing | 20-30% | `use_gradient_checkpointing="unsloth"` |
| Reduce batch size | 30-50% | `per_device_train_batch_size=1` |
| Reduce sequence length | 50% | `max_seq_length=256` |
| Lower LoRA rank | 10-20% | `r=8` instead of `r=64` |
| Gradient accumulation | 0% (trades speed) | Accumulate steps |

## 6. Parameter Impact on Training

| Parameter | Impact | Trade-off |
|-----------|--------|-----------|
| `r` (LoRA rank) | Higher = More capacity | Memory increases quadratically |
| `max_seq_length` | Higher = Longer context | Memory increases quadratically |
| `batch_size` | Higher = Faster training | Memory increases linearly |
| `max_seq_length` | Larger models | Memory increases linearly |
| `gradient_accum` | More = Larger effective batch | No memory increase, slower |
| `learning_rate` | Typically 2e-4 for LoRA | Higher = instability |

## 7. Quantization Decision Matrix

```
Choose your setup:

┌─ Memory Critical (< 10GB)
│  └─ Use 4-bit QLoRA + aggressive checkpointing + batch=1
│     Sequence: 256-512
│
├─ Normal (10-20GB)
│  └─ Use 4-bit QLoRA + checkpointing + batch=2
│     Sequence: 1024
│
├─ Comfortable (> 20GB)
│  └─ Use 8-bit LoRA + checkpointing + batch=2-4
│     Sequence: 1024-2048
│
└─ Accuracy Critical
   └─ Use 16-bit full fine-tuning
      Sequence: Limited by model
```

## 8. Common Issues & Fixes

| Issue | Cause | Fix |
|-------|-------|-----|
| OOM Error | Batch too large | Reduce `per_device_train_batch_size` |
| OOM Error | Sequence too long | Reduce `max_seq_length` |
| Slow training | Missing optimizations | Check import order |
| Low accuracy | 4-bit quantization | Use `load_in_8bit=True` |
| Slow fine-tuning | No gradient checkpointing | Add `use_gradient_checkpointing="unsloth"` |
| Model not loading | Incompatible version | Update: `pip install --upgrade transformers` |
| CUDA errors | Bad CUDA setup | Run: `python -m bitsandbytes` |

## 9. Complete Training Pipeline

```python
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import torch

# 1. Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="meta-llama/Llama-3.2-7B-Instruct",
    max_seq_length=1024,
    load_in_4bit=True,
    dtype=torch.bfloat16,
)

# 2. Add LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth",
)

# 3. Load data
dataset = load_dataset("json", data_files="train.jsonl")
dataset = dataset.map(lambda x: {"text": x["instruction"] + x["response"]})

# 4. Configure training
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        max_seq_length=1024,
        learning_rate=2e-4,
        num_train_epochs=1,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        optim="adamw_8bit",
        output_dir="output",
    ),
)

# 5. Train
trainer.train()

# 6. Save
model.save_pretrained_merged("final_model", tokenizer=tokenizer)
```

## 10. Environment Setup

```bash
# Install Unsloth
pip install unsloth

# Or with specific PyTorch version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install unsloth

# Update to latest
pip install --upgrade --force-reinstall --no-cache-dir unsloth unsloth_zoo

# Verify CUDA
python -m bitsandbytes

# Check transformers version
pip show transformers  # Should be >= 4.45.2
```

## 11. Key Files to Remember

| File | What It Does |
|------|-------------|
| `/unsloth/models/loader.py` | FastLanguageModel API |
| `/unsloth/models/_utils.py` | Memory optimizations |
| `/unsloth/kernels/fast_lora.py` | LoRA kernels |
| `/unsloth/save.py` | Model merging & saving |
| `/unsloth/trainer.py` | Training utilities |

## 12. Supported Models (Partial List)

- Llama: 3.1, 3.2, 4
- Mistral: 7B, v0.3
- Qwen: 2, 2.5, 3, 3-MOE
- Phi: 3, 3.5, 4
- Gemma: 2, 3, 3n
- DeepSeek: R1, various
- Cohere, Falcon, Granite

## 13. Useful Helper Functions

```python
# Check BFloat16 support
from unsloth import is_bfloat16_supported
print(is_bfloat16_supported())  # True on RTX 3090

# Load chat template
from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

# For inference
FastLanguageModel.for_inference(model)  # Optimizes for inference

# Back to training
FastLanguageModel.for_training(model)   # Restores training mode
```

## 14. GPU Memory Formula

For **4-bit QLoRA with gradient checkpointing**:

```
Approximate VRAM = 3.5 * batch_size * max_seq_length * model_size_billions / 8

Examples:
- 7B model, batch=2, seq=1024 → ~6 GB
- 13B model, batch=2, seq=512 → ~5 GB
- 20B model, batch=1, seq=256 → ~3 GB

Add 2-4 GB for overhead (optimizer state, activations)
```

## 15. Debugging Commands

```bash
# Check GPU usage
nvidia-smi -l 1  # Refresh every 1 second

# Monitor during training
watch -n 1 nvidia-smi

# Test setup
python -c "
from unsloth import FastLanguageModel
import torch
print(f'BFloat16: {torch.cuda.is_bf16_supported()}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'Device: {torch.cuda.get_device_name()}')
"

# Verify Unsloth imports
python -c "
import unsloth
from unsloth import FastLanguageModel
print('Unsloth imported successfully')
"
```

---

**Note**: All configurations are optimized for RTX 3090 (24GB VRAM). Adjust `per_device_train_batch_size` and `max_seq_length` for different GPUs.
