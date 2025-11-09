# Unsloth Codebase Analysis: Comprehensive Exploration

## Executive Summary

Unsloth is a high-performance fine-tuning library that achieves **2x faster training with 70% less VRAM** through advanced kernel optimizations, memory-efficient implementations, and strategic quantization strategies. The codebase supports KTO training (through TRL integration), comprehensive LoRA configurations, and multi-quantization schemes optimized for GPUs like RTX 3090.

---

## 1. Overall Codebase Structure

### Directory Organization
```
/Users/jrosenbaum/Documents/Code/unsloth/
├── unsloth/                          # Main package
│   ├── __init__.py                   # Entry point with critical import ordering
│   ├── models/                       # Model implementations
│   │   ├── loader.py                 # FastLanguageModel - main API
│   │   ├── llama.py                  # LLaMA optimizations (largest: ~3800 lines)
│   │   ├── mistral.py                # Mistral model support
│   │   ├── qwen*.py                  # Qwen family implementations
│   │   ├── vision.py                 # FastBaseModel for vision/multimodal
│   │   ├── rl.py                     # RL training (GRPO, GSPO, etc.)
│   │   ├── dpo.py                    # DPO/KTO trainer patches
│   │   ├── _utils.py                 # Memory optimization & patching
│   │   └── gemma*.py, granite.py     # Other model architectures
│   ├── kernels/                      # Triton-based optimizations
│   │   ├── fast_lora.py              # Fused LoRA kernels
│   │   ├── cross_entropy_loss.py     # Cut cross-entropy (Apple collab)
│   │   ├── flex_attention.py         # Long-context attention
│   │   ├── fp8.py                    # FP8 quantization kernels
│   │   └── moe/                      # Mixture of Experts support
│   ├── save.py                       # Model saving & merging logic
│   ├── tokenizer_utils.py            # Tokenizer operations & chat templates
│   ├── trainer.py                    # UnslothTrainer (SFTConfig wrapper)
│   └── registry/                     # Model registry & configurations
├── tests/                            # Comprehensive test suite
│   ├── qlora/                        # QLoRA training tests
│   ├── saving/                       # Model save/merge validation
│   └── utils/                        # Test utilities
└── pyproject.toml                    # Dependencies and configuration
```

**Key Entry Point Order**: `/unsloth/__init__.py` specifically warns users to import unsloth BEFORE trl, transformers, and peft to ensure all optimizations are applied.

---

## 2. KTO Training Implementation & Support

### Status in Unsloth
- **KTO Support**: Unsloth provides a **PatchKTOTrainer** stub in `/unsloth/models/dpo.py`
- **Current Implementation**: The DPO/KTO functionality delegates to TRL (Transformers Reinforcement Learning) library
- **Location**: `/unsloth/unsloth/models/dpo.py` (minimal, 27 lines)

```python
def PatchDPOTrainer():
    return

def PatchKTOTrainer():
    return
```

### Real KTO Integration
- KTO trainer is accessed via TRL's `kto_trainer.KTOTrainer`
- Reference in `/unsloth/tokenizer_utils.py`: `"kto_trainer.KTOTrainer"`
- Unsloth notebook available: KTO notebook: https://colab.research.google.com/drive/1MRgGtLWuZX4ypSfGguFgC-IblTvO2ivM?usp=sharing

### How KTO Works with Unsloth
1. Load model with `FastLanguageModel.from_pretrained()` with 4-bit quantization
2. Apply LoRA with `FastLanguageModel.get_peft_model()`
3. Use TRL's KTOTrainer with UnslothTrainingArguments
4. Leverage Unsloth's memory optimizations automatically

### Key Observation
Unsloth doesn't implement KTO directly but optimizes the underlying infrastructure (quantization, LoRA kernels, gradient checkpointing) that KTO training depends on, resulting in the 70% VRAM savings.

---

## 3. Key Optimizations & Memory-Saving Techniques

### A. Kernel Optimizations (Triton-based)

#### Fast LoRA Kernels (`fast_lora.py`)
- Custom Triton kernels for efficient LoRA computations
- Fuses operations to reduce memory bandwidth
- Applied automatically when using `FastLanguageModel.get_peft_model()`

#### Cross-Entropy Loss Optimization
- **Cut Cross-Entropy**: Collaboration with Apple
- Reduces memory by computing only necessary logits for loss
- Achieves 89K context for Llama 3.3 (70B) on 80GB GPU
- File: `/unsloth/kernels/cross_entropy_loss.py`
- **HAS_CUT_CROSS_ENTROPY** flag controls availability

#### Attention Optimizations
- **Flex Attention** (`flex_attention.py`): Long-context support with reduced memory
- **Flash Attention 2 integration**: Automatic detection and usage
- Supports soft-capping in newer versions
- Custom SDPA (Scaled Dot-Product Attention) implementations

#### FP8 Quantization (`fp8.py`)
- Support for FP8 KV cache (`float8_kv_cache=True`)
- Reduces KV cache memory overhead
- Preference order: fbgemm > torchao > triton kernels

### B. Quantization Strategies

#### 4-Bit QLoRA (Default)
```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_use_double_quant = True,      # Nested quantization
    bnb_4bit_quant_type = "nf4",          # NF4 (most efficient)
    bnb_4bit_compute_dtype = dtype,       # Compute in higher precision
    llm_int8_skip_modules = [...]         # Skip quantization for specific layers
)
```

#### 8-Bit Quantization
- Enabled with `load_in_8bit=True`
- Higher accuracy than 4-bit with modest memory increase
- `llm_int8_skip_modules` for normalization layers

#### Full Fine-Tuning (FFT)
- Enabled with `full_finetuning=True`
- 16-bit training without quantization
- Best accuracy but requires more VRAM

#### Quantization-Aware Training (QAT)
- `qat_scheme` parameter in `from_pretrained()`
- Recovers up to 70% accuracy loss from quantization
- Uses PyTorch collab optimizations

### C. Gradient Checkpointing

#### Smart Gradient Checkpointing ("unsloth")
- Location: `unsloth_zoo.gradient_checkpointing`
- More efficient than standard HF implementation
- Controlled with `use_gradient_checkpointing="unsloth"` in `get_peft_model()`
- Options: `"unsloth"`, `False`, or reentrant gradient checkpointing

#### Offloaded Gradient Checkpointing
- Classes: `Unsloth_Offloaded_Gradient_Checkpointer`
- `unsloth_offloaded_gradient_checkpoint()`
- Offloads to disk for extreme memory constraints

### D. Memory Management

#### Dynamic Cache Management
- KV_CACHE_INCREMENT = 512 (tokens per cache update)
- Reduces memory fragmentation
- Automatically managed during training/inference

#### Embedding Offloading
- `offload_embedding=True`: Offload input/output embeddings
- Functions: `offload_input_embeddings()`, `offload_output_embeddings()`
- Useful for models with large vocabularies

#### Gradient Accumulation Fix
- Fixed bug in transformers 4.45.2 and below
- Unsloth applies `patch_gradient_accumulation_fix()`
- Critical for correctness with gradient accumulation

### E. Compiler Optimizations

#### Torch Compile Support
```python
torch_compile_options = {
    "epilogue_fusion": True,
    "max_autotune": False,          # Disable Triton mm kernels
    "shape_padding": True,
    "trace.enabled": False,
    "triton.cudagraphs": False,
}
```

#### Model Compilation
- Automatic selective compilation via `unsloth_compile_transformers()`
- Skips compilation for problematic architectures
- Dramatically improves throughput on newer GPUs

### F. Mixed Precision & Data Types

#### Automatic BFloat16 Detection
```python
if torch.cuda.is_bf16_supported():
    dtype = torch.bfloat16
else:
    dtype = torch.float16
```

#### Float32 Mixed Precision
- Force float32 for specific models (Gemma3, GPT-OSS)
- Parameter: `float32_mixed_precision`
- Prevents numerical instability

---

## 4. Configuration for RTX 3090 (24GB VRAM)

### RTX 3090 Specifications
- **VRAM**: 24GB
- **Compute Capability**: 8.6 (Ampere generation)
- **Supports**: BFloat16, Flash Attention 2, Tensor Cores
- **Optimal Config**: 4-bit QLoRA with gradient checkpointing

### Recommended Configuration Profile

#### A. Small Models (1B-3B)
```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-1B-Instruct",
    max_seq_length=2048,
    dtype=torch.bfloat16,
    load_in_4bit=True,
    device_map="sequential",
)

model = FastLanguageModel.get_peft_model(
    model,
    r=64,                                    # Higher rank for small models
    lora_alpha=64,                           # lora_alpha = r for stable learning
    target_modules="all-linear",
    lora_dropout=0.0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

training_args = SFTConfig(
    per_device_train_batch_size=4,           # Can increase with larger seqs
    gradient_accumulation_steps=2,
    max_seq_length=2048,
    learning_rate=2e-4,
    num_train_epochs=1,
    bf16=True,
    optim="adamw_8bit",                      # 8-bit optimizer saves 75% memory
    save_strategy="steps",
    logging_steps=10,
)
```

**Expected Memory**: 8-10GB VRAM

#### B. Medium Models (7B-13B)
```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Mistral-7B-v0.3",
    max_seq_length=1024,                     # Reduced for 24GB
    dtype=torch.bfloat16,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=32,                                    # Moderate rank
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                    "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth",
    lora_dropout=0.05,
)

training_args = SFTConfig(
    per_device_train_batch_size=2,           # Batch size 2 is safer
    gradient_accumulation_steps=4,           # Accumulate over 4 steps
    max_seq_length=1024,                     # Key: reduce max_seq
    learning_rate=2e-4,
    warmup_ratio=0.1,
    bf16=True,
    optim="adamw_8bit",
    gradient_checkpointing=True,             # Also enable in training args
)
```

**Expected Memory**: 18-22GB VRAM

#### C. Large Models (20B+)
**Status**: NOT recommended for RTX 3090 in standard configurations
- **Option 1**: Use much smaller max_seq_length (256)
- **Option 2**: Use 8-bit instead of 4-bit
- **Option 3**: Use full flash attention offloading

```python
# Example: 20B model with aggressive memory savings
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-20B-4bit",
    max_seq_length=256,                      # Aggressive reduction
    dtype=torch.bfloat16,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,                                    # Lower rank
    target_modules="all-linear",
    use_gradient_checkpointing="unsloth",
    loftq_config={},                         # LoftQ initialization
)

training_args = SFTConfig(
    per_device_train_batch_size=1,           # Batch size 1
    gradient_accumulation_steps=8,
    max_seq_length=256,
    learning_rate=2e-4,
    bf16=True,
    optim="adamw_8bit",
)
```

**Expected Memory**: 21-23GB VRAM

### Critical Parameters for RTX 3090

| Parameter | Small (1-3B) | Medium (7-13B) | Large (20B+) |
|-----------|-------------|----------------|------------|
| `max_seq_length` | 2048 | 1024 | 256 |
| `per_device_train_batch_size` | 4 | 2 | 1 |
| `gradient_accumulation_steps` | 1-2 | 4 | 8 |
| `r` (LoRA rank) | 64 | 32 | 16 |
| `load_in_4bit` | True | True | True |
| `use_gradient_checkpointing` | "unsloth" | "unsloth" | "unsloth" |
| `optim` | adamw_8bit | adamw_8bit | adamw_8bit |
| Estimated VRAM | 8-10GB | 18-22GB | 21-23GB |

### Memory Optimization Checklist

- [x] Use `load_in_4bit=True` (saves 75% vs 16-bit)
- [x] Use `optim="adamw_8bit"` (saves 75% of optimizer state)
- [x] Use `use_gradient_checkpointing="unsloth"` (saves 20-30% activations)
- [x] Set `per_device_train_batch_size=2-4` (depending on sequence length)
- [x] Use `bf16=True` (not fp16, for numerical stability)
- [x] Reduce `max_seq_length` if needed (quadratic with attention)
- [x] Use `gradient_accumulation_steps` > 1 (simulates larger batch size)
- [x] Set `lora_dropout=0.0` for training (no dropout in inference)

---

## 5. Batch Size, Sequence Length & Optimization Parameters

### Batch Size Recommendations

#### Based on Test Suite Evidence
- **Small experiments**: `per_device_train_batch_size=2-5`
- **Standard training**: `per_device_train_batch_size=2-4`
- **RL training (GRPO)**: `per_device_train_batch_size=1-2` (more memory intensive)

From test file `/unsloth/tests/qlora/test_unsloth_qlora_train_and_merge.py`:
```python
batch_size = 5  # For 1B model with 512 sequence
```

From test file `/unsloth/tests/saving/language_models/test_merge_model_perplexity_mistral.py`:
```python
per_device_train_batch_size = 2
gradient_accumulation_steps = 4  # Effective batch = 8
```

### Sequence Length Guidelines

#### From Test Suite
```python
max_seq_length = 2048  # Default for text models
max_seq_length = 512   # For 1B-3B models
max_seq_length = 1024  # Balanced for 7B models
```

#### Memory Formula (Approximate)
```
VRAM ≈ 3.5 * (batch_size * max_seq_length) * num_params_billions / 8
       (for 4-bit QLoRA with gradient checkpointing)
```

**For RTX 3090 (24GB)**:
- 7B model, batch=2, seq=1024: ~6GB
- 7B model, batch=4, seq=512: ~6GB
- 13B model, batch=2, seq=512: ~8GB

### Gradient Accumulation Best Practices
- **Increase accumulation steps** rather than batch size to save memory
- Effective batch = `per_device_train_batch_size * gradient_accumulation_steps * num_gpus`
- Example: batch=2, accumulation=4 = effective batch of 8 with lower memory

### Learning Rate Configuration
- **Standard**: `learning_rate=2e-4` (from multiple tests)
- **With warmup**: `warmup_ratio=0.1` (warm up over 10% of training)
- **Scheduler**: `lr_scheduler_type="linear"`

### Optimizer Configuration
```python
optim = "adamw_8bit"        # Critical for memory savings (75% less than fp32)
weight_decay = 0.01         # L2 regularization
beta1 = 0.9, beta2 = 0.999  # Default Adam parameters
eps = 1e-8                  # Numerical stability
```

### LoRA Configuration Best Practices

From test implementations:

#### Default Configuration
```python
FastLanguageModel.get_peft_model(
    model,
    r = 16,                          # LoRA rank (default)
    lora_alpha = 16,                 # = r for uniform scaling (scale = alpha/r)
    lora_dropout = 0.0,              # No dropout during training
    bias = "none",                   # Don't train bias
    target_modules = [               # All attention + FFN projections
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    use_gradient_checkpointing = "unsloth",  # Memory efficient
    random_state = 3407,             # Seed for reproducibility
    use_rslora = False,              # Rank-Stabilized LoRA (optional)
)
```

#### Rank Selection Strategy
- **Small models (1-3B)**: r=64 (larger models can handle higher ranks)
- **Medium models (7-13B)**: r=32
- **Large models (20B+)**: r=16
- **Rule of thumb**: Effective dimension should be 1-2% of hidden dim

#### Target Modules Optimization
- **"all-linear"**: Apply LoRA to ALL linear layers (comprehensive but more memory)
- **Attention only**: Just Q, K, V, O projections (memory efficient)
- **With FFN**: Add gate_proj, up_proj, down_proj (recommended)

---

## 6. Model Loading, Quantization & LoRA Configuration

### FastLanguageModel.from_pretrained() Complete API

```python
FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-1B-Instruct",  # HF model ID or local path
    
    # Sequence & Data Types
    max_seq_length = 2048,                         # Context window size
    dtype = torch.bfloat16,                        # None=auto, float16, bfloat16, float32
    
    # Quantization Strategy (mutually exclusive)
    load_in_4bit = True,                           # 4-bit QLoRA (default, most memory efficient)
    load_in_8bit = False,                          # 8-bit LoRA (higher accuracy)
    load_in_16bit = False,                         # 16-bit LoRA (legacy)
    full_finetuning = False,                       # Full fine-tuning (no LoRA)
    
    # Device & Distribution
    device_map = "sequential",                     # "auto", "sequential", or custom
    offload_embedding = False,                     # Offload embeddings to RAM
    
    # Inference & vLLM
    fast_inference = False,                        # Use vLLM for inference
    gpu_memory_utilization = 0.5,                  # vLLM memory allocation
    float8_kv_cache = False,                       # Use FP8 for KV cache
    
    # RoPE Scaling (long context)
    rope_scaling = None,                           # "linear", "dynamic", or None
    
    # Tokenizer
    fix_tokenizer = True,                          # Fix tokenizer issues
    trust_remote_code = False,                     # Load custom model code
    token = None,                                  # HF API token
    
    # LoRA Configuration
    max_lora_rank = 64,                            # Max LoRA rank allowed
    random_state = 3407,                           # Seed for reproducibility
    
    # Advanced
    qat_scheme = None,                             # Quantization-aware training
    float32_mixed_precision = None,                # Force float32 precision
    revision = None,                               # Model revision/branch
    use_exact_model_name = False,                  # Don't convert model names
    disable_log_stats = True,                      # Suppress logging
)
```

### Quantization Decision Tree

```
Is memory critical?
├─ YES → load_in_4bit=True (75% memory savings)
│         ├─ High accuracy needed? → Use loftq_config
│         └─ Normal training → Default NF4 quantization
└─ NO → full_finetuning=True (best accuracy)
        └─ Medium memory budget → load_in_8bit=True
```

### LoRA Configuration API

```python
FastLanguageModel.get_peft_model(
    model = model_loaded,
    
    # Rank & Scaling
    r = 16,                                    # LoRA rank (1-64 typical)
    lora_alpha = 16,                           # Scaling = alpha/r (default: =r)
    lora_dropout = 0.0,                        # Dropout rate (0 recommended)
    
    # Target Modules
    target_modules = [                         # Modules to apply LoRA
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    
    # Bias & Weight Init
    bias = "none",                             # "none", "all", or "lora_only"
    modules_to_save = None,                    # Modules to save completely (not LoRA)
    init_lora_weights = True,                  # Initialize LoRA weights
    
    # Memory & Checkpointing
    use_gradient_checkpointing = "unsloth",   # "unsloth", False, or "reentrant"
    
    # Advanced
    use_rslora = False,                        # Rank-Stabilized LoRA
    random_state = 3407,                       # RNG seed
    max_seq_length = 2048,                     # Not used (legacy)
    
    # QAT (Quantization-Aware Training)
    qat_scheme = None,                         # QAT configuration
    
    # LoftQ Configuration
    loftq_config = {},                         # LoftQ initialization for quantized models
)
```

### Practical Loading Examples

#### Example 1: RTX 3090 - Llama 7B
```python
from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "meta-llama/Llama-2-7b-chat-hf",
    max_seq_length = 1024,
    dtype = torch.bfloat16,
    load_in_4bit = True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 32,
    lora_alpha = 32,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing = "unsloth",
)
```

#### Example 2: Maximum Accuracy
```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "meta-llama/Llama-3.1-8B-Instruct",
    max_seq_length = 2048,
    dtype = torch.bfloat16,
    load_in_8bit = True,  # Higher accuracy than 4-bit
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 64,
    lora_alpha = 64,
    target_modules = "all-linear",  # Apply to all linear layers
)
```

#### Example 3: With Quantization-Aware Training
```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Mistral-7B-v0.3",
    max_seq_length = 512,
    dtype = torch.bfloat16,
    load_in_4bit = True,
    qat_scheme = "w4_a16_per_token_dynamic",  # Recover 70% accuracy
)
```

---

## 7. Specific Issues & Limitations

### Known Limitations

#### 1. Multi-GPU Training
- **Status**: Single-GPU only in main branch
- **Note**: "Unsloth currently does not work on multi GPU setups"
- **Future**: Beta multi-GPU version available on request
- **Workaround**: Use `device_map="sequential"` for large models on single GPU

#### 2. Windows Support
- **Status**: Supported via Triton Windows fork
- **Requirements**: PyTorch >= 2.4, CUDA 12
- **Config**: Set `dataset_num_proc=1` to avoid crashes
- **Installation**: Manual setup required (not `pip install`)

#### 3. AMD GPU Limitations
- **4-bit quantization**: Not stable on AMD GPUs
- **Workaround**: Use 8-bit or 16-bit training instead
- **Block size**: AMD requires block size 128 (vs 64 in pre-quants)

#### 4. Model Architecture Limitations

**Models that skip compilation**:
- aya_vision
- modernbert
- granite + llava_next

**Models with disabled SDPA**:
- gemma3
- gemma3n

**Force Float32**:
- gemma3
- gpt_oss

#### 5. Import Order Critical
From `/unsloth/__init__.py` warning:
```
"WARNING: Unsloth should be imported before 'trl', 'transformers', 'peft'
to ensure all optimizations are applied. Your code may run slower or 
encounter memory issues without these optimizations."
```

**Correct order**:
```python
import unsloth  # FIRST
from transformers import ...
from trl import ...
from peft import ...
```

#### 6. KV Cache & Long Context
- **KV_CACHE_INCREMENT**: 512 tokens (hardcoded)
- **For long sequences**: Use flex attention (`flex_attention.py`)
- **Max context**: Depends on model architecture and VRAM

#### 7. Gradient Accumulation Bug
- **Fixed in**: transformers >= 4.45.2+
- **Issue**: Incorrect gradient scaling
- **Recommendation**: Update transformers: `pip install --upgrade transformers`

### Performance Considerations

#### When to Use 4-bit vs 8-bit
| Scenario | Recommendation | Reason |
|----------|----------------|--------|
| Maximum memory savings | 4-bit QLoRA | 75% VRAM reduction |
| High accuracy needed | 8-bit | ~5% better accuracy |
| VRAM < 8GB | 4-bit + aggressive checkpointing | Only option |
| VRAM > 16GB | 8-bit or 16-bit | Minimal memory pressure |
| Production deployment | 16-bit merged | Numerical precision |

#### Batch Size vs Gradient Accumulation
```
For same effective batch size:
Memory usage: gradient_accumulation < per_device_batch_size
Speed: per_device_batch_size > gradient_accumulation

RTX 3090 sweet spot: batch=2, accumulation=4 (effective batch=8)
```

### Debugging Tips

1. **Out of Memory (OOM) errors**:
   - Reduce `max_seq_length`
   - Reduce `per_device_train_batch_size`
   - Increase `gradient_accumulation_steps`
   - Enable gradient checkpointing
   - Try `load_in_4bit=True`

2. **Slow training**:
   - Check if all optimizations applied (import order)
   - Verify `use_gradient_checkpointing="unsloth"`
   - Ensure `bf16=True` on supported GPUs
   - Use `torch.compile` if available

3. **Accuracy degradation from 4-bit**:
   - Use `qat_scheme` parameter
   - Try `loftq_config={}` for initialization
   - Increase LoRA rank
   - Use 8-bit instead

---

## 8. Training Script Template (RTX 3090)

```python
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import torch

# Model loading
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "meta-llama/Llama-3.2-7B-Instruct",
    max_seq_length = 1024,
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    load_in_4bit = True,
)

# LoRA configuration
model = FastLanguageModel.get_peft_model(
    model,
    r = 32,
    lora_alpha = 32,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_dropout = 0.05,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 42,
)

# Data preparation
dataset = load_dataset("json", data_files="data.jsonl", split="train")

def formatting_func(example):
    return {
        "text": f"{example['instruction']}\n{example['response']}"
    }

dataset = dataset.map(formatting_func, batched=True)

# Training
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = 1024,
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_ratio = 0.1,
        num_train_epochs = 1,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 10,
        optim = "adamw_8bit",
        output_dir = "outputs",
        seed = 42,
    ),
)

trainer.train()

# Save
model.save_pretrained_merged("final_model", tokenizer=tokenizer)
```

---

## 9. Registry & Model Support

### Supported Models (via registry)
- **Llama**: 3.1, 3.2, 4
- **Mistral**: 7B, v0.3
- **Qwen**: 2, 2.5, 3, 3-MOE
- **Phi**: 3, 3.5, 4
- **Gemma**: 2, 3, 3n
- **DeepSeek**: Various versions
- **Granite**, **Cohere**, **Falcon** (H1)

Registry location: `/unsloth/unsloth/registry/`

### Adding Custom Models
Models not in registry still work via `FastLanguageModel.from_pretrained()` if compatible with transformers `AutoModelForCausalLM`.

---

## 10. Key Files Reference

| File | Purpose | Lines | Key Components |
|------|---------|-------|-----------------|
| `loader.py` | Model loading API | ~800 | FastLanguageModel.from_pretrained(), FastLanguageModel.get_peft_model() |
| `llama.py` | LLaMA optimizations | ~3800 | Core optimizations, attention, quantization |
| `trainer.py` | Training utilities | ~241 | UnslothTrainer, UnslothTrainingArguments |
| `save.py` | Saving/merging | ~3500 | save_pretrained_merged(), merge logic |
| `_utils.py` | Memory & patches | ~2500 | Gradient checkpointing, torch compile, quantization patches |
| `rl.py` | RL training | ~2000 | GRPO, GSPO, RL optimizations |
| `kernels/` | Triton kernels | ~5000+ | fast_lora, cross_entropy, attention, moe |

---

## Conclusion

Unsloth achieves its 2x faster training with 70% less VRAM through:

1. **Kernel-level optimizations** (Triton-based fast LoRA, fused operations)
2. **Strategic quantization** (4-bit QLoRA with double-quantization, NF4)
3. **Memory-efficient training** (smart gradient checkpointing, 8-bit optimizers)
4. **Compiler optimizations** (torch.compile with custom heuristics)
5. **Infrastructure integration** (seamless with transformers/peft/trl)

For RTX 3090 (24GB), optimal configs:
- **7B models**: batch=2, seq=1024, gradient_accumulation=4
- **13B models**: batch=1, seq=512, gradient_accumulation=8
- **20B+ models**: Challenging, requires max aggressive optimization

KTO training works through TRL integration with Unsloth's optimized backbone, resulting in significant VRAM savings automatically.
