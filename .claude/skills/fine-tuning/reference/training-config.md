# Training Config Reference

Full YAML configuration reference for all training methods.

---

## SFT Config (`Trainers/rtx3090_sft/configs/config.yaml`)

### Model Section
```yaml
model:
  model_name: "unsloth/Qwen3-1.7B-unsloth-bnb-4bit"  # Base model
  max_seq_length: 2048   # Context window
  dtype: null            # Auto-detection (BF16 on RTX 3090)
  load_in_4bit: true     # Essential for 24GB VRAM
```

### LoRA Section
```yaml
lora:
  r: 64                  # Rank (3B:32, 7B:64, 13B:128)
  lora_alpha: 128        # Scaling factor (2x rank)
  lora_dropout: 0.05     # Regularization
  bias: "none"
  target_modules:        # All attention + MLP projections
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
  use_gradient_checkpointing: "unsloth"  # Unsloth-optimized (saves memory)
  random_state: 3407     # Reproducibility
```

### Training Section
```yaml
training:
  output_dir: "./sft_output_rtx3090"
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 2    # Effective batch = 4
  learning_rate: 2e-4
  max_grad_norm: 1.0                # Gradient clipping
  lr_scheduler_type: "cosine"

  # SFT-specific
  max_seq_length: 2048
  packing: true                     # 2.5-5x faster!
  completion_only_loss: true        # Train only on assistant responses

  # Memory optimizations
  gradient_checkpointing: true
  optim: "adamw_8bit"               # Saves ~2GB VRAM
  fp16: false
  bf16: true                        # RTX 3090 supports BF16

  # Schedule
  num_train_epochs: 2
  warmup_ratio: 0.1

  # Logging & checkpoints
  logging_steps: 5
  save_steps: 50
  save_total_limit: 3               # Keep last 3 checkpoints

  # Performance
  dataloader_num_workers: 0         # MUST be 0 on WSL2
  dataloader_pin_memory: true
  group_by_length: false            # Can hang with multiprocessing
```

### Dataset Section
```yaml
dataset:
  dataset_name: "professorsynapse/nexus-synthetic-dataset"  # HF dataset
  dataset_file: "nonthinking_tools_sft_12.28.25.jsonl"      # File within
  local_file: "../../Datasets/my_data.jsonl"                 # Local override
  num_proc: 1                       # Must be 1 on Windows/WSL
  test_size: 0.1                    # Validation split ratio
  split_dataset: false
  filter_desirable: false           # SFT doesn't need filtering
```

### Evolutionary Section (Experimental)
```yaml
evolutionary:
  enabled: false
  candidates: 4
  eval_batch_size: 2
  validation_config: "configs/fitness/tool_calling.yaml"
  strategy:
    type: "gradient_noise"
    params:
      noise_scale: 0.1
  selection:
    method: "best"
    min_improvement: 0.0
  eval_frequency: 1
  warmup_steps: 0
  cache_baseline: true
```

---

## KTO Config (`Trainers/rtx3090_kto/configs/config.yaml`)

**Differences from SFT highlighted:**

```yaml
model:
  model_name: professorsynapse/nexus-tools_sft24  # Usually starts from SFT model
  max_seq_length: 2048
  load_in_4bit: true

training:
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 4    # Higher than SFT (more stable)
  learning_rate: 1e-6               # 100x lower than SFT!

  # KTO-specific
  beta: 0.1                         # KL divergence penalty
  desirable_weight: 1.0             # Weight for True examples
  undesirable_weight: 1.0           # Weight for False examples
  use_kto_s: false                  # KTO-S variant (stable for base models)

  # Two-stage LR (optional)
  use_two_stage_lr: false
  lr_reduction_step: 50
  lr_reduction_factor: 0.5

  # Context lengths
  max_length: 2048
  max_prompt_length: 1024           # KTO splits prompt/completion

  # Schedule
  num_train_epochs: 1               # Only 1 epoch for KTO
  warmup_ratio: 0.15
  lr_scheduler_type: cosine

  # Memory
  optim: adamw_8bit
  gradient_checkpointing: true
  bf16: true

  # Checkpoints
  logging_steps: 5
  save_steps: 25                    # More frequent than SFT
  save_total_limit: 3

dataset:
  local_file: ../../Datasets/behavior_merged_kto_v1.5_balanced.jsonl
  num_proc: 1
  test_size: 0.1
```

---

## GRPO Config (`Trainers/rtx3090_grpo/configs/config.yaml`)

```yaml
model:
  model_name: "unsloth/Qwen3-1.7B-unsloth-bnb-4bit"
  lora_path: null                   # Optional: path to SFT checkpoint

training:
  per_device_train_batch_size: 6
  gradient_accumulation_steps: 6
  num_generations: 4                # Completions per prompt
  max_prompt_length: 1024
  max_completion_length: 512
  temperature: 1.0                  # Sampling temperature
  learning_rate: 5e-6
  beta: 0.1                         # KL penalty (higher than TRL default 0.04)
  use_gspo: false                   # Toggle GSPO mode
  num_train_epochs: 1

dataset:
  local_file: "../../Datasets/grpo_data.jsonl"
  prompt_column: "prompt"
```

---

## Settings That Should NOT Be Changed

| Setting | Value | Why |
|---------|-------|-----|
| `dataloader_num_workers` | 0 | WSL2 crashes with >0 |
| `load_in_4bit` | true* | Required for 24GB VRAM with standard Transformers |
| `optim` | adamw_8bit | OOM without it |
| `use_gradient_checkpointing` | "unsloth" | Must use Unsloth variant |
| `num_proc` (dataset) | 1 | Windows/WSL compatibility |

> **\* Exception — Hybrid/SSM architectures:** `load_in_4bit` must be `false` for models like LFM2.5 that use non-standard layers (LIV convolution blocks) incompatible with bnb-4bit quantization. See [Architecture-Specific Overrides](#architecture-specific-config-overrides) below.

---

## Architecture-Specific Config Overrides

Standard config defaults work for Llama, Qwen, Mistral, Gemma, and most Transformer models. These architectures require different settings:

### LiquidAI LFM2.5 (Hybrid SSM — LIV + GQA)

> ⚠️ **Using standard defaults with LFM2.5 causes SIGABRT (exit code -6) crash at ~step 5.** The crash is caused by bnb-4bit quantization being applied to LIV convolution blocks. It presents as a cryptic PyTorch teardown error but is a config issue.

```yaml
model:
  model_name: "LiquidAI/LFM2.5-1.2B-Instruct"
  max_seq_length: 4096
  dtype: null
  load_in_4bit: false        # REQUIRED — LIV blocks incompatible with bnb-4bit

lora:
  r: 16
  lora_alpha: 16
  lora_dropout: 0            # Use 0, not 0.05
  bias: "none"
  target_modules:            # LFM2.5 uses different layer names than standard Transformers
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "out_proj"             # NOT "o_proj"
    - "in_proj"              # No equivalent in standard Transformers
    - "w1"                   # NOT "gate_proj"
    - "w2"                   # NOT "up_proj"
    - "w3"                   # NOT "down_proj"
  use_gradient_checkpointing: "unsloth"

training:
  per_device_train_batch_size: 2    # Use 2, not 4
  lr_scheduler_type: "linear"       # NOT "cosine"
  warmup_ratio: 0.02                # ~5 warmup steps out of ~233
  # All other training settings remain the same
```

**Quick checklist before training LFM2.5:**
- [ ] `load_in_4bit: false`
- [ ] `target_modules` uses `out_proj`, `in_proj`, `w1`, `w2`, `w3`
- [ ] `r: 16`, `lora_alpha: 16`, `lora_dropout: 0`
- [ ] `max_seq_length: 4096`
