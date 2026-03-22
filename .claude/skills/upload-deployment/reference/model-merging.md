# Model Merging Reference

Merging LoRA adapters into base models.

---

## Why Merge?

LoRA adapters are lightweight (~100-500 MB) but require the base model to run. Merging creates a standalone model.

**When to merge:**
- Before uploading (merged_16bit for HuggingFace)
- Before GGUF conversion (needs merged model)
- Before GRPO training (GRPO applies new LoRA on top of merged SFT)
- For standalone deployment

---

## Merge Methods

### Via Interactive Menu (Easiest)

```bash
./run.sh
# Select: Merge LoRA
# Choose training run
# Choose merge format (16-bit recommended)
```

### Via Upload (Automatic)

When using `--save-method merged_16bit`, merging happens automatically:

```bash
python3 scripts/upload_model.py ./final_model user/repo --save-method merged_16bit
```

### Via Shared Utilities

The merge utilities in `shared/model_loading/merge.py`:

```python
from shared.model_loading.merge import merge_lora_checkpoint, find_or_create_merged

# Direct merge
merged_path = merge_lora_checkpoint(
    lora_path="path/to/final_model",
    output_path="path/to/merged-16bit"
)

# Smart: find existing or create
merged_path = find_or_create_merged(
    model_path="path/to/final_model"
)

# Check if already merged
from shared.model_loading.merge import is_lora_checkpoint, is_merged_model
is_lora = is_lora_checkpoint("path/to/model")  # True if LoRA
is_merged = is_merged_model("path/to/model")    # True if already merged
```

---

## GRPO Merge Workflow

GRPO training requires a merged base to apply new LoRA adapters:

1. **Train SFT** → produces LoRA checkpoint
2. **Merge SFT LoRA** → creates merged-16bit model
3. **Train GRPO** → applies new LoRA on merged base

The GRPO trainer handles this automatically when `lora_path` is set:

```yaml
# Trainers/rtx3090_grpo/configs/config.yaml
model:
  model_name: "unsloth/Qwen3-1.7B-unsloth-bnb-4bit"
  lora_path: "../rtx3090_sft/sft_output_rtx3090/TIMESTAMP/checkpoint-1150"
```

The trainer:
1. Loads base model
2. Loads SFT LoRA
3. Merges into base weights
4. Applies fresh LoRA for GRPO training

---

## Merge Detection

The system auto-detects model type:

| Path Contains | Type | Action |
|---------------|------|--------|
| `adapter_config.json` | LoRA checkpoint | Needs merging |
| `model.safetensors` (large) | Merged model | Ready to use |
| `config.json` only | HuggingFace model | Ready to use |

---

## Output

Merged models are saved to:
```
training_run_dir/
├── final_model/              # LoRA adapters (original)
└── model-name/
    └── merged-16bit/         # Merged model (created by upload/merge)
```

---

## Memory Requirements

Merging requires loading the full model in memory:

| Model Size | RAM/VRAM Required |
|------------|-------------------|
| 3B | ~6 GB |
| 7B | ~14 GB |
| 13B | ~26 GB |
| 20B | ~40 GB |

For large models, ensure sufficient GPU memory or use CPU merging (slower but works with system RAM).
