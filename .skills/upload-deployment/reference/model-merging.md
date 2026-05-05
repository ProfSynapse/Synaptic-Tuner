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

### Via Matched Docker Runtime (Preferred For Local Docker Runs)

For adapters produced by local Docker training, merge inside the same configured
training runtime/container whenever possible. That runtime has the GPU,
Transformers, PEFT, Unsloth, and model-family support that produced the
adapter.

Host-side Python merge is a fallback. It often fails for avoidable reasons:
- CPU-only torch or no visible accelerator
- stale Transformers that cannot load the current model architecture
- PEFT/Unsloth versions that differ from the training job
- Windows PowerShell quoting issues around nested `docker exec bash -lc` and
  long inline `python -c` commands

On Windows, prefer executing a script file in the container instead of passing a
large inline command:

```bash
docker exec -w /workspace/repo <training-container> \
  python -u path/to/run-local-merge-launcher.py
```

The launcher should import existing repo utilities such as
`shared.model_loading.merge.merge_lora_checkpoint`; do not duplicate merge
logic in one-off scripts.

Some adapter auto-mapping loaders can save a merged checkpoint whose tensor
names no longer line up with the base checkpoint, even though the merge itself
appears to finish. Common symptoms are vLLM errors about weights not initialized
from the checkpoint, or saved keys with extra nested path segments compared to
the verified base model. Before treating this as an eval problem, compare the
base and merged safetensor key sets.

If the repo's merge helper supports it, use explicit post-save alignment flags
instead of hand-editing model files:

```bash
python scripts/merge_peft_adapter.py \
  --base path/to/merged-base \
  --adapter path/to/final_model \
  --output path/to/merged-output \
  --base-loader adapter-auto-mapping \
  --align-saved-keys-to-base \
  --copy-missing-base-keys
```

These options are generic repairs:
- `--align-saved-keys-to-base` only renames saved tensors when the repaired key
  exists in the base checkpoint.
- `--copy-missing-base-keys` copies unchanged base tensors that PEFT omitted
  from the merged save, such as auxiliary or multimodal modules.

After merge, verify that the merged key set matches the base key set before
running vLLM eval. Keep the repaired output in a new directory rather than
overwriting a suspect merge artifact.

### Via Interactive Menu

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

For local SFT runs, complete the merge and eval the merged model before GRPO.
Do not start GRPO from a host-merged artifact unless it was verified in a
runtime that supports the same model family.

The GRPO trainer handles this automatically when `lora_path` is set:

```yaml
# Trainers/grpo/configs/config.yaml
model:
  model_name: "unsloth/Qwen3-1.7B-unsloth-bnb-4bit"
  lora_path: "../sft/sft_output/TIMESTAMP/checkpoint-1150"
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
