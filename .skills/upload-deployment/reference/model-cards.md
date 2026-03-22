# Model Cards Reference

Auto-generated documentation for uploaded models.

---

## What Gets Generated

The upload process auto-generates three documentation files:

### 1. README.md (Model Card)

Auto-generated HuggingFace model card containing:
- Model description
- Training method (SFT/KTO/GRPO)
- Base model info
- LoRA configuration
- Training hyperparameters
- Dataset info
- Available formats (merged, GGUF quantizations)
- Usage instructions
- Evaluation results (if `--update-model-card` used)

### 2. upload_manifest.json

Machine-readable upload metadata:
```json
{
  "repo_id": "username/model-name",
  "upload_date": "2025-02-14T10:30:00Z",
  "save_method": "merged_16bit",
  "formats": ["merged-16bit", "gguf-Q4_K_M", "gguf-Q5_K_M", "gguf-Q8_0"],
  "files_uploaded": ["model.safetensors", "..."],
  "base_model": "unsloth/Qwen2.5-7B-Instruct",
  "training_method": "sft"
}
```

### 3. training_lineage.json

Complete training provenance:
```json
{
  "model": {
    "name": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    "max_seq_length": 2048,
    "load_in_4bit": true
  },
  "lora": {
    "r": 64,
    "lora_alpha": 128,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
  },
  "training": {
    "method": "sft",
    "learning_rate": 0.0002,
    "batch_size": 2,
    "gradient_accumulation": 2,
    "epochs": 3,
    "max_steps": null,
    "optimizer": "adamw_8bit",
    "scheduler": "cosine"
  },
  "dataset": {
    "name": "professorsynapse/nexus-synthetic-dataset",
    "file": "tools_sft.jsonl",
    "size": 2676
  },
  "hardware": {
    "gpu": "NVIDIA RTX 3090",
    "cuda_version": "12.1",
    "vram_gb": 24
  },
  "results": {
    "total_steps": 336,
    "final_loss": 0.42,
    "training_time_minutes": 45
  }
}
```

---

## Adding Evaluation Results to Model Card

After evaluation:

```bash
python -m Evaluator.cli \
  --backend unsloth \
  --model path/to/model \
  --scenario behavior_prompts.yaml \
  --scenario tool_prompts.yaml \
  --lineage eval_lineage.json \
  --upload-to-hf username/model-name \
  --update-model-card
```

This adds an "Evaluation Results" section to the README with:
- Pass rates (overall, schema, behavior)
- Per-tag breakdown
- Test scenarios used
- Evaluation date

---

## Lineage Chain

For models trained in sequence (SFT → KTO → GRPO), each step's `training_lineage.json` references the previous:

```
SFT lineage: base_model → SFT
KTO lineage: base_model → SFT → KTO
GRPO lineage: base_model → SFT → GRPO
```

This creates a complete training history for reproducibility.

---

## Verifying Uploads

After upload, check the manifest:

```bash
# Download and verify
python -c "
import json
manifest = json.load(open('upload_manifest.json'))
print(f'Repo: {manifest[\"repo_id\"]}')
print(f'Formats: {manifest[\"formats\"]}')
print(f'Files: {len(manifest[\"files_uploaded\"])}')
"
```

Or check on HuggingFace:
```bash
# List repo files
pip install huggingface_hub
python -c "
from huggingface_hub import list_repo_files
files = list_repo_files('username/model-name')
for f in files: print(f)
"
```
