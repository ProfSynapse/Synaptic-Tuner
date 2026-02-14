---
name: upload-deployment
description: Complete reference for model upload and deployment. Covers HuggingFace upload, save strategies (LoRA, merged 16-bit, merged 4-bit), GGUF conversion, model merging, model cards, and the full upload workflow. Use when uploading models, creating GGUF files, merging LoRA adapters, or deploying to HuggingFace. This skill is about USING the upload/deployment tools via CLI — never modifying source code.
allowed-tools: Read, Bash, Write, Grep, Glob
---

# Upload & Deployment

Upload trained models to HuggingFace with optional GGUF conversion and model card generation.

## Quick Reference

| Task | Command |
|------|---------|
| Interactive menu | `./run.sh` → Upload |
| Upload merged 16-bit | `python src/upload_to_hf.py MODEL_PATH user/repo --save-method merged_16bit` |
| Upload with GGUF | `python src/upload_to_hf.py MODEL_PATH user/repo --save-method merged_16bit --create-gguf` |
| Upload LoRA only | `python src/upload_to_hf.py MODEL_PATH user/repo --save-method lora_only` |
| Merge LoRA manually | `./run.sh` → Merge LoRA |
| Convert to GGUF only | `./run.sh` → Convert |
| Full pipeline | `./run.sh` → Full Pipeline (Train → Upload → Eval) |

## Save Strategies

| Strategy | Size (7B) | GPU Required | Best For |
|----------|-----------|--------------|----------|
| `lora_only` | ~100-500 MB | No | Sharing adapters, fast upload |
| `merged_16bit` | ~14 GB | Yes | Production inference, GGUF source |
| `merged_4bit` | ~4 GB | Yes | Smaller footprint, slight quality loss |

## GGUF Quantizations

| Format | Size (7B) | Quality | Use Case |
|--------|-----------|---------|----------|
| Q8_0 | ~7 GB | Highest | Best quality, more RAM |
| Q5_K_M | ~5 GB | High | Good balance |
| Q4_K_M | ~4 GB | Good | Most popular, efficient |

## Key Directories

- `Trainers/rtx3090_sft/src/upload_to_hf.py` — SFT upload entry point
- `Trainers/rtx3090_kto/src/upload_to_hf.py` — KTO upload entry point
- `shared/upload/` — Upload orchestrator and strategies
- `shared/upload/converters/` — GGUF and WebGPU converters
- `shared/model_loading/` — Model loading and LoRA merge utilities

## Progressive Reference

Load the specific reference you need:

| Reference | When to Load | Path |
|-----------|-------------|------|
| **Upload Workflow** | Uploading to HuggingFace, full process | `reference/upload-workflow.md` |
| **GGUF Conversion** | Creating GGUF files, quantization options | `reference/gguf-conversion.md` |
| **Model Merging** | Merging LoRA into base, preparing for GRPO | `reference/model-merging.md` |
| **Model Cards** | Documentation, lineage, manifests | `reference/model-cards.md` |

## Common Patterns

**Standard upload after SFT:**
```bash
cd Trainers/rtx3090_sft
python src/upload_to_hf.py \
  ./sft_output_rtx3090/TIMESTAMP/final_model \
  username/model-name \
  --save-method merged_16bit \
  --create-gguf
```

**Merge LoRA for GRPO continuation:**
```bash
# Use shared merge utility
./run.sh → Merge LoRA
# Or the GRPO trainer auto-merges when lora_path is set in config
```

**Upload with evaluation results:**
```bash
# Evaluate first
python -m Evaluator.cli --backend unsloth --model path/to/model \
  --lineage eval_lineage.json --upload-to-hf user/model --update-model-card
```

## Output Structure

After upload, HuggingFace repo contains:
```
username/model-name/
├── lora/                      # LoRA adapters (if lora_only)
├── merged-16bit/              # Full model (if merged_16bit)
├── gguf/                      # GGUF quantizations (if --create-gguf)
│   ├── model-Q4_K_M.gguf
│   ├── model-Q5_K_M.gguf
│   ├── model-Q8_0.gguf
│   └── model-mmproj.gguf     # Vision projector (VL models only)
├── upload_manifest.json       # Upload metadata
├── training_lineage.json      # Training provenance
└── README.md                  # Auto-generated model card
```

## Environment Variables

```bash
HF_TOKEN=hf_...                       # Required for uploads
```

## Tips

- Always use `merged_16bit` as the source for GGUF conversion (best quality)
- The reliable GGUF converter merges LoRA once, then creates all quants (~10 min saved)
- Vision-language models auto-get an `mmproj.gguf` for the vision projector
- On WSL, temp files use native filesystem to avoid NTFS performance issues
- `training_lineage.json` is auto-generated — includes model, LoRA, dataset, hardware info
- Use `upload_manifest.json` to verify what was uploaded
- The upload orchestrator handles everything — prefer `./run.sh` → Upload over manual commands
