---
name: upload-deployment
description: Complete reference for model upload and deployment. Covers HuggingFace upload, save strategies (LoRA, merged 16-bit, merged 4-bit), GGUF conversion, model merging, model cards, and the full upload workflow. Use when uploading models, creating GGUF files, merging LoRA adapters, or deploying to HuggingFace. This skill is about USING the upload/deployment tools via CLI — never modifying source code.
allowed-tools: Read, Bash, Write, Grep, Glob
---

# Upload & Deployment

Upload trained models to HuggingFace with optional GGUF conversion and model card generation.

For cloud training, provider-native storage remains the source of truth. Hugging Face Hub publishing is optional and only applies to `final_model`.

## Quick Reference

| Task | Command |
|------|---------|
| Interactive menu | `./run.sh` → Upload |
| Upload merged 16-bit | `python3 scripts/upload_model.py MODEL_PATH user/repo --save-method merged_16bit` |
| Upload with GGUF | `python3 scripts/upload_model.py MODEL_PATH user/repo --save-method merged_16bit --create-gguf` |
| Upload LoRA only | `python3 scripts/upload_model.py MODEL_PATH user/repo --save-method lora` |
| Merge LoRA manually | `./run.sh` → Merge LoRA |
| Convert to GGUF only | `./run.sh` → Convert |
| Cloud GGUF conversion | `python tuner.py cloud-run --job-config Trainers/cloud/jobs/gguf_conversion.yaml --yes` |
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

- `scripts/upload_model.py` — Generic upload entry point
- `scripts/cloud_gguf_convert.py` — Cloud GGUF conversion CLI (download → convert → upload)
- `Trainers/cloud/jobs/gguf_conversion.yaml` — HF Jobs config for cloud GGUF conversion
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
| **Local Mac GGUF Workflow** | Pull from HF bucket, merge locally on macOS, create GGUF, and place into LM Studio/Ollama | `reference/local-mac-bucket-to-gguf.md` |
| **Model Cards** | Documentation, lineage, manifests | `reference/model-cards.md` |
| **Cloud Training** | Provider-native storage, optional final-model publish, artifact discovery | `../fine-tuning/reference/cloud-training.md` |

## Common Patterns

**Standard upload after SFT:**
```bash
python3 scripts/upload_model.py \
  Trainers/sft/sft_output/TIMESTAMP/final_model \
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

**Cloud GGUF conversion (when local RAM is insufficient):**
```bash
# 1. Upload merged model to HF first (if not already there)
# 2. Edit env vars in the job YAML or override at runtime:
#    GGUF_MODEL_REPO: the HF repo with the merged model
#    GGUF_QUANT_TYPE: q8_0, q5_k_m, or q4_k_m
python tuner.py cloud-run --job-config Trainers/cloud/jobs/gguf_conversion.yaml --yes
# 3. GGUF is uploaded back to the same HF repo under gguf/
```

**Cloud GGUF conversion (direct script, outside cloud-run):**
```bash
python scripts/cloud_gguf_convert.py \
  --model-repo user/model-name \
  --quant q8_0 \
  --upload-to user/model-name
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

Cloud artifact policy:
- Default: artifacts stay in provider-native storage
- `hf_jobs`: Hugging Face Bucket
- `modal`: Modal Volume
- `runpod`: RunPod Network Volume
- Optional publish: only `final_model` is pushed to the target HF repo when enabled

## Environment Variables

```bash
HF_TOKEN=hf_...                       # Required for uploads
```

## Tips

- Always use `merged_16bit` as the source for GGUF conversion (best quality)
- The reliable GGUF converter merges LoRA once, then creates all quants (~10 min saved)
- Vision-language models auto-get an `mmproj.gguf` for the vision projector
- On macOS, bucket-backed cloud adapters are often easiest to handle one model at a time: pull the `final_model`, merge locally, create the quant you actually need first, then clean temp files before moving to the next model
- If the local machine lacks `unsloth`, a plain `transformers` + `peft` merge venv is an acceptable fallback for text models before llama.cpp conversion
- For merged local models, call the lower-level llama.cpp conversion path directly; the current reliable converter's top-level `convert()` flow assumes it starts from a LoRA adapter
- LM Studio on this repo owner's Mac uses `~/.lmstudio/models/<publisher>/<model-folder>/`; placing the `.gguf` there plus an optional `config.json` is enough for local testing after refresh/restart
- Qwen 3.5 adapters may need a `ConditionalGeneration` merge path instead of `AutoModelForCausalLM`; if the adapter keys live under `language_model.*`, inspect the base architecture before merging
- If `llama.cpp` says a merged model architecture is unsupported, update the local `Trainers/llama.cpp` checkout before retrying conversion; newer model families are often converter-gated rather than merge-gated
- On WSL, temp files use native filesystem to avoid NTFS performance issues
- `training_lineage.json` is auto-generated — includes model, LoRA, dataset, hardware info
- Use `upload_manifest.json` to verify what was uploaded
- The upload orchestrator handles everything — prefer `./run.sh` → Upload over manual commands
- Cloud jobs never rely on the remote container filesystem as the only copy; inspect provider-native storage first, then publish `final_model` if needed
- If local GGUF conversion OOMs (common on machines with <32GB RAM), use the cloud GGUF job (`cpu-upgrade` flavor, 32GB RAM, no GPU needed)
- The cloud GGUF script uses pure Python conversion (llama.cpp `convert_hf_to_gguf.py`) — no compilation required
- Some models (e.g., Gemma 4) may need tokenizer config patching before conversion — the cloud script handles known quirks automatically
