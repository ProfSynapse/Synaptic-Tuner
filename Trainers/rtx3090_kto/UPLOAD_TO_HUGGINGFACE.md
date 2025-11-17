# Upload to HuggingFace - Quick Guide

After your training completes, you can easily upload your model to HuggingFace Hub.

## 1. Setup HuggingFace Token (One-time)

### Get Your Token
1. Go to https://huggingface.co/settings/tokens
2. Create a new token with **WRITE** permissions
3. Copy the token (starts with `hf_...`)

### Save Token to .env File
```bash
# Copy the example file
cp .env.example .env

# Edit .env and paste your token
nano .env  # or use any text editor
```

Your `.env` file should look like:
```bash
HF_TOKEN=hf_YourActualTokenHere
```

**Security**: The `.env` file is in `.gitignore`, so your token won't be committed to git.

## 2. Upload After Training Completes

When training finishes, you'll see:
```
✓ Model saved to: ./kto_output_rtx3090/final_model
```

### Basic Upload (Recommended)
Upload as 16-bit merged model (best quality):

```bash
python src/upload_to_hf.py \
  ./kto_output_rtx3090/final_model \
  your-username/your-model-name
```

Example:
```bash
python src/upload_to_hf.py \
  ./kto_output_rtx3090/final_model \
  professorsynapse/claudesidian-tools-7b-v1
```

### Upload Options

**4-bit quantized (smaller size, ~3.5GB):**
```bash
python src/upload_to_hf.py \
  ./kto_output_rtx3090/final_model \
  your-username/your-model-name \
  --save-method merged_4bit
```

**LoRA adapters only (smallest, ~200MB):**
```bash
python src/upload_to_hf.py \
  ./kto_output_rtx3090/final_model \
  your-username/your-model-name \
  --save-method lora
```

**Private repository:**
```bash
python src/upload_to_hf.py \
  ./kto_output_rtx3090/final_model \
  your-username/your-model-name \
  --private
```

## 3. Upload GGUF Versions (For llama.cpp)

If you want to create GGUF versions for use with llama.cpp, Ollama, etc.:

```bash
python src/upload_to_hf.py \
  ./kto_output_rtx3090/final_model \
  your-username/your-model-name \
  --create-gguf
```

This will:
1. Upload the standard 16-bit model
2. Create GGUF versions (Q4_K_M, Q5_K_M, Q8_0)
3. Upload all GGUF files

**GGUF only (skip standard upload):**
```bash
python src/upload_to_hf.py \
  ./kto_output_rtx3090/final_model \
  your-username/your-model-name \
  --create-gguf \
  --skip-standard
```

**Custom GGUF quantizations:**
```bash
python src/upload_to_hf.py \
  ./kto_output_rtx3090/final_model \
  your-username/your-model-name \
  --create-gguf \
  --gguf-quantizations Q4_K_M Q5_K_M Q6_K Q8_0
```

## 4. Upload from Checkpoint

If you want to upload from a specific checkpoint instead of the final model:

```bash
python src/upload_to_hf.py \
  ./kto_output_rtx3090/checkpoint-100 \
  your-username/your-model-name-checkpoint100
```

## 5. What Gets Uploaded

### Standard Upload
- Model weights (safetensors)
- Tokenizer files
- Configuration files
- README with model card

### GGUF Upload
All of the above plus:
- `model-unsloth.gguf` (f16 base)
- `model-unsloth-Q4_K_M.gguf` (~3.5GB, fast)
- `model-unsloth-Q5_K_M.gguf` (~4.5GB, balanced)
- `model-unsloth-Q8_0.gguf` (~7GB, high quality)

## 6. Complete Example Workflow

```bash
# 1. Training completes
# Training saves to: ./kto_output_rtx3090/final_model

# 2. Upload to HuggingFace (token auto-loaded from .env)
python src/upload_to_hf.py \
  ./kto_output_rtx3090/final_model \
  professorsynapse/claudesidian-tools-7b-v1

# 3. View your model
# https://huggingface.co/professorsynapse/claudesidian-tools-7b-v1
```

## 7. Troubleshooting

### "Error: HuggingFace token required"
Make sure your `.env` file exists and has the correct token:
```bash
cat .env  # Should show: HF_TOKEN=hf_...
```

### "Permission denied"
Token needs WRITE permissions. Create a new token at:
https://huggingface.co/settings/tokens

### "Repository not found"
The repository will be created automatically. Make sure:
- Username is correct (your HF username)
- You have internet connection
- Token has write permissions

### "Out of memory during GGUF conversion"
GGUF conversion needs more RAM. Either:
- Use a machine with more RAM
- Skip GGUF: remove `--create-gguf` flag
- Convert GGUF separately later

## 8. Alternative: Manual Upload

If you prefer manual control:

```python
from unsloth import FastLanguageModel

# Load the trained model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./kto_output_rtx3090/final_model",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=False,
)

# Upload
model.push_to_hub_merged(
    "your-username/your-model-name",
    tokenizer,
    save_method="merged_16bit",  # or "merged_4bit" or "lora"
    token="hf_YourTokenHere"
)
```

## 9. Model Card Example

After upload, edit the README.md on HuggingFace to add details:

```markdown
---
license: apache-2.0
language:
- en
tags:
- unsloth
- kto
- mistral
datasets:
- your-dataset
---

# Your Model Name

Fine-tuned using KTO (Kahneman-Tversky Optimization) on RTX 3090.

## Training Details
- Base model: unsloth/mistral-7b-v0.3-bnb-4bit
- Method: KTO with LoRA
- Training time: ~2-3 hours
- Dataset: [your dataset]

## Usage

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "your-username/your-model-name",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True
)
```
```

## Quick Reference

| Task | Command |
|------|---------|
| Setup token | `cp .env.example .env` then edit `.env` |
| Basic upload | `python src/upload_to_hf.py ./kto_output_rtx3090/final_model username/model` |
| 4-bit upload | Add `--save-method merged_4bit` |
| Private repo | Add `--private` |
| With GGUF | Add `--create-gguf` |
| GGUF only | Add `--create-gguf --skip-standard` |

---

Your `.env` file is protected by `.gitignore` and won't be committed to git. Always keep your HF token private!
