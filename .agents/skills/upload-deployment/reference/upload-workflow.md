# Upload Workflow Reference

Complete upload process from trained model to HuggingFace.

For cloud runs, provider-native storage is the default durable location. Treat Hugging Face Hub upload as a separate optional publish step for `final_model`, not as the canonical checkpoint store.

---

## Upload Entry Points

### From SFT Trainer
```bash
python3 scripts/upload_model.py \
  Trainers/sft/sft_output_rtx3090/TIMESTAMP/final_model \
  username/model-name \
  --save-method merged_16bit \
  --create-gguf
```

### From KTO Trainer
```bash
python3 scripts/upload_model.py \
  Trainers/kto/kto_output_rtx3090/TIMESTAMP/final_model \
  username/model-name \
  --save-method merged_16bit
```

### Via Interactive Menu
```bash
./run.sh
# Select: Upload → Choose training run → Configure save method
```

---

## Upload Orchestrator Workflow

The orchestrator (`shared/upload/orchestrator.py`) handles everything:

### Step 1: Save Model Locally

Choose a save strategy:

| Strategy | Flag | What Happens | Size (7B) |
|----------|------|--------------|-----------|
| `lora_only` | `--save-method lora_only` | Copy LoRA adapters only | ~100-500 MB |
| `merged_16bit` | `--save-method merged_16bit` | Merge LoRA → base, save FP16 | ~14 GB |
| `merged_4bit` | `--save-method merged_4bit` | Merge LoRA → base, save 4-bit | ~4 GB |

**Recommendation:** `merged_16bit` for production, `lora_only` for sharing adapters.

### Step 2: Upload to HuggingFace Hub

- Authenticates with `HF_TOKEN`
- Creates repo if it doesn't exist
- Uploads saved model files
- For cloud runs, this step is optional and only applies when final-model publishing is enabled

### Step 3: GGUF Conversion (Optional)

If `--create-gguf`:
1. Merges LoRA into base model (once)
2. Creates base GGUF (f16/bf16)
3. Quantizes to Q4_K_M, Q5_K_M, Q8_0
4. For VL models: creates `mmproj.gguf`
5. Uploads all GGUF files

### Step 4: Generate Documentation

Auto-generates:
- `README.md` — Model card with training info
- `upload_manifest.json` — Upload metadata
- `training_lineage.json` — Complete training provenance

### Step 5: Upload Documentation

Uploads all documentation files to HuggingFace.

---

## CLI Flags

| Flag | Description | Example |
|------|-------------|---------|
| `MODEL_PATH` | Path to trained model (positional) | `./sft_output/TIMESTAMP/final_model` |
| `REPO_ID` | HuggingFace repo (positional) | `username/model-name` |
| `--save-method` | Save strategy | `merged_16bit`, `merged_4bit`, `lora_only` |
| `--create-gguf` | Create GGUF quantizations | (flag) |
| `--private` | Make repo private | (flag) |

---

## Output Directory Structure

After upload, local structure:
```
model-name/
├── lora/                      # LoRA adapters
├── merged-16bit/              # Full merged model
├── gguf/                      # GGUF quantizations
│   ├── model-Q4_K_M.gguf
│   ├── model-Q5_K_M.gguf
│   ├── model-Q8_0.gguf
│   └── model-mmproj.gguf     # VL models only
├── upload_manifest.json
├── training_lineage.json
└── README.md
```

---

## Full Pipeline (Train + Upload + Eval)

```bash
./run.sh
# Select: Full Pipeline
# Runs: Train → Upload → Evaluate in sequence
```

Or manually:
```bash
# 1. Train
cd Trainers/rtx3090_sft && python train_sft.py --model-size 7b

# 2. Upload
python3 scripts/upload_model.py Trainers/sft/sft_output/LATEST/final_model user/model \
  --save-method merged_16bit --create-gguf

# 3. Evaluate
python -m Evaluator.cli --backend unsloth --model ./sft_output/LATEST/final_model \
  --scenario behavior_prompts.yaml --upload-to-hf user/model --update-model-card
```
