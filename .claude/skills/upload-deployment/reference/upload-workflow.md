# Upload Workflow Reference

Complete upload process from trained model to HuggingFace.

For cloud runs, provider-native storage is the default durable location. Treat Hugging Face Hub upload as a separate optional publish step for `final_model`, not as the canonical checkpoint store.

---

## Upload Entry Points

### From SFT Trainer
```bash
python3 .skills/upload-deployment/scripts/upload_model.py \
  Trainers/sft/sft_output/TIMESTAMP/final_model \
  username/model-name \
  --save-method merged_16bit \
  --create-gguf
```

### From KTO Trainer
```bash
python3 .skills/upload-deployment/scripts/upload_model.py \
  Trainers/kto/kto_output/TIMESTAMP/final_model \
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
| `lora` | `--save-method lora` | Copy LoRA adapters only | ~100-500 MB |
| `merged_16bit` | `--save-method merged_16bit` | Merge LoRA → base, save FP16 | ~14 GB |
| `merged_4bit` | `--save-method merged_4bit` | Merge LoRA → base, save 4-bit | ~4 GB |

**Recommendation:** `merged_16bit` for production, `lora` for sharing adapters.

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
| `--save-method` | Save strategy | `merged_16bit`, `merged_4bit`, `lora` |
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
cd Trainers/sft && python train_sft.py --model-size 7b

# 2. Upload
python3 .skills/upload-deployment/scripts/upload_model.py Trainers/sft/sft_output/LATEST/final_model user/model \
  --save-method merged_16bit --create-gguf

# 3. Evaluate
python -m Evaluator.cli --backend unsloth --model ./sft_output/LATEST/final_model \
  --scenario behavior_prompts.yaml --upload-to-hf user/model --update-model-card
```


## Public verified native model publication to an HF model repository

The optional `synaptic_tuner.api.v1.reference.hf_publication.HFModelPublicationDestinationV1`
implements the existing publication destination port. Compose it through the
caller-owned registry and `compose_publication_artifacts`; the default reference
host still has no destinations. Existing legacy upload commands are unchanged.

The host supplies the original verified RunsAPI/source, durable publication store,
private disk spool/render root, original evidence keys/clock, an explicitly named
SecretRef/resolver and exact existing repository scope. Use Hub1.33.0, explicit
endpoint/repo/branch/visibility, reviewed parent commit, finite native/aggregate/
upload/commit-body limits and deadline. It creates no repository, performs no
merge/GGUF/model loading, and never overwrites or deletes an existing release slot.
Repository namespace is target scope, not proof of the token's authenticated owner
or least privilege. Real account/repository authorization remains a qualification
input; never substitute a scripted fixture identity.

Verified native `final_model` and `tokenizer` are tar archives. The admitted
renderer validates their actual members and commits disjoint model/tokenizer
files together under `releases/<publication_id>/model/`; unchanged five native
artifacts and an authenticated command/ownership/render marker are retained
alongside them. A PEFT adapter requires its pinned base model; it is not a merged
standalone model. Receipt/read paths bind an immutable commit OID.

The pinned SDK prepares content uploads; the final Add-only commit uses its real
NDJSON payload and parent CAS through bounded HTTP with redirects/retries disabled.
Preupload may involve multiple calls or vendor retries. After a lost reply or
process death, read-only lookup verifies original ownership plus the exact native
and rendered inventory at an immutable owned snapshot. It never resubmits. This
is content verification, not proof of the original introducing commit or RPC
causality. Missing/changed markers or unexpected files remain uncertain.

HTTP streams enforce actual per-file and aggregate content limits without bulk
cache downloads. Metadata, pagination and commit response bodies are finite. Caller
deadlines bound waiting, not physical SDK/HTTP termination; late uploads may
continue with independently owned file handles. Host validation/serialization and
OS calls are synchronous, without a total hard deadline or invoice guarantee.

CPU acceptance uses real public native SFT verification, SQLite close/reopen and
a file-backed scripted provider ledger, including SIGKILL before local receipt.
The GRPO join uses the real G2 native verifier over explicitly synthetic CPU model
files, then the existing trusted source port, admitted renderer and public
ArtifactsAPI. It verifies warm and lost-reply/cold immutable snapshots without
fabricating a TrainingAPI success. These controls qualify neither live HF uploads,
GPU training, model loading nor model quality.

The renderer selects the model only from its original SFT workload contract or the
exact `synaptic-packaged-grpo-workload/v1` contract revalidated by the existing
pure GRPO compiler. Preserve the original workload bytes, configured model and
pinned revisions, disjoint archive validation and five native roles; never rewrite
GRPO into SFT. Its parser adds five explicitly reviewed source members to the
inference closure (119 to 124); runtime/dependency/image pins are unchanged. A source
inventory update is not live inference-image qualification.
