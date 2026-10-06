# GGUF Conversion Reference

Creating GGUF quantized models for llama.cpp, Ollama, and other runtimes.

---

## Overview

GGUF (GPT-Generated Unified Format) is the standard format for running models with llama.cpp, Ollama, LM Studio, and other local inference tools.

---

## Quantization Formats

| Format | Size (7B) | Quality | Speed | Use Case |
|--------|-----------|---------|-------|----------|
| **f16/bf16** | ~14 GB | Maximum | Baseline | Source for other quants |
| **Q8_0** | ~7 GB | Very high | Fast | Best quality, more RAM |
| **Q5_K_M** | ~5 GB | High | Faster | Good balance |
| **Q4_K_M** | ~4 GB | Good | Fastest | Most popular, efficient |
| **IQ3_XXS / IQ2_\*** | ~2-3 GB | Lower | Fast | Very low RAM; imatrix required |

**Default quantizations:** Q4_K_M, Q5_K_M, Q8_0

Any `llama-quantize` type name is accepted (case-insensitive), e.g. `Q6_K`, `Q3_K_M`, `IQ4_XS`, `IQ3_XXS`.

---

## Creating GGUFs

### Via Upload (Recommended)

```bash
python3 .skills/upload-deployment/scripts/upload_model.py MODEL_PATH user/repo \
  --save-method merged_16bit \
  --create-gguf
```

This creates all three quantizations and uploads them together with `gguf_manifest.json`.

With an importance matrix calibrated on the training data:

```bash
python3 .skills/upload-deployment/scripts/upload_model.py MODEL_PATH user/repo \
  --save-method merged_16bit --create-gguf \
  --gguf-quantizations Q4_K_M IQ3_XXS \
  --gguf-calibration Datasets/train.jsonl \
  [--gguf-calibration-rows 512] [--gguf-imatrix-chunks 128] [--gguf-imatrix-ctx 512]
```

Calibration is wired into `--create-gguf` (the reliable converter) only; `--gguf-only` does not support it.

### From an Already-Merged Model

```bash
python -m shared.upload.converters.gguf_reliable MERGED_DIR OUT_DIR --merged \
  --name my-model --quants Q4_K_M Q8_0 \
  [--calibration train.jsonl] [--llama-cpp-dir Trainers/llama.cpp] [--no-cleanup]
```

Writes `OUT_DIR/gguf/*.gguf` plus `gguf_manifest.json`; the base GGUF, calibration text and imatrix live in `OUT_DIR/work/` and are deleted unless `--no-cleanup`. From Python, call `ReliableGGUFConverter.convert_merged_model(...)`.

### Via Interactive Menu

```bash
./run.sh
# Select: Convert → GGUF
```

### Via Cloud Job (When Local RAM is Insufficient)

Use this when the model's tensors exceed local RAM (common with large vocab models like Gemma 4).
Requires the merged model to be on HuggingFace first.

```bash
# Edit GGUF_MODEL_REPO, GGUF_QUANT_TYPE (space/comma list, e.g. "q4_k_m q8_0") and
# optionally GGUF_CALIBRATION_REPO + GGUF_CALIBRATION_FILE in the job YAML, then:
python tuner.py cloud-run --job-config Trainers/recipes/gguf_conversion.yaml --yes
```

- **Flavor**: `cpu-upgrade` (32GB RAM, no GPU needed)
- **Image**: Same unsloth image as training jobs
- **Fast path (no compilation)**: when every requested type is a `convert_hf_to_gguf.py --outtype` type (`f32`, `f16`, `bf16`, `q8_0`, `tq1_0`, `tq2_0`), each file is written directly by the Python converter
- **Quantize path**: any other type (`q4_k_m`, `q5_k_m`, `q6_k`, IQ types) → bf16 base GGUF → optional imatrix → `llama-quantize`. The job compiles `llama-quantize` and `llama-imatrix` (CPU-only) from the cloned llama.cpp first, installing the `cmake` wheel if the image lacks cmake; it fails with an explicit message if no C/C++ compiler exists. Budget a few minutes of build time on `cpu-upgrade`
- **Calibration**: `--calibration-repo` + `--calibration-file` (HF dataset repo, downloaded with `hf_hub_download(repo_type="dataset")`) or `--calibration-path` (local)
- **Output**: all produced GGUFs plus `gguf_manifest.json` are uploaded to `<repo>/gguf/`; the bf16 base is not uploaded on the quantize path. The job fails before uploading if any requested type was not produced
- **Script**: `scripts/cloud_gguf_convert.py` — can also be run standalone outside cloud-run

**When to use cloud vs local:**

| Scenario | Method |
|----------|--------|
| System RAM >= 2x model size | Local (`./run.sh` → Convert) |
| System RAM < 2x model size | Cloud job (`cpu-upgrade`) |
| Large vocab models (Gemma 4, etc.) | Cloud job (vocab tensor can be 5GB+) |
| Multiple quantizations needed | Either: both convert once and quantize every type from one base |

---

## Reliable GGUF Converter

The system uses a "reliable" converter (`shared/upload/converters/gguf_reliable.py`) that optimizes the conversion process:

### Key Optimization: Single Merge

**Traditional approach (Unsloth):**
```
Each quantization: Load LoRA → Merge → Convert → Quantize (~8 min each)
3 quants = ~24 minutes total
```

**Reliable approach:**
```
Merge LoRA once (~3 min)
Then quantize each: ~3-5 min each
3 quants = ~14 minutes total (10 min saved!)
```

### Vision-Language Model Support

Auto-detected for models like Qwen-VL, LLaVA, Pixtral:
- Creates main model GGUF
- Creates separate `mmproj.gguf` for vision projector
- Detection based on:
  - `preprocessor_config.json` or `image_processor_config.json` present
  - Vision config keys in `config.json`
  - Model type indicators (qwen2-vl, llava, pixtral)

### GGUF Manifest

Every conversion writes `gguf_manifest.json` next to the GGUF files (uploaded with them). Keys are stable snake_case so evaluation can cite the exact artifact:

```json
{
  "schema_version": 1,
  "created_at": "2026-01-01T00:00:00+00:00",
  "model_name": "my-model",
  "source_model": "user/my-model",          // repo id, or file name only for a local path
  "base_dtype": "bf16",                      // null when files were written directly by convert_hf_to_gguf
  "llama_cpp_commit": "84e76d8a...",         // git rev-parse HEAD of the llama.cpp checkout, or null
  "calibration": {                           // null when no imatrix was used
    "source": "hf://datasets/user/data/train.jsonl",
    "text_sha256": "...", "render_mode": "chat_template",
    "rows_total": 2676, "rows_used": 512, "rows_skipped": 0, "rows_filtered": 0,
    "chars": 1234567, "seed": 0, "max_rows": 512, "max_chars": 4000000,
    "label_field": null, "chunks": 128, "ctx_size": 512, "imatrix_sha256": "..."
  },
  "files": [
    {"filename": "my-model-Q4_K_M.gguf", "role": "quant", "quant_type": "Q4_K_M",
     "size_bytes": 4683073536, "sha256": "...", "imatrix_used": true}
  ]
}
```

`role` is `base`, `mmproj` or `quant`. `imatrix_used` is false for types whose llama.cpp kernels ignore an imatrix (F32/F16/BF16/Q8_0/TQ1_0/TQ2_0) even when one was computed.

### WSL Handling

On WSL, the converter:
- Uses native Linux filesystem (`~/tmp_gguf/`) for temp files
- Avoids NTFS performance issues with `/mnt/c/` paths
- Auto-detects WSL environment

---

## Importance Matrix (imatrix)

### What It Is

`llama-imatrix` runs calibration text through the unquantized (bf16) GGUF and records, per weight column, how strongly it is activated. `llama-quantize --imatrix` then spends its precision budget on the columns that matter. It is the llama.cpp analogue of post-training-quantization calibration (e.g. NVIDIA Model-Optimizer calibrating on 128-512 domain samples).

### When It Helps

| Types | Effect |
|-------|--------|
| F16/BF16/Q8_0 (and TQ1_0/TQ2_0) | None; the kernels ignore it, so it is skipped |
| Q6_K, Q5_K_* | Small |
| Q4_K_*, Q4_0, Q3_K_*, IQ4_* | Noticeable; recommended |
| IQ1_\*, IQ2_\*, IQ3_XXS, IQ3_XS, Q2_K_S | **Required**; llama.cpp refuses to quantize without it and the converter fails fast |

It is off unless a calibration dataset is supplied.

### Calibration Data

- Use the SFT training data (or representative prompts in the deployment format). Calibrating on the model's own domain beats generic text such as wikitext for a fine-tuned model
- Any JSONL row shape the trainers accept works (`messages`, `conversations`, `prompt`/`completion`); rows are normalized by `shared/sft_preprocessing.py` and rendered with the merged model's own chat template, so the text matches the prompt format the model sees. Without a chat template the message contents are joined as plain text
- Default sample: 512 rows (seeded, deterministic), 4M character cap, 128 chunks × 512 tokens (~65k tokens). llama-imatrix needs at least 2 × ctx tokens. Rows that cannot be rendered are skipped and counted
- CPU cost scales with chunks × model size; on `cpu-upgrade` expect minutes for small models and longer for 7B+. Lower `--imatrix-chunks` if needed
- Preview the rendered text: `python -m shared.upload.converters.calibration train.jsonl scratch/cal.txt --tokenizer-dir MERGED_DIR`. For KTO data, `--label-field <field>` leaves out rows whose field is falsy (not set by default)

### Privacy

The calibration text and the imatrix file stay in the local/job work directory and are never uploaded. `gguf_manifest.json` records only the source name (a repo id, or the file name for a local path, never the full local path), row counts and sha256 hashes of the rendered text and imatrix. Note that `llama-quantize` itself embeds the imatrix and calibration **file paths** (not contents) in the GGUF metadata (`quantize.imatrix.file`, `quantize.imatrix.dataset`), along with entry and chunk counts.

---

## llama.cpp Requirement

GGUF conversion requires llama.cpp to be built locally.

### Auto-Build

`./run.sh` auto-offers to clone and build if missing:
```bash
git clone --depth 1 https://github.com/ggerganov/llama.cpp.git Trainers/llama.cpp
cd Trainers/llama.cpp
cmake -B build -DGGML_CUDA=ON    # Linux/WSL with NVIDIA
cmake --build build --config Release -j$(nproc)
```

**Platform-specific flags:**
- Apple Silicon: `-DGGML_METAL=ON`
- Linux/WSL (NVIDIA): `-DGGML_CUDA=ON`
- Windows: `-DGGML_CUDA=ON`

### Verify Build

```bash
ls Trainers/llama.cpp/build/bin/llama-cli
# Should exist after successful build
```

---

## Using GGUF Models

### With Ollama
```bash
# Create Modelfile
echo "FROM ./model-Q4_K_M.gguf" > Modelfile
ollama create my-model -f Modelfile
ollama run my-model
```

### With LM Studio
1. Copy `.gguf` file to LM Studio models directory
2. Load in LM Studio UI
3. Start local server

### With llama.cpp
```bash
./Trainers/llama.cpp/build/bin/llama-cli \
  -m path/to/model-Q4_K_M.gguf \
  -p "User prompt here" \
  -n 512
```

---

## Cleanup

Orphaned temp files from interrupted conversions:
```bash
# The converter has auto-cleanup
# For manual cleanup:
rm -rf ~/tmp_gguf/
```
