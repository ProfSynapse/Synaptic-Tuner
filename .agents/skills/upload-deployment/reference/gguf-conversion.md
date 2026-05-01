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

**Default quantizations:** Q4_K_M, Q5_K_M, Q8_0

---

## Creating GGUFs

### Via Upload (Recommended)

```bash
python3 scripts/upload_model.py MODEL_PATH user/repo \
  --save-method merged_16bit \
  --create-gguf
```

This creates all three quantizations and uploads them.

### Via Interactive Menu

```bash
./run.sh
# Select: Convert → GGUF
```

### Via Cloud Job (When Local RAM is Insufficient)

Use this when the model's tensors exceed local RAM (common with large vocab models like Gemma 4).
Requires the merged model to be on HuggingFace first.

```bash
# Edit GGUF_MODEL_REPO and GGUF_QUANT_TYPE in the job YAML, then:
python tuner.py cloud-run --job-config Trainers/recipes/gguf_conversion.yaml --yes
```

- **Flavor**: `cpu-upgrade` (32GB RAM, no GPU needed)
- **Image**: Same unsloth image as training jobs
- **Process**: Downloads model from HF → clones llama.cpp → pure Python conversion → uploads GGUF back to HF repo
- **No compilation**: Uses `convert_hf_to_gguf.py` directly (Python-only, no cmake/make)
- **Script**: `scripts/cloud_gguf_convert.py` — can also be run standalone outside cloud-run

**When to use cloud vs local:**

| Scenario | Method |
|----------|--------|
| System RAM >= 2x model size | Local (`./run.sh` → Convert) |
| System RAM < 2x model size | Cloud job (`cpu-upgrade`) |
| Large vocab models (Gemma 4, etc.) | Cloud job (vocab tensor can be 5GB+) |
| Multiple quantizations needed | Local reliable converter (merges once, quants in parallel) |

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

### WSL Handling

On WSL, the converter:
- Uses native Linux filesystem (`~/tmp_gguf/`) for temp files
- Avoids NTFS performance issues with `/mnt/c/` paths
- Auto-detects WSL environment

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
