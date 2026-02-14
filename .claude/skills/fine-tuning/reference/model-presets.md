# Model Presets Reference

Supported models, VRAM requirements, and LoRA configuration per model size.

---

## Model Size Presets

Use `--model-size` for quick selection:

### 3B Models (Fast Iteration)

| Flag | Model | VRAM | Batch Size | Use Case |
|------|-------|------|------------|----------|
| `--qwen-3b` | Qwen2.5-3B-Instruct | ~6-8 GB | 12 | Fast prototyping |
| `--llama-3b` | Llama-3.2-3B-Instruct | ~6-8 GB | 12 | Fast prototyping |

### 7-8B Models (Production — Recommended)

| Flag | Model | VRAM | Batch Size | Use Case |
|------|-------|------|------------|----------|
| `--mistral-7b` | Mistral-7B-v0.3 | ~7-9 GB | 6 | Production default |
| `--llama-8b` | Llama-3.1-8B-Instruct | ~7-9 GB | 6 | Production |
| `--qwen-7b` | Qwen2.5-7B-Instruct | ~7-9 GB | 6 | Production |
| `--magistral` | Magistral-Small-2509 | ~7-9 GB | 6 | Production |
| `--deepseek-7b` | DeepSeek-R1-Distill-Qwen-7B | ~7-9 GB | 6 | Reasoning |
| `--qwen-vl-8b` | Qwen3-VL-8B-Instruct | ~9-11 GB | 4 | Vision-Language |
| `--qwen-thinking-8b` | Qwen3-VL-8B-Thinking | ~9-11 GB | 4 | Reasoning + Vision |

### 13-14B Models (Advanced)

| Flag | Model | VRAM | Batch Size |
|------|-------|------|------------|
| `--llama-13b` | Llama-2-13B | ~12-14 GB | 4 |
| `--llama-vision-11b` | Llama-3.2-11B-Vision | ~12-14 GB | 4 |
| `--gemma-12b` | Gemma-3-12B-Instruct | ~12-14 GB | 4 |
| `--deepseek-14b` | DeepSeek-R1-Distill-Qwen-14B | ~12-14 GB | 4 |

### 20-24B Models (Very Large)

| Flag | Model | VRAM | Batch Size |
|------|-------|------|------------|
| `--llama-scout-17b` | Llama-4-Scout-17B | ~16-18 GB | 4 |
| `--gpt-20b` | GPT-OSS-20B | ~16-18 GB | 4 |
| `--mistral-24b` | Mistral-Small-3.2-24B | ~18-20 GB | 2 |

---

## LoRA Settings by Size

| Size | Rank (r) | Alpha | Dropout | Target Modules |
|------|----------|-------|---------|----------------|
| 3B | 32 | 64 | 0.05 | q,k,v,o,gate,up,down |
| **7B** | **64** | **128** | **0.05** | **q,k,v,o,gate,up,down** |
| 13B | 128 | 256 | 0.05 | q,k,v,o,gate,up,down |
| 20B | 128 | 256 | 0.05 | q,k,v,o,gate,up,down |

**Rule of thumb:** `lora_alpha = 2 × rank`

---

## VRAM Budget (RTX 3090 — 24 GB)

```
Base model (4-bit):     ~3.5 GB (7B)
LoRA adapters:          ~1.0 GB
Activations/gradients:  ~2.5 GB
Optimizer state:        ~1.5 GB
CUDA context:           ~0.5 GB
                        --------
Total:                  ~9 GB
Free headroom:          ~15 GB
```

**With KTO (reference model):**
- Implicit ref (default): ~0 extra (shared weights)
- Explicit ref: ~8 GB extra

---

## Chat Template Mapping

Models auto-detect their chat template:
- **Qwen** → `chatml`
- **Llama** → `llama-3`
- **Mistral** → `mistral`
- **Default** → `chatml`

---

## Manual Model Selection

Override any preset with:
```bash
python train_sft.py --model-name "unsloth/Qwen2.5-7B-Instruct-bnb-4bit" --max-seq-length 4096
```

Or in `config.yaml`:
```yaml
model:
  model_name: "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
  max_seq_length: 4096
  load_in_4bit: true
```

**4-bit model naming convention:** `unsloth/<model>-bnb-4bit`

---

## Recommendation

| Use Case | Model | Why |
|----------|-------|-----|
| Quick iteration | 3B (Qwen) | Fast, fits easily |
| **Production** | **7B (Mistral/Qwen)** | **Best quality/speed balance** |
| Maximum quality | 13-14B | Higher quality, slower |
| Reasoning tasks | DeepSeek-R1 | Distilled reasoning |
| Multimodal | Qwen-VL | Vision + Language |
