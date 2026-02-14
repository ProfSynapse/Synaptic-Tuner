# Evaluation Backends Reference

How to configure each supported evaluation backend.

---

## Supported Backends

| Backend | Use Case | Model Specification |
|---------|----------|---------------------|
| **unsloth** | Direct LoRA evaluation | Path to `final_model/` directory |
| **llamacpp** | Quantized GGUF models | Path to `.gguf` file |
| **lmstudio** | Local inference server | Model name loaded in LM Studio |
| **ollama** | Local inference server | Model name in Ollama |
| **openrouter** | Cloud API | Model ID (e.g., `qwen/qwen-2.5-72b`) |
| **mlc** | Browser-based WebLLM | Path to MLC model |

---

## Backend Configuration

### Unsloth (Direct LoRA)

Best for evaluating freshly trained LoRA adapters without a server.

```bash
python -m Evaluator.cli \
  --backend unsloth \
  --model ./Trainers/rtx3090_sft/sft_output_rtx3090/TIMESTAMP/final_model
```

**Requirements:**
- GPU with Unsloth installed
- Loads base model + merges LoRA in memory

---

### llama.cpp (GGUF)

For evaluating quantized GGUF models.

```bash
python -m Evaluator.cli \
  --backend llamacpp \
  --model ./path/to/model-Q4_K_M.gguf
```

**Requirements:**
- llama.cpp built locally (`Trainers/llama.cpp/`)
- `./run.sh` auto-offers to build if missing

---

### LM Studio

For evaluating models loaded in LM Studio's local server.

```bash
python -m Evaluator.cli \
  --backend lmstudio \
  --model qwen2.5-7b-instruct
```

**Environment variables:**
```bash
LMSTUDIO_HOST=localhost    # Auto-detects Windows host in WSL
LMSTUDIO_PORT=1234
```

**Setup:**
1. Start LM Studio
2. Load model
3. Start local server (port 1234)
4. Run evaluator

---

### Ollama

For evaluating Ollama-served models.

```bash
python -m Evaluator.cli \
  --backend ollama \
  --model qwen2.5:7b-instruct
```

**Environment variables:**
```bash
OLLAMA_HOST=127.0.0.1
OLLAMA_PORT=11434
```

**Setup:**
1. `ollama serve` (if not already running)
2. `ollama pull qwen2.5:7b-instruct`
3. Run evaluator

---

### OpenRouter (Cloud)

For evaluating via OpenRouter's API.

```bash
python -m Evaluator.cli \
  --backend openrouter \
  --model qwen/qwen-2.5-72b-instruct
```

**Environment variables:**
```bash
OPENROUTER_API_KEY=sk-or-...
```

**Use case:** Evaluating large models (70B+) that don't fit locally, or baseline comparisons against frontier models.

---

### MLC (WebLLM)

For browser-based evaluation with WebGPU.

```bash
python -m Evaluator.cli \
  --backend mlc \
  --model path/to/mlc-model
```

---

## Auto-Detection

The backend can sometimes be auto-detected from the model path:
- Path ending in `.gguf` → `llamacpp`
- Path to directory with `adapter_config.json` → `unsloth`
- Otherwise → must specify `--backend`

---

## Comparison Testing Pattern

Evaluate the same scenarios across different backends to compare:

```bash
# Base model (LM Studio)
python -m Evaluator.cli --backend lmstudio --model qwen2.5-7b-instruct \
  --scenario behavior_prompts.yaml --output Evaluator/results/base.json

# Fine-tuned LoRA (Unsloth)
python -m Evaluator.cli --backend unsloth --model path/to/lora \
  --scenario behavior_prompts.yaml --output Evaluator/results/finetuned.json

# GGUF quantized (llama.cpp)
python -m Evaluator.cli --backend llamacpp --model path/to/model-Q4_K_M.gguf \
  --scenario behavior_prompts.yaml --output Evaluator/results/gguf.json
```

Then compare `pass_rate` across the JSON results.

---

## Environment Runtime Backends (Optional)

Separate from model inference backend, evaluator can execute tool calls in a runtime:

```bash
# Local temp-dir runtime
python -m Evaluator.cli --backend lmstudio --model MODEL --scenario tool_prompts.yaml --env-backend local

# E2B sandbox runtime
python -m Evaluator.cli --backend lmstudio --model MODEL --scenario tool_prompts.yaml --env-backend e2b --env-template YOUR_TEMPLATE
```

Use `--env-tool-schema` and `--env-exec-config` to support custom tool names and rules.
