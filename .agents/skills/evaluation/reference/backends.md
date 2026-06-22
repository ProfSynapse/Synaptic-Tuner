# Evaluation Backends Reference

How to configure supported evaluation backends.

---

## Supported Backends

| Backend | Use Case | Model Specification |
|---------|----------|---------------------|
| `vllm` | Dedicated local/server inference for OpenAI-compatible chat completions | Served model name |
| `lmstudio` | Local LM Studio server | Model name loaded in LM Studio |
| `ollama` | Local Ollama server | Ollama model name |
| `unsloth` | Direct LoRA evaluation | Path to `final_model/` directory |
| `llamacpp` | Quantized GGUF models | Path to `.gguf` file |
| `openrouter` | Cloud API | Provider model id |
| `mlc` | Browser/WebLLM evaluation | MLC model path |

---

## vLLM

For a local vLLM container exposing an OpenAI-compatible endpoint:

```bash
python -m Evaluator.cli \
  --backend vllm \
  --model finetuned \
  --scenario tool_prompts.yaml \
  --host 127.0.0.1 \
  --port 8011 \
  --temperature 0 \
  --max-tokens 768
```

Use this for dedicated eval containers and fine-tuned model serving. The evaluator reads backend responses, builds a generic response view, and applies YAML `correct` assertions. `--model` is the **served model name** (what the container registered the model under), not a host path.

### Starting the container: `local-serve`

You do not need to hand-run `docker run` for the vLLM endpoint. The `local-serve` command launches a `vllm/vllm-openai` container that serves a local **merged (16-bit) model directory** on an OpenAI-compatible port, waits for it to become ready, and leaves it running for evaluation. No host pip install of vLLM is needed — serving lives in the container.

```bash
# Start: serve a merged model dir on port 8011 as "finetuned"
python tuner.py local-serve \
  --model path/to/merged-model \
  --serve-port 8011 \
  --served-model-name finetuned \
  --yes

# Inspect / stop
python tuner.py local-serve --status
python tuner.py local-serve --stop
```

Defaults: image `vllm/vllm-openai:latest` (override with `--serve-image vllm/vllm-openai:<your-custom-tag>` for an architecture an older `:latest` image does not yet support), port `8011`, served name `finetuned`, `--gpu-memory-utilization 0.90`, `--max-model-len 16384`. The container is named `tuner-vllm-serve`; a re-run detects an already-running container instead of double-launching. It mounts the model dir read-only at `/model` and publishes the container's port `8000` to the host `--serve-port`.

The command forces the **default** Docker daemon socket (`unix:///var/run/docker.sock`) regardless of the active Docker context, so it never drives a non-default (e.g. colima) context. It fails loud with clear messages for: daemon unreachable, image pull failure, model dir missing, or the server never becoming ready (it dumps the container log tail). First serve pulls the image — layer progress is streamed so a large download is distinguishable from a hang.

Once it reports ready, run the evaluator command above against `--host 127.0.0.1 --port <serve-port> --model <served-model-name>`.

---

## LM Studio

```bash
python -m Evaluator.cli \
  --backend lmstudio \
  --model qwen2.5-7b-instruct \
  --scenario tool_prompts.yaml
```

Environment variables:

```bash
LMSTUDIO_HOST=localhost
LMSTUDIO_PORT=1234
```

Setup:

1. Start LM Studio.
2. Load the model.
3. Start the local server.
4. Run evaluator.

---

## Ollama

```bash
python -m Evaluator.cli \
  --backend ollama \
  --model qwen2.5:7b-instruct \
  --scenario tool_prompts.yaml
```

Environment variables:

```bash
OLLAMA_HOST=127.0.0.1
OLLAMA_PORT=11434
```

---

## Unsloth

Best for evaluating a saved LoRA adapter directly without a separate server.

```bash
python -m Evaluator.cli \
  --backend unsloth \
  --model ./Trainers/sft/sft_output/TIMESTAMP/final_model \
  --scenario tool_prompts.yaml
```

Requirements:

- GPU runtime with Unsloth installed.
- Model path points at a compatible saved model/adaptor directory.

---

## llama.cpp

For evaluating quantized GGUF models.

```bash
python -m Evaluator.cli \
  --backend llamacpp \
  --model ./path/to/model-Q4_K_M.gguf \
  --scenario tool_prompts.yaml
```

---

## OpenRouter

For cloud-hosted model comparisons.

```bash
python -m Evaluator.cli \
  --backend openrouter \
  --model qwen/qwen-2.5-72b-instruct \
  --scenario tool_prompts.yaml
```

Environment variable:

```bash
OPENROUTER_API_KEY=sk-or-...
```

---

## MLC / WebLLM

```bash
python -m Evaluator.cli \
  --backend mlc \
  --model path/to/mlc-model \
  --scenario tool_prompts.yaml
```

---

## Backend Auto-Detection

The backend can sometimes be inferred from the model path:

- Path ending in `.gguf` -> `llamacpp`
- Directory with `adapter_config.json` -> `unsloth`
- Otherwise pass `--backend` explicitly.

---

## Comparison Pattern

Run the same scenario file against each model:

```bash
python -m Evaluator.cli --backend vllm --model base \
  --scenario tool_prompts.yaml \
  --output Evaluator/results/base_tools.json

python -m Evaluator.cli --backend vllm --model finetuned \
  --scenario tool_prompts.yaml \
  --output Evaluator/results/finetuned_tools.json
```

Compare `summary.correctness_pass_rate`.

---

## Environment Runtime Backends

Environment runtime is separate from model inference. It can execute or simulate tool calls after response validation:

```bash
# Local temp-dir runtime
python -m Evaluator.cli --backend lmstudio --model MODEL \
  --scenario tool_prompts.yaml \
  --env-backend local

# E2B sandbox runtime
python -m Evaluator.cli --backend lmstudio --model MODEL \
  --scenario tool_prompts.yaml \
  --env-backend e2b \
  --env-template YOUR_TEMPLATE
```

Use `--env-tool-schema` and `--env-exec-config` for custom runtime schemas and execution rules. Keep task correctness in scenario `correct` assertions unless runtime execution is explicitly part of the test.
