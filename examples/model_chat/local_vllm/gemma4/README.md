# Local Gemma 4 text serving candidate

This is a reusable, config-only Docker Compose profile for the official Google
Gemma 4 E4B and 12B instruction-tuned QAT W4A16 compressed-tensors checkpoints.
It serves one model at a time through vLLM's OpenAI-compatible API. E4B means
roughly 4B **effective** parameters, not 4B total parameters. Neither model is
the BF16 checkpoint, a pretrained base model, or a user-trained LoRA. No model
weights, prompts, credentials, or evaluation results are checked in here.

`model-inventory.json` pins the public Hub revisions, the verified
`model.safetensors` sizes and SHA-256 values, and the official Linux/AMD64 image
digest. `serving-packages.json` records the complete observed installed-package
inventory as package names and versions only. Both models are marked
`long_context_generation_observed`: each became HTTP-ready and completed three
full-chapter requests with curl exit 0 and finish reason `stop` on the observed
RTX 3090 24 GB / Windows WSL2 Docker Desktop host (NVIDIA driver 610.88,
Docker Engine 29.3.1). The E4B prompt/completion/total token counts were
58,632/3,618/62,250; 55,303/2,136/57,439; and 56,907/2,890/59,797. The
12B counts were 58,636/1,042/59,678; 55,307/2,490/57,797; and
56,911/1,822/58,733. Full per-request token and word counts are in the
inventory; no reasoning was observed in the 12B responses. These are three
technical trials per model, not reliability, narrative-fidelity or writing-
quality proof, nor a saturation test of the configured 98,304-token maximum.
The overall profile therefore remains **candidate**.

## Prepare model directories

Use the existing `hf download` CLI for the exact revision, with two distinct,
initially empty absolute directories; do not mix checkpoint revisions. For
example, in PowerShell, set these to directories you
control (do not put a token into the command or Compose file):

```powershell
$env:GEMMA4_E4B_MODEL_DIR = "C:/models/gemma4-e4b-qat"
$env:GEMMA4_12B_MODEL_DIR = "C:/models/gemma4-12b-qat"
hf download google/gemma-4-E4B-it-qat-w4a16-ct --revision 6cd26aaa2357fb2bad8c51699a7558a4d1a965bb --local-dir $env:GEMMA4_E4B_MODEL_DIR
hf download google/gemma-4-12B-it-qat-w4a16-ct --revision 1d2c2d7f2466070e69d6fb3fd5ce9a7d75f2f6ee --local-dir $env:GEMMA4_12B_MODEL_DIR
Get-FileHash "$env:GEMMA4_E4B_MODEL_DIR/model.safetensors" -Algorithm SHA256
Get-FileHash "$env:GEMMA4_12B_MODEL_DIR/model.safetensors" -Algorithm SHA256
```

Compare both hashes and byte lengths with `model-inventory.json` before
serving. Keep the complete snapshot together, including config, tokenizer,
chat template and generation config. Compose mounts it read-only and disables
Hub access inside the container. `pull_policy: never` also requires pulling
the reviewed image digest explicitly before starting a service.

## Serve one model

From this directory, after setting both model-directory variables (Compose
interpolates both services even when starting only one):

```powershell
docker pull vllm/vllm-openai@sha256:770fe65b2c73ee74a5c42165cf3433de4048cc2cd9c57a937ca4e35aba5aa87b
docker compose -f compose.yaml up -d e4b
curl.exe -fsS http://127.0.0.1:8012/health
docker compose -f compose.yaml stop e4b
docker compose -f compose.yaml up -d 12b
```

The services share loopback port 8012, so do not start both at once. Capture
any needed startup/inference logs to a fresh private path before stopping;
`stop` also preserves the container and its logs for later inspection, unlike
`down`/removal. For a user-approved OpenAI chat-completion request file, use
ordinary HTTP tooling:

```powershell
curl.exe -N -H "Content-Type: application/json" --data-binary "@C:/private/approved-request.json" http://127.0.0.1:8012/v1/chat/completions
```

The request file supplies the private prompt. Its model alias must be
`gemma4-e4b-raw` or `gemma4-12b-raw` for the running service. For comparable
decoding, preserve the public request settings below; do not rely on server
defaults to disable thinking. The profile cannot reproduce a private source
request unless that same request is separately supplied by its owner.
The request body below is a settings template, not a test prompt or a result;
replace the message content with an approved input and select the served alias
for the active model. Deliberately omit `max_tokens`:

```json
{
  "model": "gemma4-e4b-raw",
  "messages": [{"role": "user", "content": "<approved input>"}],
  "temperature": 1.0,
  "top_p": 0.95,
  "top_k": 64,
  "min_p": 0.0,
  "presence_penalty": 0.0,
  "repetition_penalty": 1.0,
  "seed": 20261008,
  "chat_template_kwargs": {"enable_thinking": false},
  "stream": true,
  "stream_options": {"include_usage": true}
}
```

Retain streamed chunks and final finish reason/usage when evaluating lengthy
outputs. An omitted request `max_tokens` adds no request-level output ceiling,
but the server's 98,304-token prompt-plus-output window, memory, and request
deadline remain finite. A partial or failed stream is not a quality result.

Both server configs use BF16 activations with the checkpoint's embedded 4-bit
quantization, the model's generation config and EOS tokens, Gemma 4 reasoning
parser, text-only mode, one sequence, and 1,024-token chunked prefill. They do
not enable LoRA, MTP/speculative decoding, or a forced quantization flag. The
WSL-compatible V1 model runner flag avoids the V2 UVA startup failure observed
on this host; eager execution is a separate setting. The configured 98,304
tokens are a candidate total window, not a promise of long-context fit.

| Observed startup boundary | Evidence-backed remedy in this profile |
| --- | --- |
| `AmbiguousGlobalPerLayerAttributeError` at `get_head_size` before model loading with vLLM 0.26 plus Transformers 5.17 | Use the digest-pinned official vLLM 0.26 image observed with Transformers 5.14.1. This version mismatch is also described in upstream [#51744](https://github.com/vllm-project/vllm/issues/51744) and [#52768](https://github.com/vllm-project/vllm/issues/52768). Do not set blanket global per-layer access. |
| `RuntimeError: UVA is not available` in the V2 runner on WSL after config admission | `VLLM_USE_V2_MODEL_RUNNER=0` selects the V1 runner before startup; upstream [#50239](https://github.com/vllm-project/vllm/issues/50239) reports this workaround. `enforce-eager` alone did not address the UVA failure. |

These are observed remedies for this host and image, not upstream guarantees or
automatic fallback behavior. A model name by itself does not select the tested
runtime, runner, server settings, or request settings.

For the observed requests, the caller—not the server YAML—set temperature 1,
top-p 0.95, top-k 64, min-p 0, presence penalty 0, repetition penalty 1,
seed 20261008, `chat_template_kwargs.enable_thinking: false`, streaming with
`stream_options.include_usage: true`, and omitted `max_tokens`. Preserve those
request settings when comparing future outputs; the server's reasoning parser
does not itself disable thinking. No user prompt or response is included here.

This example is independent of the bounded `scripts/chat_model.py` path in
`examples/model_chat/README.md`; it exposes a local HTTP server and does not
claim a verified-run or cloud-training identity.
