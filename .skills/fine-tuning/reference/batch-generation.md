# Batch generation and structured decoding

Use the existing public `batch-generate` verb. Prefer pinned vLLM with
`VLLM_BATCH_INVARIANT=1` when the model's attention backend supports it. Model
support alone does not establish batch-invariance support. Never bypass the
backend's capability refusal.

An HF fallback can preserve the configured JSON interface:

```bash
synaptic batch-generate --engine hf-batched --model LOCAL_CHECKPOINT \
  --prompts PROMPTS_JSONL --out-dir OUTPUT --batch-size 1 \
  --json-schema SCHEMA_JSON --structured-output-backend xgrammar --resume
```

HF JSON-schema decoding requires the optional `xgrammar` package. Pin its
version alongside Torch, Transformers, model/tokenizer revisions, dtype and
hardware in the consuming study. A fresh grammar matcher is created per batch.
No grammar dependency is loaded when a schema is absent. Registered conditional
generation models can serve text-only inputs through their Transformers auto
model class; no architecture-name heuristic or retry after a weight error is used.

HF batch size one avoids changes in batch composition. It is not a claim of
cross-version or cross-hardware bitwise determinism. Qualify exact prompt/token
bytes, repeated and permuted completions, natural termination, whole-schema
validation, and hard-kill/resume on the actual model before scientific use.
Never silently change a signed backend or resume across different configuration
hashes. HF does not accept the vLLM-only arbitrary-whitespace control.

The XGrammar HF integration follows its [official quick start](https://xgrammar.mlc.ai/docs/latest/start/quick_start.html).
