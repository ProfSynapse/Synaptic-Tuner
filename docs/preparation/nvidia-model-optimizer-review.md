# NVIDIA Model Optimizer: What To Borrow

**Date**: 2026-09-24
**Status**: Reviewed; items 1 and 2 implemented
**Context**: Does NVIDIA Model Optimizer (ModelOpt, `nvidia-modelopt`, https://github.com/NVIDIA/Model-Optimizer, reviewed at `main` ed7e879, v0.47.0) contain techniques worth adopting in this engine?

---

## 1. Verdict

Do not take ModelOpt as a dependency of the training path. Borrow three ideas.

Why not the dependency:

- **No Unsloth support.** Its LLaMA-Factory integration asserts `not model_args.use_unsloth`. Its QAT, QAD and KD trainers assume plain HF/TRL models, typically under FSDP2.
- **Data-center hardware.** The documented QAT/QAD examples assume 80GB GPUs (Qwen3-8B QAT needs 2x80GB). Minitron pruning runs only on Megatron-Core models inside the NeMo container.
- **Wrong deployment targets.** It exports unified HF checkpoints for TensorRT-LLM, vLLM and SGLang. There is no GGUF writer. The GGML IQ1/IQ2 formats in the 0.48 dev branch are packed into safetensors, not `.gguf`. Our main deployment targets are GGUF (llama.cpp, Ollama, LM Studio) and vLLM.
- **Unstable.** It is pre-1.0 with a monthly cadence. Deprecations get one release of notice (`--qformat`, `llm_ptq`/`vlm_ptq` folders and the legacy TRT-LLM export are all recent removals). The `[hf]` extra pins `transformers<5.15`.

## 2. Borrowed

| # | ModelOpt idea | Our version | Status |
|---|---------------|-------------|--------|
| 1 | `llm_eval`: same benchmark on full-precision and quantized model | `python -m Evaluator.compare` gate plus quantization recorded in eval lineage | Implemented |
| 2 | PTQ calibrates on 128-512 domain samples | llama.cpp importance matrix built from the training dataset, passed to `llama-quantize --imatrix` | Implemented |
| 3 | QAD: fine-tuned full-precision model as teacher to recover quantization loss | Unsloth `qat_scheme` (torchao) for vLLM int4/int8 targets; offline top-k logprob KD from same-tokenizer teachers | Not started |

Item 2 also fixed a bug found during the review. The cloud GGUF path (`scripts/cloud_gguf_convert.py`) passed k-quant names to `convert_hf_to_gguf.py --outtype`, which only accepts f32/f16/bf16/q8_0/tq*. The skill documented `q4_k_m`/`q5_k_m` for that path, but those could not be produced there. The cloud path now converts to a bf16 base and runs `llama-quantize`.

### Notes on item 3

- ModelOpt's HF `KDTrainer` is logit-only, returns only the KD loss (no CE+KD weighting yet), keeps the teacher resident, and needs FSDP2. The reusable part is the pattern: a TRL `SFTTrainer` subclass whose loss is a weighted sum of CE and KL against teacher logits.
- Holding a teacher in memory does not fit 24GB training. The practical variant is to capture the teacher's top-k logprobs during `batch-generate --engine vllm` and train against them offline.
- Teacher and student must share a tokenizer. That rules out the default SynthChat teacher (`openai/gpt-oss-120b` via OpenRouter) for logit KD, so this only applies to local same-family teachers.
- QAT pays off only when serving the exact simulated format. GGUF k-quants are not what torchao simulates, so QAT targets vLLM int4/int8 serving, not GGUF.

## 3. Deferred

- **FP8 / NVFP4 export** of the merged 16-bit checkpoint through ModelOpt, run in its own HF Jobs container so its pins stay isolated from Unsloth. FP8 serves on L4, L40S and H100; NVFP4 needs Blackwell. Revisit when vLLM serving is a primary deployment path (the enterprise flywheel plan points that way).
- **EAGLE-3 draft heads** (`mtsp.convert`, exports to vLLM and SGLang). The interaction with vLLM LoRA hot-swap is unverified and must be checked first.
- **AutoQuantize** (per-layer mixed precision under a bits budget). The GGUF analogue is per-tensor type overrides in `llama-quantize`. Only worth trying once the comparison gate can measure the effect.

## 4. Skipped

- Minitron pruning: Megatron-Core only.
- Puzzletron NAS: heavy search, pays off at scale.
- 2:4 sparsity: no unified HF export; sparsity-aware training needs 8x80GB.
- ModelOpt-Windows: ONNX-only PTQ.

## 5. Sources

- https://github.com/NVIDIA/Model-Optimizer: `examples/hf_ptq`, `examples/llm_qat` (including `llama_factory/`), `examples/llm_distill`, `examples/llm_eval`, `examples/pruning`, `examples/speculative_decoding`, `docs/source/guides/0_support_matrix.rst`, `docs/source/guides/1_quantization.rst`, `docs/source/deployment/3_unified_hf.rst`, `CHANGELOG.rst`
- https://pypi.org/project/nvidia-modelopt/
