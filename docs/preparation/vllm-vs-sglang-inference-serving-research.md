# Research: vLLM vs SGLang for Multi-Tenant LoRA Inference Serving

**Date**: 2026-03-20
**Context**: Technical evaluation for a hosted fine-tuning platform serving user fine-tuned models (3B-14B) with multi-LoRA, OpenAI-compatible API, tool-calling workloads, on RTX 3090 24GB GPUs.

---

## Executive Summary

**vLLM is the recommended starting point for this platform**, primarily due to its mature multi-LoRA hot-swap capabilities, more stable tool-calling support, and production-proven ecosystem. SGLang offers 20-30% higher raw throughput and superior structured output performance, but its LoRA management is less mature (no filesystem resolver, blocking during load/unload) and its tool-calling support is experimental/unstable as of early 2026.

For a platform where LoRA hot-swap reliability and tool-calling are critical (both are for this use case), vLLM's maturity outweighs SGLang's throughput advantage. The APIs are similar enough that migration to SGLang is feasible once its LoRA and tool-calling ecosystems mature.

---

## 1. LoRA Adapter Serving

### vLLM

**Multi-adapter concurrent serving**: Fully supported. Multiple users can hit different LoRA adapters simultaneously from one base model. Based on S-LoRA paper integration.

**Key parameters**:
| Parameter | Purpose | Default |
|-----------|---------|---------|
| `--enable-lora` | Activate LoRA support | Off |
| `--max-loras` | Max number of LoRA adapters loaded simultaneously | 1 |
| `--max-lora-rank` | Maximum supported LoRA rank | 16 |
| `--lora-extra-vocab-size` | Extra vocabulary size for LoRA adapters | 256 |

**Memory model**: Uses PagedAttention for KV cache. LoRA adapters are stored separately with LRU eviction when GPU memory is full. Each LoRA adapter adds relatively small overhead (typically 10-50MB depending on rank and target modules) on top of the base model.

**Dynamic loading API**:
- `POST /v1/load_lora_adapter` — load by name + path
- `POST /v1/unload_lora_adapter` — unload by name
- `load_inplace` parameter — replace existing adapter weights in-place without name change
- Requires `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` environment variable
- **Filesystem resolver**: Automatically discovers and loads LoRA adapters from a directory when requests arrive — no manual registration needed. Critical for multi-tenant environments.

**Limitations**:
- Not all model architectures support LoRA (check supported model list)
- `strict` mode for schema-constrained LoRA outputs not yet implemented

### SGLang

**Multi-adapter concurrent serving**: Supported via S-LoRA/Punica integration. Each sequence in a batch can specify its own adapter. Multiple adapters batched together on single GPU.

**Key parameters**:
| Parameter | Purpose | Default |
|-----------|---------|---------|
| `--enable-lora` | Activate LoRA support | Off |
| `--max-loras-per-batch` | Max adapters per batch | 8 |
| `--max-loaded-loras` | CPU memory limit for loaded adapters | - |
| `--lora-eviction-policy` | LRU or FIFO | LRU |
| `--lora-backend` | `triton` or `csgmv` (20-80% faster) | `csgmv` |

**Memory model**: Similar to vLLM — GPU pool slots with LRU/FIFO eviction. Adapters can be "pinned" to prevent eviction. Max pinned = `max_loras_per_batch - 1`.

**Dynamic loading API**:
- `POST /load_lora_adapter` — load by name + path
- `POST /unload_lora_adapter` — unload by name
- **BLOCKING**: Server cannot handle requests during load/unload operations
- **No filesystem resolver**: Requires explicit registration. A feature request (issue #15417) and PR (#15454) exist but are still open as of March 2026.

**Limitations**:
- Blocking load/unload is a significant production concern for zero-downtime adapter updates
- No automatic adapter discovery — must register each adapter manually
- LoRA overlap loading (async H2D transfer) available but constrained: `max_loaded_loras` limited to 2x `max_loras_per_batch`

### Verdict: LoRA Serving

**vLLM wins clearly.** Non-blocking hot-swap, filesystem resolver for automatic discovery, `load_inplace` for zero-downtime adapter updates, and more mature API. SGLang's blocking load/unload and lack of filesystem resolver are deal-breakers for a multi-tenant platform.

---

## 2. Throughput & Latency Benchmarks

### Raw Throughput (H100 80GB, Llama 3.1 8B)

| Engine | Tokens/sec | Relative |
|--------|-----------|----------|
| SGLang | ~16,200 | 1.00x |
| LMDeploy | ~16,100 | ~1.00x |
| vLLM | ~12,500 | 0.77x |

Source: [PremAI 2026 benchmark](https://blog.premai.io/vllm-vs-sglang-vs-lmdeploy-fastest-llm-inference-engine-in-2026/)

**SGLang is ~29% faster than vLLM** on raw throughput in these benchmarks.

### Multi-GPU Scaling

SGLang shows ~1.5x request volume vs vLLM when scaled across two GPUs with data parallel configurations.

### RTX 3090 Specific

No direct H100-equivalent benchmarks exist for RTX 3090. Key constraints on 24GB:
- 7B/8B models fit comfortably with room for KV cache (~4-6GB for model in 4-bit, ~16GB in 16-bit)
- 14B models require 4-bit quantization to fit (8-bit won't leave enough KV cache room)
- Multi-LoRA adds ~10-50MB per adapter — negligible on 24GB
- vLLM's PagedAttention reduces memory fragmentation by 50%+, critical for maximizing throughput on limited VRAM
- `--max-model-len` must be set manually (default may OOM on 3090); ~4000 tokens recommended for single 3090

### Latency

| Metric | SGLang | vLLM |
|--------|--------|------|
| TTFT (p50) | Lower under high concurrency | Competitive with tuned batching |
| TPOT | Lower overall | Higher |
| Agent workloads (p50) | <400ms for schema-constrained hops | Higher |

**Important caveat**: These benchmarks are on H100s. The throughput gap may narrow on RTX 3090s where memory bandwidth (not compute) is the bottleneck, since both engines will be memory-bandwidth-bound on consumer GPUs.

### Verdict: Throughput

**SGLang wins** with ~29% higher throughput on datacenter GPUs. The gap likely narrows on RTX 3090 but SGLang still has the edge. However, for this use case, the throughput difference may matter less than LoRA management maturity.

---

## 3. Structured Output / Tool-Calling Performance

### SGLang's Advantages

**RadixAttention**: Automatically reuses KV cache across requests via a radix tree. Cache hit rates are dramatically higher:

| Workload Type | vLLM PagedAttention | SGLang RadixAttention |
|---------------|--------------------|-----------------------|
| Few-shot learning | 15-25% hit rate | 85-95% hit rate |
| Multi-turn chat | 10-20% hit rate | 75-90% hit rate |
| Code analysis | 5-15% hit rate | 60-80% hit rate |

This matters significantly for tool-calling workloads where the system prompt + tool definitions are repeated across requests.

**Compressed Finite State Machine**: SGLang's constrained decoding runs JSON decoding 3x faster vs unconstrained generation with post-processing.

### Guided Decoding Benchmarks (H100, Qwen3-8B/32B)

From [SqueezeBits benchmark](https://blog.squeezebits.com/guided-decoding-performance-vllm-sglang):

| Backend | Repetitive Schema | Dynamic Schema | Notes |
|---------|-------------------|----------------|-------|
| XGrammar on SGLang | Best throughput | Good, 2.21% invalid JSON | SGLang overlaps mask generation with inference |
| XGrammar on vLLM | Good throughput | Good, 2.21% invalid JSON | More overhead than SGLang |
| LLGuidance on SGLang | Good | Best accuracy (0.12% invalid) | Better for unique schemas |
| LLGuidance on vLLM | Significant degradation | OK | CPU bottleneck under load |

**Key finding**: SGLang minimizes guided decoding overhead through CPU-GPU parallelization, while vLLM shows significant throughput degradation with guided decoding enabled.

### Verdict: Structured Output

**SGLang wins decisively.** RadixAttention's cache reuse for repeated tool definitions + faster constrained decoding + lower overhead for guided generation. For a tool-calling-heavy platform, this is SGLang's strongest advantage.

---

## 4. OpenAI API Compatibility

### vLLM

**Completeness**: Most comprehensive OpenAI-compatible implementation in the ecosystem.

| Feature | Status |
|---------|--------|
| `/v1/chat/completions` | Full support |
| `/v1/completions` | Full support |
| `tool_choice: auto` | Supported |
| `tool_choice: required` | Supported |
| `tool_choice: none` | Supported |
| Named function calling | Supported |
| Parallel tool calls | Supported (model-dependent) |
| Streaming with tool calls | Supported |
| `strict` mode | Accepted but **no-op** (not enforced) |

**Supported models for auto tool calling**: Hermes, Mistral, Llama3, IBM Granite, InternLM, Jamba, xLAM, Qwen, DeepSeek-V3, and many more (20+ model families).

**Configuration required**:
- `--enable-auto-tool-choice` flag
- `--tool-call-parser <parser>` (e.g., `hermes`, `mistral`, `llama3_json`)

### SGLang

| Feature | Status |
|---------|--------|
| `/v1/chat/completions` | Supported |
| Tool calling (Chat API) | Supported but **unstable** |
| Tool calling (Responses API) | Only built-in tools work; custom tools fail |
| `tool_choice` options | Partial |
| Parallel tool calls | Unclear/undocumented |
| Streaming with tool calls | Unclear |

**Per GitHub issue #10038**: Users report tool calling is "sometimes possible but most of the time not." Community feedback: vLLM 0.11.0 worked "perfectly" for tool calling while SGLang had persistent issues.

### Verdict: OpenAI API / Tool Calling

**vLLM wins decisively.** Mature, well-documented tool calling with broad model support and all `tool_choice` modes. SGLang's tool calling is experimental and unreliable. For a platform where users expect OpenAI API drop-in compatibility with function calling, this is critical.

---

## 5. Request Logging / Observability

### vLLM

| Capability | Details |
|------------|---------|
| **OpenTelemetry** | Native support. Distributed tracing with spans for request lifecycle. Integrates with Jaeger, Langfuse, Grafana. |
| **Prometheus** | Built-in metrics endpoint. Request counts, latencies, queue depths, GPU utilization. |
| **ASGI Middleware** | `--middleware` flag accepts custom middleware classes/functions. Full request/response interception. |
| **Request logging** | `--enable-log-requests` + `--enable-log-outputs` captures prompts and completions to logs. |
| **Custom logging** | FastAPI/Starlette middleware — write custom middleware to capture every prompt/completion to database or file. |

**For flywheel data capture**: Write a custom ASGI middleware that intercepts request bodies (prompts, tools) and response bodies (completions, tool calls), serialize to JSONL for training data. This is well-supported and documented.

### SGLang

| Capability | Details |
|------------|---------|
| **OpenTelemetry** | Supported via OTLP export. Distributed tracing for HTTP and gRPC. |
| **Prometheus** | 40+ metrics across HTTP, router, worker, circuit breaker layers. Default port 29000. |
| **Request tracking** | `x-request-id` header, configurable via `--request-id-headers`. |
| **Structured logging** | `--log-dir` file sink, `--log-level` (debug/info/warn/error). |
| **WASM middleware** | Gateway supports WebAssembly middleware modules for custom processing (auth, billing, logging). Sandboxed. |

**For flywheel data capture**: SGLang's WASM middleware is more isolated but less flexible than vLLM's Python ASGI middleware. You'd need to write middleware in a WASM-compatible language or use the structured logging + post-processing approach.

### Verdict: Observability

**Roughly equal, with different trade-offs.** vLLM's Python ASGI middleware is easier to write custom request/response capture. SGLang has more built-in metrics (40+) and a more mature observability architecture. For the flywheel use case (capture every request/response), vLLM's ASGI middleware is simpler to implement.

---

## 6. Hot-Swap / Dynamic LoRA Loading

### vLLM

**Mechanism**:
1. Set `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`
2. Start server with `--enable-lora --max-loras N --max-lora-rank R`
3. Load: `POST /v1/load_lora_adapter {"lora_name": "user_v2", "lora_path": "/path/to/adapter"}`
4. Replace: `POST /v1/load_lora_adapter {"lora_name": "user_v2", "lora_path": "/new/path", "load_inplace": true}`
5. Unload: `POST /v1/unload_lora_adapter {"lora_name": "user_v2"}`

**Key properties**:
- Non-blocking — inference continues during load
- `load_inplace` replaces adapter weights atomically
- Filesystem resolver can auto-discover new adapters from directory
- Used in production for continuous RL training loops (train → swap → serve)

### SGLang

**Mechanism**:
1. Start server with `--enable-lora` + adapter config
2. Load: `POST /load_lora_adapter {"lora_name": "...", "lora_path": "..."}`
3. Unload: `POST /unload_lora_adapter {"lora_name": "..."}`

**Key properties**:
- **BLOCKING** — server cannot handle requests during load/unload
- No `load_inplace` equivalent
- No filesystem resolver (PR #15454 in progress)
- Async loading requested (issue #8162) but not yet implemented

### Verdict: Hot-Swap

**vLLM wins clearly.** Non-blocking load/unload, in-place replacement, automatic discovery. SGLang's blocking behavior means downtime during adapter updates — unacceptable for a multi-tenant production platform.

---

## 7. GPU Memory Management on 24GB RTX 3090

### Model Sizing on 24GB

| Model Size | Precision | VRAM Usage | KV Cache Room | Feasible? |
|-----------|-----------|-----------|---------------|-----------|
| 3B | FP16 | ~6GB | ~18GB | Yes, comfortable |
| 7B/8B | FP16 | ~16GB | ~8GB | Yes, limited context |
| 7B/8B | INT4 (AWQ/GPTQ) | ~4-6GB | ~18GB | Yes, recommended |
| 14B | FP16 | ~28GB | N/A | No, doesn't fit |
| 14B | INT4 | ~8-10GB | ~14GB | Yes, with quantization |

### vLLM Memory Management

- **PagedAttention**: Reduces KV cache fragmentation by 50%+, critical for maximizing concurrent requests on 24GB
- `--max-model-len`: Must set manually on 3090 (default often OOMs). ~4000 tokens for single 3090.
- `--gpu-memory-utilization`: Default 0.9, adjustable (e.g., 0.85 to leave room for LoRA adapters)
- Multi-GPU: Tensor parallelism across 2x 3090 gives 48GB total but NOT unified memory — each GPU has 24GB with inter-GPU communication overhead
- Known issue: RTX 3090 uses PCIe (not NVLink by default), so multi-GPU communication is slower than datacenter GPUs

### SGLang Memory Management

- Similar PagedAttention-based approach
- `csgmv` LoRA backend claimed 20-80% faster than Triton for LoRA operations
- LoRA overlap loading can pre-stage adapters from CPU to GPU asynchronously
- Pinned adapter slots prevent eviction of frequently-used adapters

### Verdict: Memory Management

**Roughly equal.** Both use similar memory management techniques. SGLang's `csgmv` backend may give a slight edge for LoRA-heavy workloads. vLLM has more documentation and community experience with RTX 3090 specifically. For 24GB, the key is quantization (INT4) for 7B+ models to leave room for KV cache and multiple LoRA adapters.

---

## 8. Maturity & Ecosystem

### vLLM

| Aspect | Status |
|--------|--------|
| **First release** | ~2023 |
| **GitHub stars** | 50k+ |
| **Community** | Large, mature. Dedicated forum, extensive Stack Overflow answers |
| **Production users** | Widely deployed (Anyscale, various enterprises) |
| **Cloud integrations** | Ray, Kubernetes, major cloud providers |
| **Documentation** | Comprehensive, well-maintained |
| **Release cadence** | Regular releases, active development |
| **Production stack** | `vllm-project/production-stack` — Kubernetes-native deployment |
| **Known issues** | `strict` mode not enforced, some streaming edge cases with tool calls |

### SGLang

| Aspect | Status |
|--------|--------|
| **First release** | ~2024 |
| **Origin** | LMSYS (Chatbot Arena team at UC Berkeley) |
| **GitHub stars** | 30k+ |
| **Community** | Smaller but engaged, responsive maintainers |
| **Production users** | Growing but less documented |
| **Cloud integrations** | Fewer out-of-box integrations |
| **Documentation** | Thinner, some gaps |
| **Release cadence** | Very active development (Q1 2026 roadmap published) |
| **Known issues** | Tool calling unstable, LoRA hot-swap blocking, filesystem resolver missing |

### Verdict: Maturity

**vLLM wins.** More battle-tested in production, larger community, better documentation, more integrations. SGLang is catching up fast and has strong academic backing, but for production infrastructure today, vLLM is the safer bet.

---

## 9. Migration Path

### API Compatibility

Both implement OpenAI-compatible APIs. The core endpoints (`/v1/chat/completions`, `/v1/completions`, `/v1/models`) are compatible. Client code using the OpenAI Python SDK should work with either backend with minimal changes (just the `base_url`).

### Differences That Affect Migration

| Aspect | vLLM | SGLang | Migration Impact |
|--------|------|--------|-----------------|
| LoRA loading endpoint | `/v1/load_lora_adapter` | `/load_lora_adapter` | Minor path change |
| Tool call parser config | `--tool-call-parser` | Different mechanism | Config change |
| Middleware | Python ASGI | WASM gateway | Rewrite middleware |
| Metrics | Prometheus (different metric names) | Prometheus (different names) | Update dashboards |
| Server startup flags | Different flag names | Different flag names | Update deploy scripts |

### Migration Difficulty: Low-Medium

- **Client-side**: Near-zero effort (OpenAI SDK compatible)
- **Server-side**: Medium effort (different CLI flags, middleware rewrite, monitoring updates)
- **LoRA management**: Medium effort (different APIs, different loading semantics)
- **Observability**: Medium effort (different metric names, different tracing setup)

### Recommended Strategy

Start with vLLM. Abstract the inference backend behind a thin proxy/gateway layer from day one:
1. Clients hit your API gateway (not vLLM directly)
2. Gateway handles auth, logging, routing
3. Gateway proxies to vLLM (or SGLang later)
4. LoRA management logic in your platform code, not coupled to backend API

This makes the backend swappable with minimal client impact.

---

## Comparison Matrix

| Criterion | vLLM | SGLang | Winner | Weight (for this use case) |
|-----------|------|--------|--------|---------------------------|
| **Multi-LoRA serving** | Mature, filesystem resolver, non-blocking | Functional, blocking load, no auto-discovery | vLLM | Critical |
| **Throughput** | ~12,500 tok/s (H100) | ~16,200 tok/s (H100) | SGLang | Medium |
| **Tool calling** | Stable, 20+ model families | Experimental, unstable | vLLM | Critical |
| **Structured output** | Good (XGrammar) | Excellent (RadixAttention + FSM) | SGLang | High |
| **OpenAI API compat** | Most complete | Good but gaps in tool calling | vLLM | Critical |
| **Observability** | OTel + ASGI middleware | OTel + WASM + 40+ metrics | Tie | High |
| **LoRA hot-swap** | Non-blocking, load_inplace | Blocking, no in-place | vLLM | Critical |
| **24GB GPU efficiency** | Well-documented, PagedAttention | Similar, csgmv LoRA backend | Tie | High |
| **Maturity** | 3+ years, large community | ~2 years, growing fast | vLLM | High |
| **Migration path** | N/A | Low-medium effort from vLLM | N/A | Low (plan for it) |

---

## Recommendation

### Start with vLLM

For a hosted fine-tuning platform where users expect reliable tool-calling, zero-downtime adapter updates, and OpenAI API drop-in compatibility:

1. **vLLM** is the right choice for initial launch
2. Its LoRA management is production-ready (non-blocking, auto-discovery, in-place replacement)
3. Tool calling is stable across 20+ model families
4. Larger community means faster answers to RTX 3090-specific issues

### Plan for SGLang Migration

SGLang's throughput advantage (29%) and structured output performance (RadixAttention) are compelling. Plan the architecture to enable future migration:

1. Abstract the inference backend behind your own API gateway
2. Keep LoRA management logic in your platform, not coupled to backend specifics
3. Monitor SGLang's LoRA hot-swap progress (async loading issue #8162, filesystem resolver PR #15454)
4. Monitor SGLang's tool-calling maturity
5. Re-evaluate in 6 months (Q3 2026) when SGLang's LoRA and tool-calling gaps may be resolved

### Architecture Implication

The abstraction layer is not optional — it's required regardless of backend choice:
- Auth/multi-tenancy routing
- Request/response logging for flywheel
- LoRA adapter lifecycle management
- Rate limiting, billing hooks
- Backend-agnostic client API

This same layer makes backend swapping nearly transparent.

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| vLLM throughput insufficient for load | Medium | High | Quantize models (INT4), add GPU, or migrate to SGLang |
| SGLang LoRA features don't mature | Low | Medium | vLLM remains viable long-term |
| RTX 3090 memory too constrained for 14B | Medium | Medium | INT4 quantization, or limit to 7B/8B models |
| Tool calling breaks with specific model | Medium | Medium | Test each model family; maintain parser configs |
| vLLM deprecates features we depend on | Low | Low | Active project with stable API versioning |

---

## Sources

- [vLLM LoRA Adapters Documentation](https://docs.vllm.ai/en/latest/features/lora/)
- [SGLang LoRA Serving Documentation](https://docs.sglang.io/advanced_features/lora.html)
- [vLLM Tool Calling Documentation](https://docs.vllm.ai/en/latest/features/tool_calling/)
- [SGLang Tool Calling Issue #10038](https://github.com/sgl-project/sglang/issues/10038)
- [SGLang Dynamic LoRA Loading Issue #15417](https://github.com/sgl-project/sglang/issues/15417)
- [SGLang Async LoRA Loading Issue #8162](https://github.com/sgl-project/sglang/issues/8162)
- [PremAI: vLLM vs SGLang vs LMDeploy Benchmarks 2026](https://blog.premai.io/vllm-vs-sglang-vs-lmdeploy-fastest-llm-inference-engine-in-2026/)
- [SqueezeBits: Guided Decoding Performance on vLLM and SGLang](https://blog.squeezebits.com/guided-decoding-performance-vllm-sglang)
- [Unsloth: LoRA Hot Swapping Guide for vLLM](https://unsloth.ai/docs/basics/inference-and-deployment/vllm-guide/lora-hot-swapping-guide)
- [SGLang NeurIPS 2024 Paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/724be4472168f31ba1c9ac630f15dec8-Paper-Conference.pdf)
- [S-LoRA: Serving Thousands of Concurrent LoRA Adapters](https://arxiv.org/pdf/2311.03285)
- [vLLM Production Deployment Guide 2026](https://www.sitepoint.com/vllm-production-deployment-guide-2026/)
- [SGLang Development Roadmap Q1 2026](https://github.com/sgl-project/sglang/issues/12780)
- [Yotta Labs: vLLM vs SGLang 2026](https://www.yottalabs.ai/post/vllm-vs-sglang-which-inference-engine-should-you-use-in-2026)
- [Multi-LoRA Serving Performance (arXiv 2025)](https://arxiv.org/html/2505.03756v1)
- [vLLM RTX 3090 Dual GPU Setup Guide](https://thamizhelango.medium.com/setting-up-vllm-with-dual-nvidia-rtx-3090-gpus-a-complete-guide-ab2235cef256)
- [vLLM RTX 3090 Performance Benchmarks (4x 3090)](http://himeshp.blogspot.com/2025/03/vllm-performance-benchmarks-4x-rtx-3090.html)
