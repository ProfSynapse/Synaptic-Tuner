# Implementation Plan: vLLM Inference Plugins — Config-Driven Decoding Strategies

> Generated from DoLa research session on 2026-04-10
> Research: [DoLa: Decoding by Contrasting Layers (ICLR 2024)](https://arxiv.org/abs/2309.03883)
> Status: PENDING APPROVAL

<!-- Status Lifecycle:
     PENDING APPROVAL → APPROVED → IN_PROGRESS → IMPLEMENTED
                    ↘ SUPERSEDED (if replaced by newer plan)
                    ↘ BLOCKED (if unresolved conflicts)
-->

## Summary

Build a **config-driven inference plugin system** for vLLM that allows decoding strategies (DoLa, contrastive search, activation steering, etc.) to be toggled on/off via YAML config. The first plugin is DoLa — contrastive layer decoding for improved factuality — but the architecture is generalized so future strategies slot in without structural changes.

The system leverages [IBM vLLM Hook](https://github.com/IBM/vLLM-Hook) (March 2026) for intermediate layer access and vLLM's native custom LogitsProcessor API for final-logit-only strategies. Both hook into the existing `services/proxy/` infrastructure without forking vLLM.

### Why Not Just HuggingFace `model.generate()`?

HF's `dola_layers` parameter works for local inference but **does not help** with production vLLM serving. vLLM's `LogitsProcessor` API only receives final logits — DoLa needs intermediate layer hidden states. The IBM vLLM Hook plugin provides exactly this capability via PyTorch forward hooks registered on model layers.

### Key Principles

1. **Config-driven** — YAML configs under `configs/inference/` control which plugins are active and their parameters
2. **Generalized** — The plugin interface supports both "layer-access" strategies (DoLa, activation steering) and "logits-only" strategies (repetition penalty, temperature scaling)
3. **Composable** — Multiple plugins can be stacked (e.g., DoLa + repetition penalty)
4. **No vLLM fork** — All hooks are external via IBM vLLM Hook or native LogitsProcessor API
5. **Observable** — Plugin activity is logged through the existing proxy logging pipeline

---

## Dependency DAG

```
┌─────────────────────┐
│  1. Plugin Framework │
│  (config + registry) │
└──────────┬──────────┘
           │
    ┌──────┴──────┐
    │             │
┌───▼───┐   ┌────▼────┐
│ 2A    │   │ 2B      │
│ DoLa  │   │ Logits  │
│ Plugin│   │ Plugins │
└───┬───┘   └────┬────┘
    │             │
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │ 3. Proxy    │
    │ Integration │
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │ 4. Eval     │
    │ A/B Testing │
    └─────────────┘
```

### Dependency Edges

| From | To | Reason |
|------|----|--------|
| **1** → **2A, 2B** | Plugin framework defines the interface that DoLa and logits plugins implement |
| **2A, 2B** → **3** | Proxy must know how to pass plugin config to vLLM |
| **3** → **4** | Evaluator A/B testing needs the proxy to support plugin toggling per-request |

### Parallelism

| Can Run in Parallel | Why |
|---------------------|-----|
| **2A** ∥ **2B** | Independent plugin implementations against the same interface |

---

## Stage 1: Plugin Framework (Config + Registry)

**Effort**: Medium
**Risk**: Low

### Design

A plugin is a Python class implementing a defined interface. Two plugin categories:

1. **LayerHookPlugin** — Needs intermediate layer access (via IBM vLLM Hook). Registers PyTorch forward hooks on specified layers and modifies final logits before sampling.
2. **LogitsPlugin** — Only needs final logits (via vLLM native `LogitsProcessor`). Simpler, no extra dependencies.

### Config Schema: `configs/inference/default.yaml`

```yaml
# Inference plugin configuration
# Each plugin can be enabled/disabled independently

plugins:
  # Layer-access plugins (require IBM vLLM Hook)
  dola:
    enabled: false
    # "low" = lower half of layers, "high" = upper half, or explicit list
    premature_layers: "high"
    # Final layer index (null = auto-detect from model config)
    mature_layer: null
    # Suppress low-probability tokens after contrasting
    relative_top: 0.1
    # JSD threshold for dynamic layer selection (0 = always contrast)
    jsd_threshold: 0.0

  activation_steering:
    enabled: false
    # Placeholder for future activation steering plugin
    vectors_path: null
    scale: 1.0
    target_layers: []

  # Logits-only plugins (native vLLM LogitsProcessor)
  repetition_penalty:
    enabled: false
    penalty: 1.1
    window: 64

  min_p:
    enabled: false
    threshold: 0.05

# IBM vLLM Hook settings (only needed if any layer-access plugin is enabled)
vllm_hook:
  enabled: false  # Auto-enabled when any LayerHookPlugin is active
  registry_port: 9090
  log_activations: false
  activation_log_dir: "scratch/activations"

# Global inference overrides (applied after plugins)
inference:
  temperature: null      # null = use request value
  top_p: null
  max_tokens: null
  seed: null
```

### Config Schema: Per-model overrides `configs/inference/profiles/`

```yaml
# configs/inference/profiles/factual.yaml
# Profile for factuality-critical inference (e.g., tool-call generation)

extends: default
plugins:
  dola:
    enabled: true
    premature_layers: "high"
    relative_top: 0.1
```

```yaml
# configs/inference/profiles/creative.yaml
# Profile for creative/diverse generation (e.g., essay writing)

extends: default
plugins:
  dola:
    enabled: false
  min_p:
    enabled: true
    threshold: 0.02
inference:
  temperature: 0.9
```

### Files to Create

| File | Purpose |
|------|---------|
| `shared/inference/__init__.py` | Package init |
| `shared/inference/config.py` | `InferencePluginConfig` dataclass, `load_inference_config()` |
| `shared/inference/registry.py` | Plugin registry — discovers and loads plugins by name |
| `shared/inference/base.py` | `BaseLayerHookPlugin` and `BaseLogitsPlugin` ABCs |
| `shared/inference/loader.py` | Reads YAML, resolves profile inheritance (`extends:`), instantiates plugins |

### Config Dataclass: `shared/inference/config.py`

```python
@dataclass
class DoLaConfig:
    enabled: bool = False
    premature_layers: str | list[int] = "high"
    mature_layer: int | None = None
    relative_top: float = 0.1
    jsd_threshold: float = 0.0

@dataclass
class VLLMHookConfig:
    enabled: bool = False
    registry_port: int = 9090
    log_activations: bool = False
    activation_log_dir: str = "scratch/activations"

@dataclass
class InferencePluginConfig:
    dola: DoLaConfig = field(default_factory=DoLaConfig)
    # Future plugins add fields here
    vllm_hook: VLLMHookConfig = field(default_factory=VLLMHookConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "InferencePluginConfig": ...

    @classmethod
    def from_profile(cls, profile_name: str) -> "InferencePluginConfig": ...
```

### Plugin ABCs: `shared/inference/base.py`

```python
class BaseLayerHookPlugin(ABC):
    """Plugin that needs intermediate layer access via vLLM Hook."""

    @abstractmethod
    def target_layers(self, model_config: dict) -> list[int]:
        """Return layer indices to hook."""

    @abstractmethod
    def on_layer_output(self, layer_idx: int, hidden_states: Tensor) -> None:
        """Called when a hooked layer produces output."""

    @abstractmethod
    def modify_logits(self, final_logits: Tensor, lm_head: Module) -> Tensor:
        """Modify final logits before sampling using captured layer outputs."""


class BaseLogitsPlugin(ABC):
    """Plugin that only needs final logits (vLLM LogitsProcessor)."""

    @abstractmethod
    def __call__(self, token_ids: list[int], logits: Tensor) -> Tensor:
        """Modify logits in-place. Standard LogitsProcessor interface."""
```

---

## Stage 2A: DoLa Plugin

**Effort**: Medium
**Risk**: Medium (depends on IBM vLLM Hook compatibility with target vLLM version)

### Algorithm

Per-token during generation:

1. **Capture** hidden states from candidate premature layers (registered via `target_layers()`)
2. **Project** each premature hidden state through the LM head to get premature logits
3. **Select** the premature layer with maximum Jensen-Shannon Divergence from the mature layer
4. **Contrast**: `output_logits = mature_logits.log_softmax() - selected_premature_logits.log_softmax()`
5. **Filter**: Apply `relative_top` threshold to suppress noise tokens

### Layer Selection Heuristics

| `premature_layers` value | Behavior | Best for |
|--------------------------|----------|----------|
| `"low"` | Lower half of model layers (e.g., 0-15 for 32-layer model) | Long-form reasoning (GSM8K, StrategyQA) |
| `"high"` | Upper half (e.g., 16-31) | Short factual answers (TruthfulQA) |
| `[2, 4, 6, 8, 10, 12, 14]` | Explicit layer indices | Custom experimentation |

### Files to Create

| File | Purpose |
|------|---------|
| `shared/inference/plugins/__init__.py` | Plugin package |
| `shared/inference/plugins/dola.py` | `DoLaPlugin(BaseLayerHookPlugin)` — core DoLa implementation |
| `shared/inference/hooks/vllm_hook_bridge.py` | Bridge between our plugin interface and IBM vLLM Hook's Worker/Registry API |

### `shared/inference/plugins/dola.py` — Core Logic

```python
class DoLaPlugin(BaseLayerHookPlugin):
    """DoLa: Decoding by Contrasting Layers (Chuang et al., ICLR 2024).

    Contrasts mature (final) layer logits against premature (earlier) layer
    logits to amplify factual knowledge and suppress hallucinations.
    """

    def __init__(self, config: DoLaConfig):
        self.config = config
        self._captured_states: dict[int, Tensor] = {}

    def target_layers(self, model_config: dict) -> list[int]:
        num_layers = model_config["num_hidden_layers"]
        if self.config.premature_layers == "low":
            return list(range(0, num_layers // 2, 2))
        elif self.config.premature_layers == "high":
            return list(range(num_layers // 2, num_layers - 1, 2))
        else:
            return list(self.config.premature_layers)

    def on_layer_output(self, layer_idx: int, hidden_states: Tensor) -> None:
        self._captured_states[layer_idx] = hidden_states

    def modify_logits(self, final_logits: Tensor, lm_head: Module) -> Tensor:
        if not self._captured_states:
            return final_logits

        mature_log_probs = final_logits.log_softmax(dim=-1)

        # Compute JSD for each candidate premature layer
        best_jsd = -1.0
        best_premature_log_probs = None

        for layer_idx, hidden in self._captured_states.items():
            premature_logits = lm_head(hidden)
            premature_log_probs = premature_logits.log_softmax(dim=-1)

            # Jensen-Shannon Divergence
            m = 0.5 * (mature_log_probs.exp() + premature_log_probs.exp())
            jsd = 0.5 * (
                (mature_log_probs.exp() * (mature_log_probs - m.log())).sum(-1)
                + (premature_log_probs.exp() * (premature_log_probs - m.log())).sum(-1)
            )

            if jsd.mean().item() > best_jsd:
                best_jsd = jsd.mean().item()
                best_premature_log_probs = premature_log_probs

        if best_premature_log_probs is None or best_jsd < self.config.jsd_threshold:
            return final_logits

        # Contrastive logits
        diff = mature_log_probs - best_premature_log_probs

        # Relative top filtering
        if self.config.relative_top > 0:
            max_logit = diff.max(dim=-1, keepdim=True).values
            threshold = max_logit + torch.log(
                torch.tensor(self.config.relative_top, device=diff.device)
            )
            diff[diff < threshold] = float("-inf")

        self._captured_states.clear()
        return diff
```

### IBM vLLM Hook Bridge: `shared/inference/hooks/vllm_hook_bridge.py`

```python
class DoLaWorker:
    """IBM vLLM Hook Worker that bridges our plugin interface.

    Registers as a vLLM Hook Worker, captures intermediate layer activations,
    and feeds them to the configured LayerHookPlugin.
    """

    def __init__(self, plugin: BaseLayerHookPlugin, model_config: dict):
        self.plugin = plugin
        self.target_layer_indices = plugin.target_layers(model_config)

    def register_hooks(self, model: Module) -> list:
        """Register PyTorch forward hooks on target layers."""
        hooks = []
        for name, module in model.named_modules():
            layer_idx = self._extract_layer_index(name)
            if layer_idx in self.target_layer_indices:
                hook = module.register_forward_hook(
                    lambda mod, inp, out, idx=layer_idx:
                        self.plugin.on_layer_output(idx, out[0])
                )
                hooks.append(hook)
        return hooks

    def post_forward(self, final_logits: Tensor, lm_head: Module) -> Tensor:
        """Called after model forward, before sampling."""
        return self.plugin.modify_logits(final_logits, lm_head)
```

---

## Stage 2B: Logits-Only Plugins (Examples)

**Effort**: Small
**Risk**: Low

These use vLLM's native `LogitsProcessor` API — no IBM vLLM Hook needed.

### Files to Create

| File | Purpose |
|------|---------|
| `shared/inference/plugins/min_p.py` | Min-P sampling plugin |
| `shared/inference/plugins/repetition.py` | Windowed repetition penalty |

These serve as examples and validate the plugin interface works for simple cases.

---

## Stage 3: Proxy Integration

**Effort**: Medium
**Risk**: Low

### Design

The proxy (`services/proxy/app.py`) already forwards requests to vLLM. We extend it to:

1. Read the inference plugin config at startup
2. Initialize plugins (register hooks via IBM vLLM Hook if needed)
3. Optionally accept per-request plugin overrides via an `x-inference-profile` header

### Files to Modify

| File | Change |
|------|--------|
| `services/proxy/config.py` | Add `inference_config_path: str` field to `ProxyConfig` |
| `services/proxy/app.py` | Load inference config at startup; initialize plugin registry; apply plugins |

### Per-Request Profile Override

```
POST /v1/chat/completions
x-inference-profile: factual
```

The proxy resolves `factual` → `configs/inference/profiles/factual.yaml`, enabling DoLa for that request. If no header, the default profile applies.

### Config Addition to `configs/flywheel/default.yaml`

```yaml
# Inference plugins
inference_config_path: configs/inference/default.yaml
```

---

## Stage 4: Evaluator A/B Testing

**Effort**: Small
**Risk**: Low

### Design

Allow the Evaluator to run the same scenario suite with different inference profiles, producing side-by-side comparison reports.

### Config Addition to `Evaluator/config/eval_run.yaml`

```yaml
model:
  backend: vllm
  name: my-model
  inference:
    # Compare two profiles
    ab_test:
      control: default        # configs/inference/profiles/default.yaml
      treatment: factual      # configs/inference/profiles/factual.yaml
```

### Files to Modify

| File | Change |
|------|--------|
| `Evaluator/config.py` | Add `ab_test` field to model inference config |
| `Evaluator/runner.py` | Run scenario suite twice (control + treatment), output comparison |

---

## Skill Integration

### New Skill Section in `.skills/fine-tuning/SKILL.md`

Add an "Inference Plugins" section covering:
- Available plugins and what they do
- Config reference for `configs/inference/`
- How to create a new inference profile
- How to A/B test with the Evaluator
- How to write a new plugin

### Config Discovery via CLI

```bash
./run.sh list plugins         # List available inference plugins
./run.sh list profiles        # List inference profiles
tuner inference --profile factual --prompt "What is..."  # Quick test
```

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| IBM vLLM Hook incompatible with target vLLM version | Medium | High | Pin compatible versions; fallback to monkey-patching `model.forward()` like vLLM speculators |
| DoLa overhead degrades vLLM throughput under batching | Medium | Medium | Benchmark early; add `max_batch_size_for_dola` config to disable for high-concurrency |
| LM head projection cost per premature layer | Low | Low | Cache LM head weight; use `torch.no_grad()` for premature projections |
| Plugin interaction conflicts (two plugins modifying logits) | Low | Medium | Define plugin execution order in config; document composability constraints |

---

## Open Items

- [ ] Verify IBM vLLM Hook compatibility with vLLM version in use
- [ ] Benchmark DoLa overhead per token on representative model (7B, 14B)
- [ ] Test whether Unsloth's forward pass modifications interfere with PyTorch hooks
- [ ] Determine if vLLM's continuous batching interacts with per-request hook state
- [ ] Check `transformers-community/dola` source for any algorithmic improvements over original paper

---

## Phase Requirements

| Phase | Prerequisites | Deliverables | Validation |
|-------|---------------|--------------|------------|
| **1. Framework** | None | Config schema, plugin ABCs, registry, loader | Unit tests for config loading and plugin registration |
| **2A. DoLa** | Phase 1, IBM vLLM Hook installed | DoLa plugin, vLLM Hook bridge | Manual test: DoLa vs baseline on TruthfulQA-style prompts |
| **2B. Logits** | Phase 1 | min_p and repetition plugins | Unit tests for logit modification |
| **3. Proxy** | Phase 1, Phase 2A or 2B | Proxy reads config, applies plugins, supports profiles | Integration test: proxy serves with DoLa enabled |
| **4. Eval A/B** | Phase 3 | Evaluator A/B comparison mode | Side-by-side report generation |
