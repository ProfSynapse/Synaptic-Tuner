# LoRA Techniques Guide

Reference for LoRA variants, initialization methods, and configuration recipes available in the training pipeline.

---

## Quick Decision Guide

| Your Situation | Recommended Config | Template |
|---|---|---|
| Single GPU, VRAM-limited (24GB) | QLoRA + DoRA | `configs/qlora_dora.yaml` |
| Want full fine-tuning quality | High-rank + rsLoRA + all-linear | `configs/regret_free.yaml` |
| Small dataset, best data-efficient init | EVA (or EVA + DoRA) | `configs/eva.yaml` |
| Fast convergence, no quant overhead | PiSSA or OLoRA | `configs/pissa.yaml`, `configs/olora.yaml` |
| RL/GRPO (reasoning tasks) | Rank-1 to 8, all-linear | `configs/grpo_minimal.yaml` |
| 4-bit base, minimize quant error | LoftQ | `configs/loftq.yaml` |
| Drop-in quality boost, any method | DoRA | `configs/dora.yaml` |

---

## Techniques by Category

### Tier 1: Ready to Use (config flags only)

These work today via existing `LoraConfig` fields. Set the flags in your trainer's `config.yaml` under the `lora:` section.

#### DoRA (Weight-Decomposed LoRA)

Decomposes weight updates into **magnitude** and **direction** components separately. Matches the update structure of full fine-tuning more closely than standard LoRA.

```yaml
lora:
  use_dora: true
```

- **Status**: Config field exists in KTO and GRPO. SFT config_loader needs the field added.
- **VRAM**: ~1-2GB more than standard LoRA at same rank
- **When**: Drop-in quality improvement at any rank

#### rsLoRA (Rank-Stabilized LoRA)

Changes scaling from `alpha/r` to `alpha/sqrt(r)`. Stabilizes gradients at high ranks so you actually benefit from large r.

```yaml
lora:
  use_rslora: true
```

- **Status**: Config field exists in KTO and GRPO. SFT config_loader needs the field added.
- **VRAM**: No additional cost
- **When**: Practically mandatory at r=128+. Use with the regret-free recipe.

### Tier 2: Needs Unsloth Compatibility Check

These use PEFT's `init_lora_weights` parameter. Whether Unsloth's `FastLanguageModel.get_peft_model()` passes this through is the key question.

**How to test**: Add `init_lora_weights` to the LoRA config and run a `--dry-run`. If Unsloth silently ignores it, you'll need to fall back to raw PEFT for these variants.

#### PiSSA (Principal Singular Values Adaptation)

Initializes adapters from the **principal SVD components** of pretrained weights. Freezes the residual instead of the principal components (opposite of standard LoRA).

```yaml
lora:
  init_lora_weights: "pissa"
```

- **Benchmark**: Mistral-7B on GSM8K: PiSSA 72.86% vs LoRA 67.7%
- **When**: Want faster convergence without quantization

#### OLoRA (Orthonormal Low-Rank Adaptation)

Uses QR decomposition for orthonormal basis initialization. Better optimization landscape than random init.

```yaml
lora:
  init_lora_weights: "olora"
```

- **When**: Simpler alternative to EVA when you don't want data-dependent init

#### EVA (Explained Variance Adaptation)

**Data-driven** initialization — runs SVD on actual activations from your dataset to adapt rank allocation per layer. SOTA init method.

```yaml
lora:
  init_lora_weights: "eva"
```

- **Note**: Requires a small dataloader pass at init time (PEFT handles via `get_eva_state_dict()`)
- **Best combo**: EVA + DoRA can exceed full fine-tuning quality
- **When**: Small datasets where every example matters

#### LoftQ (Low-Rank Fine-Tuning with Quantization)

Quantization-aware SVD initialization that minimizes the error introduced by 4-bit quantization.

```yaml
lora:
  init_lora_weights: "loftq"
```

- **When**: Using `load_in_4bit: true` and want to minimize quality loss from quantization

### Tier 3: Different Architecture (not LoRA adapters)

#### Spectrum (Full-Layer Selective Training)

Not an adapter method. Computes Signal-to-Noise Ratio per layer using Random Matrix Theory, then trains the top X% most informative layers at full precision.

- **Status**: Supported in TRL's `SFTTrainer` natively, but the pipeline uses Unsloth's `FastLanguageModel` which would need a separate code path.
- **VRAM**: Higher than QLoRA (full-precision selected layers)
- **When**: Multi-GPU setups where VRAM isn't the bottleneck
- **Skip unless**: You move beyond single RTX 3090

#### GaLore (Gradient Low-Rank Projection)

Optimizer-level technique — projects gradients into a low-rank subspace for 65.5% less optimizer-state memory. Orthogonal to LoRA (can use both).

- **Status**: Not integrated. Would require optimizer changes.
- **When**: Want full-parameter learning on consumer GPU
- **Skip unless**: You want to experiment with full fine-tuning on a 3090

---

## The "LoRA Without Regret" Recipe

The most impactful recent finding (Schulman et al., Thinking Machines Lab, Sep 2025): LoRA can match full fine-tuning using ~67% of compute with three rules:

1. **All-linear targets** (`target_modules: "all-linear"`) — MLP layers matter more than most guides suggest
2. **High rank** (r=128-256 for SFT; r=1-8 for RL/GRPO)
3. **~10x higher learning rate** than full fine-tuning (the 1/r scaling makes optimal LR approximately rank-independent)

Template: `configs/regret_free.yaml`

```yaml
r: 256
lora_alpha: 16
learning_rate: 2e-3
use_rslora: true
target_modules: "all-linear"
warmup_ratio: 0.1
num_train_epochs: 2
batch_size: 2
gradient_accumulation_steps: 8
```

**Key insight for GRPO**: Rank-1 to 8 is remarkably effective for RL tasks. You don't need high rank for reward-driven optimization. Template: `configs/grpo_minimal.yaml`

---

## Config Templates

All templates are in `.skills/fine-tuning/configs/` and follow the same flat-YAML override pattern as the existing tier presets (`Trainers/sft/configs/tiers/`).

| Template | Technique | Key Settings |
|----------|-----------|-------------|
| `regret_free.yaml` | LoRA Without Regret | r=256, all-linear, rsLoRA, LR=2e-3 |
| `dora.yaml` | DoRA | use_dora=true, r=64 |
| `qlora_dora.yaml` | QLoRA + DoRA | use_dora=true, pair with load_in_4bit |
| `pissa.yaml` | PiSSA init | init_lora_weights="pissa" |
| `eva.yaml` | EVA init | init_lora_weights="eva" |
| `olora.yaml` | OLoRA init | init_lora_weights="olora" |
| `loftq.yaml` | LoftQ init | init_lora_weights="loftq" |
| `grpo_minimal.yaml` | Minimal RL LoRA | r=8, all-linear, rsLoRA |

**How to use**: These are reference configs, not wired into `--tier` yet. To apply:
1. Copy the relevant settings into your trainer's `config.yaml` under `lora:`
2. Or merge manually before a training run

**To wire as tiers**: Copy the template into `Trainers/{sft,kto}/configs/tiers/` and add the tier name to the `--tier` CLI choices. Requires adding `use_dora`, `use_rslora`, `init_lora_weights`, and `target_modules` to the `_tier_config_map` in the trainer's `train_*.py`.

---

## Integration Status

### What works today

| Feature | SFT | KTO | GRPO |
|---------|-----|-----|------|
| `use_dora` | needs config_loader field | config field exists, needs model_loader passthrough | works |
| `use_rslora` | needs config_loader field | config field exists, needs model_loader passthrough | works |
| `target_modules` list | works | works | works |
| `target_modules: "all-linear"` | needs testing | needs testing | needs testing |
| `init_lora_weights` | needs Unsloth compat check | needs Unsloth compat check | needs Unsloth compat check |

### What needs code changes to fully enable

1. **SFT config_loader.py**: Add `use_dora: bool = False` and `use_rslora: bool = False` to `LoRAConfig` dataclass
2. **SFT model_loader.py**: Pass `use_dora` and `use_rslora` to `FastLanguageModel.get_peft_model()`
3. **KTO model_loader.py**: Pass the existing config fields through `apply_lora_adapters()`
4. **SFT/KTO/GRPO train_*.py**: Add `use_dora`, `use_rslora`, `target_modules`, `init_lora_weights` to `_tier_config_map`
5. **Unsloth compatibility**: Test whether `FastLanguageModel.get_peft_model()` accepts `init_lora_weights` kwarg

### LoRA Surgery compatibility

The existing LoRA surgery operations (`configs/lora_surgery.yaml`) should work with DoRA/rsLoRA adapters since they operate on weight tensors directly. However:
- **DoRA**: `alpha_sweep` may need awareness of the magnitude/direction decomposition
- **SVD rank reduction**: Should work unchanged since it operates on the combined delta
- **DARE drop+rescale**: Works unchanged

---

## Sources

- LoRA Without Regret: https://thinkingmachines.ai/blog/lora/
- DoRA: https://arxiv.org/abs/2402.09353
- PiSSA: https://arxiv.org/abs/2404.02948
- EVA: https://arxiv.org/abs/2410.07170
- OLoRA: https://arxiv.org/abs/2406.01775
- LoftQ: https://arxiv.org/abs/2310.08659
- GaLore: https://arxiv.org/abs/2403.03507
- PEFT LoRA docs: https://github.com/huggingface/peft/blob/main/docs/source/developer_guides/lora.md
- Spectrum: https://huggingface.co/posts/anakin87/865363319225333
