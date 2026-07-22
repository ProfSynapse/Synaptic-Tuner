# SFT Training Reference

Supervised Fine-Tuning — teaches the model tool-calling format and behaviors from positive examples.

---

## CLI Flags

```bash
python train_sft.py [options]
```

### Model Selection
| Flag | Description | Default |
|------|-------------|---------|
| `--model-size {3b,7b,13b,20b}` | Use preset configuration | — |
| `--config PATH` | Load direct SFT YAML (`.yaml`/`.yml`) or legacy Python config | — |

### Complexity Tiers
| Flag | Description |
|------|-------------|
| `--tier {quick,standard,thorough}` | Use preset complexity tier (overrides individual training params) |

| Tier | LoRA Rank | LR | Epochs | Steps | Time | Use Case |
|------|-----------|------|--------|-------|------|----------|
| `quick` | r=8, alpha=16 | 5e-4 | 1 | 200 max | ~5 min | Rapid prototyping, idea validation |
| `standard` | r=64, alpha=128 | 2e-4 | 1 | — | ~30-60 min | Production training |
| `thorough` | r=128, alpha=256 | 1e-4 | 3 | — | ~2-4 hrs | Maximum quality, publication |

Tier configs: `Trainers/sft/configs/tiers/{quick,standard,thorough}.yaml`

Explicit CLI flags (e.g., `--learning-rate`) override tier defaults.

### Training Parameters
| Flag | Description | Default |
|------|-------------|---------|
| `--batch-size N` | Per-device batch size | 2 |
| `--gradient-accumulation N` | Gradient accumulation steps | 2 |
| `--learning-rate FLOAT` | Learning rate | 2e-4 |
| `--num-epochs N` | Number of epochs | 3 |
| `--max-steps N` | Max steps (overrides epochs) | — |
| `--max-seq-length N` | Max sequence length | 2048 |

### Dataset
| Flag | Description | Default |
|------|-------------|---------|
| `--dataset-name STR` | HuggingFace dataset name | config value |
| `--dataset-file STR` | Specific file in HF dataset | config value |
| `--local-file PATH` | Local JSONL file (overrides HF) | — |
| `--split-dataset` | Create train/validation split | false |

### Experiment Tracking
| Flag | Description | Default |
|------|-------------|---------|
| `--wandb` | Enable W&B logging | false |
| `--wandb-project STR` | W&B project name | — |
| `--wandb-run-name STR` | W&B run name | — |

### Utility
| Flag | Description |
|------|-------------|
| `--dry-run` | Setup only, don't train |
| `--resume-from-checkpoint PATH` | Resume from checkpoint |
| `--hf-token STR` | HuggingFace token (gated models) |
| `--no-dashboard` | Disable live dashboard, use table output |
| `--quiet` | Suppress verbose library logs |

---

## Key SFT Settings

### What Makes SFT Different
- **No reference model** — simpler than KTO/GRPO
- **Higher learning rate** (2e-4 vs KTO's 1e-6) — more aggressive learning
- **Multiple epochs** (3 vs KTO's 1) — learn patterns thoroughly
- **Positive examples only** — no True/False labels needed
- **Packing support** — 2.5-5x faster training

### Packing (Recommended)
When `packing: true` in config:
1. Multiple examples packed into single sequences
2. **2.5-5x faster** training due to better GPU utilization
3. Dataset auto-preprocessed with chat template

### Completion-Only Loss
When `completion_only_loss: true` (default):
- Loss computed only on assistant response tokens
- User prompt tokens ignored during training
- Prevents model from learning to generate user messages

### Configured Special Tokens (`model.tokenizer`, optional)

SFT can register any ordered list of special tokens without embedding their
meaning in trainer code. The feature is off when the block is absent or the
list is empty. The same block works in the direct SFT YAML and in a
`local-run` recipe:

```yaml
model:
  tokenizer:
    additional_special_tokens:
      - "<MODE_A>"
      - "<MODE_B>"
    existing_token_policy: error       # error | reuse
    initialization: mean_existing_rows # deterministic; currently the only policy
    train_new_embedding_rows: true
    train_new_lm_head_rows: true
    verify_tokenizer_roundtrip: true
    verify_adapter_roundtrip: true
    verify_merged_model_roundtrip: false # expensive smoke-only merge/save/fresh reload check
```

The list order is preserved. Empty strings and duplicates fail before the
model is mutated. `existing_token_policy: error` rejects vocabulary collisions;
`reuse` explicitly accepts them, but reused rows are never made trainable.
New rows are initialized from the float32 mean of the pre-existing vocabulary
rows before LoRA is applied.

Selective row training uses PEFT's native `trainable_token_indices`: the old
input/output rows remain frozen and do not enter AdamW weight decay or optimizer
state. Both untied `torch.nn.Embedding` + `torch.nn.Linear` vocabularies and
coherently configured tied vocabularies are supported. Unsupported module
types, incoherent tied-head flags, or a PEFT version without selective-token
support fail loudly. A current runtime with PEFT 0.18 or newer is required when
either row-training flag is active; recipes should pin that runtime in
`setup.pip` when the selected image does not already provide it.

After training, the final model directory contains the tokenizer, the PEFT
adapter (including selective rows), and `special_tokens_lineage.json`.
`verify_tokenizer_roundtrip` reloads the saved tokenizer locally and proves every
configured string still encodes to exactly its recorded single token ID.
`verify_adapter_roundtrip` reloads the saved PEFT adapter into the already-live
model under a temporary adapter name, compares the complete adapter state
(LoRA plus selective rows), rechecks row IDs and tied-head sharing, then removes
the temporary adapter and restores the original. It does not load a second copy
of the base model and is not a full base-weight reload check. Before this check,
the trainer explicitly re-saves the adapter with `save_embedding_layers=false`:
selective-token deltas are absolute replacement rows, so the artifact contains
no redundant full embedding or output-head weights. When loading this adapter
elsewhere, load the saved tokenizer and resize an unpadded base model to
`len(tokenizer)` before attaching the adapter; padded bases that already cover
the recorded token IDs must not be shrunk.
Only after the adapter-only save and requested verification pass does the
trainer write success lineage. Both `special_tokens_lineage.json` and
`training_lineage.json` bind the exact adapter config/state files by SHA-256,
record a deterministic adapter tensor manifest (keys, shapes, and dtypes), and
state that no full base vocabulary tensors are present. They also record token
strings, IDs, vocab sizes, policies, module names, tied-weight status, and
deterministic config/vocabulary SHA-256 hashes.

For a bounded same-model smoke, set `verify_merged_model_roundtrip: true` (or
pass `--verify-merged-model-roundtrip`). After the adapter-only and tokenizer
checks pass, the trainer writes a temporary `merged_16bit` model, reloads it
locally on CPU, rechecks every configured token ID, and compares the configured
input/output vocabulary rows after the deterministic save dtype conversion. It
then removes the temporary merged model and keeps only
`merged_model_roundtrip.json` plus the same report embedded in
`special_tokens_lineage.json`. The report is bound to validated original Hub
repository + full model revision evidence. Before the first final adapter save,
every live PEFT config is restored from any runtime cache/snapshot path to that
portable repository/revision pair; the saved `adapter_config.json` is re-read
and checked. The feature is off by default and never publishes.

The `local-run` handler forwards this SFT-only block as one JSON object through
`--tokenizer-config`. Recipes without the block emit no flag, and DPO/KTO ignore
a stray block rather than receiving an unsupported trainer argument.

### Auxiliary Readout Head (`aux_head`, optional)
An optional auxiliary scalar readout head that learns to predict a per-row
target from a base model's hidden state. It runs in two modes:

- **Phase A** (`freeze_base: true`, `lm_loss_weight: 0`): the base/LoRA is frozen
  and only the small head trains; the LM loss is not used.
- **Phase B** (`freeze_base: false`, `lm_loss_weight > 0`): the base stays
  unfrozen (the LoRA params PEFT left trainable are added to the optimizer as a
  second group at the trainer LR) and the head co-trains jointly with the LM
  loss — total loss is `lm_loss + lm_loss_weight * head_loss`.

The feature is **off by default** — when the `aux_head` block is absent or
`enabled: false`, the SFT path is byte-identical to a standard run. The Phase-A
path (frozen base, no LM term, `input_norm: none`) is itself byte-identical
whether or not the Phase-B code is present, because both branches gate on the
config values.

When enabled, every training row must carry a finite numeric target under the
column named by `target_field`; a missing, null, non-numeric, or non-finite
value fails loudly (there is no silent default). After training, the head is
written next to the model as a sidecar (`aux_head.safetensors` +
`aux_head_config.json`) so it can be reloaded for inference independently of the
base weights. The sidecar persists `input_norm`, so a head trained with
normalization reloads and infers identically.

```yaml
aux_head:
  enabled: true            # absent / false => feature fully off
  layer: 35                # required when enabled: which hidden_states index to read (0 = embeddings)
  token_position: last     # "last" (last non-pad) | "mean" | int | "end_of_prompt"
  target_field: target     # per-row dataset column carrying the scalar target
  loss: bce                # "bce" | "brier" (MSE on probability)
  head_type: linear        # "linear" | "mlp"
  hidden_dims: []          # hidden widths when head_type: mlp
  out_activation: sigmoid  # "sigmoid" (prob in [0,1]) | "identity"
  input_norm: none         # "none" (off; default) | "layernorm" (so the head trains at a normal LR)
  freeze_base: true        # Phase A = true; Phase B = false (co-train the unfrozen base)
  lm_loss_weight: 0.0      # Phase A = 0.0; Phase B > 0 (weight on the head loss in the joint sum)
  head_lr: null            # optional dedicated head LR; defaults to trainer LR
```

Two Phase-B knobs:

- **`token_position: end_of_prompt`** reads the last *prompt* token (the position
  right before generation) instead of the last completion token. At train time
  the prompt/completion boundary is recovered from the label mask (prompt tokens
  are masked to `-100`); a row with no prompt span (completion-first or empty
  completion) falls back to the last real token. At inference the input is
  prompt-only, so `end_of_prompt` and `last` coincide.
- **`input_norm: layernorm`** applies a `LayerNorm` to the pulled hidden state
  before the head, so a linear head trains at a normal LR on unnormalized
  activations instead of saturating.

### `prompt_render` (a `training` knob, paired with `end_of_prompt`)

`token_position: end_of_prompt` is only faithful when the row was tokenized so the
prompt ends exactly at the generation anchor. The default render
(`training.prompt_render: full_conversation`) renders the whole conversation and
derives the assistant-only mask by a prefix match — but many chat templates render
the assistant scaffold differently with vs. without `add_generation_prompt` (e.g.
one fewer newline), so the masked boundary token is NOT the generation anchor and
the head reads an off-anchor representation.

Set `training.prompt_render: prompt_completion` for a faithful boundary: the row's
`input_ids` are built from the `add_generation_prompt=True` prompt render followed
by the raw completion plus the tokenizer's derived terminal (`eos_token_id`), with
the prompt segment masked to `-100`. The prompt then ends exactly at the
generation anchor, so the existing `end_of_prompt` read is faithful.

```yaml
training:
  prompt_render: prompt_completion   # full_conversation (default) | prompt_completion
```

`prompt_render` lives on the `training` config (it replaces the SFT masking
region, so it is not an `aux_head` field), but it is grouped with the aux_head
knobs for launching (see below). It defaults to `full_conversation`, so every
existing recipe is byte-identical. When `aux_head.enabled` and
`token_position: end_of_prompt` are set while `prompt_render` is still
`full_conversation`, the trainer prints a one-line WARNING (not an error — the
combo is legitimate for single-turn / inference-shaped rows whose two renders
coincide).

Complete runnable examples live at
`Trainers/sft/configs/aux_head_example.yaml` (Phase A) and
`Trainers/sft/configs/aux_head_phase_b_example.yaml` (Phase B). `layer` has no
default — choosing the hidden-state index is a deliberate per-run decision, so an
enabled block without it fails fast.

### Launching aux_head (local-run recipe + direct CLI)

The `aux_head` block flows through **both** launch paths:

- **Direct trainer** (`cd Trainers/sft && python train_sft.py ...`): every field
  has a CLI flag — `--aux-head-enabled` / `--no-aux-head-enabled`,
  `--aux-head-layer`, `--aux-head-token-position`, `--aux-head-target-field`,
  `--aux-head-loss`, `--aux-head-head-type`, `--aux-head-out-activation`,
  `--aux-head-input-norm`, `--aux-head-freeze-base` / `--no-aux-head-freeze-base`,
  `--aux-head-lm-loss-weight`, `--aux-head-head-lr`, and `--aux-head-prompt-render`
  (which sets `training.prompt_render`). An unset flag never overrides the loaded
  config; a CLI value takes precedence over the YAML/preset config.
- **local-run recipe** (`python tuner.py local-run --job-config <recipe>.yaml`):
  put an `aux_head:` block (and, when using `end_of_prompt`, a
  `training.prompt_render: prompt_completion`) in the recipe; the runner forwards
  every field to the trainer's `--aux-head-*` flags automatically. Recipes with no
  `aux_head` block and no `training.prompt_render` emit zero new flags, so they
  stay byte-identical. aux_head forwarding is SFT-only.

---

## Training Workflow

1. **Choose runtime**: prefer `python tuner.py local-run --job-config Trainers/recipes/<recipe>.yaml --yes` for repeatable local Docker runs; use direct `cd Trainers/sft && python train_sft.py ...` for tight trainer iteration.
2. **Prepare dataset**: JSONL with `conversations` field, positive examples only
3. **Test setup**: set `run.dry_run: true` in local-run YAML or use `python train_sft.py --model-size 7b --tier quick --dry-run`
4. **Quick iteration**: cap `training.max_steps` in local-run YAML or use `--tier quick`
5. **Production run**: remove the step cap and use the intended `training`, `model`, `dataset`, and `lora` settings in YAML
6. **Monitor**: Watch live dashboard or `tail -f logs/training_latest.jsonl`
7. **Upload**: `python3 .skills/upload-deployment/scripts/upload_model.py ./final_model user/repo --save-method merged_16bit`

---

## Typical SFT Performance (7B on RTX 3090)

| Metric | Value |
|--------|-------|
| VRAM usage | ~7-9 GB |
| Speed (with packing) | ~500-800 tokens/sec |
| Time per epoch (~2700 examples) | ~15 min |
| Total (3 epochs) | ~45 min |

---

## Config File

**Direct trainer location:** `Trainers/sft/configs/config.yaml`

**Config-driven local Docker location:** `Trainers/recipes/*.yaml` (recipes with `target: local` or `target: both`)

Local Docker job configs keep runtime choices outside shell history:

```yaml
name: my-sft-run
provider: local_docker
job:
  image: unsloth/unsloth:latest
  pull_policy: missing
  transfer: auto
setup:
  pip: []
run:
  method: sft
  trainer: Trainers/sft/train_sft.py
model:
  name: Qwen/Qwen3.5-2B
  revision: null  # Optional; prefer an immutable Hugging Face commit SHA
  max_seq_length: 2048
  load_in_4bit: false
dataset:
  local_file: Datasets/my_data.jsonl
training:
  batch_size: 2
  gradient_accumulation: 8
  learning_rate: 1.0e-4
  num_epochs: 1
lora:
  r: 64
  alpha: 128
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
artifacts:
  output_root: toolset-training-artifacts/runs/local_docker/sft/{name}
  run_timestamp: "{timestamp}"
```

`model.revision` is optional in both direct SFT YAML and local-run recipes. Use
a full 40-character immutable Hub commit SHA to pin the joint model/tokenizer
snapshot. The trainer resolves and verifies that exact snapshot first, then
passes its local path to Unsloth so downstream family loaders cannot drop the
pin. Training lineage records requested and resolved source evidence. Missing
or mismatched snapshots fail closed; omission preserves the prior load call.

Key sections:
- `model` — model name, seq length, quantization
- `model.tokenizer` — optional generic special-token block described above
- `lora` — rank, alpha, dropout, target modules
- `training` — batch size, LR, epochs, packing, etc.
- `dataset` — source, filtering, split
- `evolutionary` — experimental gradient evolution (disabled by default)
- `aux_head` — optional auxiliary scalar readout head, head-only over a frozen base (Phase A) or co-trained with the LM loss over an unfrozen base (Phase B); disabled by default

For cloud runs, evolutionary SFT is now expressible through `run-experiment` specs or `cloud-pipeline --train-evolutionary-*` overrides. Keep the first run short and capped by `max_steps`; the wrapper adds real per-step overhead.

See `reference/training-config.md` for full config documentation.
