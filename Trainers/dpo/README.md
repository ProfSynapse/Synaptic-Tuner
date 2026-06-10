# DPO Fine-Tuning

Direct Preference Optimization (DPO) trainer, built by mirroring the KTO trainer
(`Trainers/kto/`) so it inherits the same env bootstrap, Unsloth model loader,
shared callbacks, cloud-artifact plumbing, and experiment tracking. DPO learns
from explicit chosen/rejected preference pairs.

## When to use DPO vs KTO

| Method | Data shape | Signal |
|--------|-----------|--------|
| **DPO** | Paired `chosen` / `rejected` per prompt | Relative preference between two completions |
| **KTO** | Unpaired completions with a boolean `label` | Per-completion desirable / undesirable |

Use DPO when you have a clear better/worse pair for each prompt. Use KTO when you
only have a thumbs-up / thumbs-down signal on individual completions.

## Dataset format

One JSON object per line, with `prompt` / `chosen` / `rejected` as message lists
(the TRL conversational DPO convention):

```jsonl
{"prompt":[{"role":"system","content":"..."},{"role":"user","content":"<question>"}],"chosen":[{"role":"assistant","content":"<desirable>"}],"rejected":[{"role":"assistant","content":"<undesirable>"}]}
```

There is no interleaving and no label column (contrast KTO): the chosen/rejected
pairing carries the preference signal directly.

## Quick start

```bash
# Validate config + data + preset resolution without downloading a model:
python train_dpo.py --qwen3-4b --local-file ../../path/to/dpo_train.jsonl --dry-run

# Real training run (local 4B pilot):
python train_dpo.py --qwen3-4b --local-file ../../path/to/dpo_train.jsonl
```

`--dry-run` exits before any model load (it validates config, loads and checks
the dataset, and resolves the model preset), so it is safe to run without GPUs or
the full ML stack.

## Model presets

Friendly flags resolve to `(size, hf_repo)` in `configs/model_presets.py`. The
Phase 1 pins are:

| Flag | Model |
|------|-------|
| `--qwen3-4b` | `unsloth/Qwen3-4B-Instruct-bnb-4bit` (pilot) |
| `--qwen3-8b` | `unsloth/Qwen3-8B-Instruct-bnb-4bit` (confirm) |

Any model can also be set directly with `--model-name`.

## Config and LoRA budget

`configs/config.yaml` holds the defaults; per-arm recipes live under
`Trainers/recipes/`. The LoRA budget surface (`r`, `lora_alpha`, `lora_dropout`,
`target_modules`, `use_rslora`, `use_dora`) is identical to the KTO and SFT
trainers, so an experiment can pin the same budget across SFT / DPO / KTO arms.

## DPO-specific config fields

- `beta` — DPO KL-regularization strength (default `0.1`).
- `loss_type` — DPO loss variant; `sigmoid` is vanilla DPO (default).

The KTO-only fields (`desirable_weight`, `undesirable_weight`, `use_kto_s`) do
not exist here.

## Reference model

Like KTO, DPO uses TRL's implicit (shared-weight) reference model by default to
save VRAM. Set `USE_EXPLICIT_REF_MODEL=true` to load an explicit frozen copy.

## TRL version

`train_dpo.py` targets modern TRL (>=0.22.x, used by the cloud recipes), where
`DPOTrainer` takes `processing_class=`. See `requirements.txt` for the local
RTX-3090 baseline pin and the version note.
