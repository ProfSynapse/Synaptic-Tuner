# RTX 3090 GRPO Trainer

Config-driven GRPO training for tool-calling using **Unsloth + TRL** on RTX 3090 (24GB VRAM).

This trainer also supports **GSPO** via a YAML toggle (`training.use_gspo: true`), since GSPO is implemented by TRL's `GRPOTrainer` with sequence-level importance sampling.

## Quick Start (via Synaptic Tuner)

From repo root:

```bash
python3 tuner.py train
```

Select:
- Platform: `rtx`
- Method: `grpo`

## Direct Run

```bash
cd Trainers/grpo
python3 train_grpo.py
```

## Cloud Env-GRPO

For real multi-step environment-backed GRPO, use the separate cloud-first entrypoint:

```bash
cd Trainers/grpo
python3 train_env_grpo.py --config ./configs/env_config.yaml --dry-run
python3 train_env_grpo.py --config ./configs/env_config.yaml --print-cloud-bootstrap
```

This path is intentionally separate from the current static projected-dataset
GRPO trainer:
- it assumes canonical SynthChat rollout artifacts as the source dataset
- it expects a newer TRL/OpenEnv runtime
- it is designed to run in an isolated virtualenv on top of the shared
  Unsloth Docker image rather than mutating the old trainer environment
- it uses a published/merged SFT model repo as `model.model_name`, not a
  local LoRA checkpoint path
- it is wired for stock TRL `rollout_func` multi-step environment episodes,
  not the older single-step projected dataset reward path

## Configuration

Edit `Trainers/grpo/configs/config.yaml`:
- Model and LoRA settings
- Dataset (HF/local)
- GRPO/GSPO training hyperparameters
- Reward configuration (built-in + custom)

## Rewards

Rewards are defined in YAML under `rewards`:
- `rewards.items`: list of built-in reward components with weights/params
- `rewards.custom`: optional custom reward functions (module import or file path)

Custom reward functions should accept:
```python
def my_reward(completions, prompts=None, **kwargs) -> list[float]:
    ...
```

## Notes

- GRPO is **WSL/Linux only** (native Windows not supported). Use WSL2 if training on Windows.
