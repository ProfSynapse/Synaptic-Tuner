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
cd Trainers/rtx3090_grpo
python3 train_grpo.py
```

## Configuration

Edit `Trainers/rtx3090_grpo/configs/config.yaml`:
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
