# Autonomous Experiment Loop Reference

Automated hyperparameter search using LLM reasoning + LightGBM surrogate optimization.

---

## Quick Start

```bash
# Run experiment loop
python tuner.py experiment-loop

# With custom config
python tuner.py experiment-loop --experiment-config configs/flywheel/experiment_loop.yaml
```

---

## How It Works

The experiment loop automates the train → evaluate → propose-next-config cycle:

1. **Propose** — LLM advisor suggests a hyperparameter configuration
2. **Train** — Run a short training experiment with proposed config
3. **Evaluate** — Score the resulting model via CheckpointEvaluator
4. **Record** — Store (config, score) pair in experiment history
5. **Transition** — After `surrogate_phase_threshold` experiments, switch from LLM to LightGBM surrogate
6. **Repeat** — Until `max_experiments` reached or convergence

### Dual Search Strategy

| Phase | When | How it picks configs |
|-------|------|---------------------|
| **LLM Advisor** | First N experiments (`surrogate_phase_threshold`) | LLM reasons about which hyperparameters to try, considering past results |
| **LightGBM Surrogate** | After N experiments | Trains a surrogate model on (config → score) data, samples candidates, picks predicted best |

The LLM phase explores broadly; the surrogate phase exploits learned patterns.

---

## Configuration

**File:** `configs/flywheel/experiment_loop.yaml`

```yaml
experiment_loop:
  max_experiments: 20             # Total experiments to run
  max_steps_per_experiment: 200   # Training steps per experiment
  trainer_type: "sft"             # "sft" or "kto"
  eval_backend: "local"           # "local" or "cloud"
  search_strategy: "llm_surrogate" # "llm_surrogate" or "random"
  surrogate_retrain_every: 5      # Retrain surrogate model every N experiments
  surrogate_phase_threshold: 10   # Switch from LLM to surrogate after N experiments

  search_space:
    learning_rate: [5e-5, 1e-4, 2e-4, 5e-4]
    r: [8, 16, 32, 64]
    lora_alpha: [16, 32, 64, 128]
    num_train_epochs: [1, 2]
    warmup_ratio: [0.02, 0.05, 0.1]
    weight_decay: [0.0, 0.01, 0.05]
    evolutionary.enabled: [true, false]
    evolutionary.noise_scale: [0.01, 0.03, 0.05]
```

### Search Space

Each key maps to a list of candidate values. The advisor/surrogate picks one value per key per experiment.

- **learning_rate** — Most impactful hyperparameter
- **r / lora_alpha** — LoRA rank and scaling
- **num_train_epochs** — Duration
- **warmup_ratio / weight_decay** — Regularization
- **evolutionary.\*** — Gradient noise evolution settings

---

## Key Components

### LLMAdvisor
Uses an LLM to propose hyperparameter configurations based on:
- Previous experiment results (config → score pairs)
- Domain knowledge about fine-tuning
- Exploration vs. exploitation balance

### SurrogateModel
LightGBM model trained on experiment history:
- Input: flattened hyperparameter config
- Output: predicted eval score
- Retrained every `surrogate_retrain_every` experiments
- Samples N random configs, picks the one with highest predicted score

### ExperimentResult
Each experiment produces:
- Config used
- Eval score (from CheckpointEvaluator)
- Training loss
- Duration
- Status (success / failed / timeout)

---

## Integration with Other Features

- **CheckpointEvaluator** — Used to score each experiment's best checkpoint
- **Tier presets** — `max_steps_per_experiment` serves a similar role to `--tier quick`
- **LoRA Surgery** — Can be run on the best experiment's adapter for further optimization

---

## Tips

- Start with `max_experiments: 10` to get a feel for the search space
- Use `max_steps_per_experiment: 200` for fast iteration (similar to `--tier quick`)
- LightGBM surrogate requires `pip install lightgbm` when it is not already installed
- The loop saves all results, so you can restart and it resumes from history
- Review experiment history to understand which hyperparameters matter most
- After finding the best config, run a full training with `--tier thorough` using those settings
