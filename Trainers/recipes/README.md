# Training Recipes

Unified directory for training, eval, and pipeline-job YAMLs. Each
recipe declares which runner consumes it (`target`) and what kind of
job it is (`method`). The CLI handlers (`local-run`, `cloud-run`) and
the interactive TUI both discover recipes from this directory via
`tuner.discovery.recipes`.

## Header Schema

Every recipe MUST include these top-level fields:

```yaml
name: "human-readable-recipe-name"        # used for run naming
description: "One-line summary"            # shown in TUI
target: local                              # local | cloud | both
method: sft                                # sft | kto | grpo | gguf | loss-bench | datagen | eval
```

The remaining fields depend on `target`. See `tuner/discovery/recipes.py`
for the loader and the existing recipes in this directory for examples.

### `target: local`

Recipes consumed by `tuner.handlers.local_run_handler`. Top-level keys:
`name`, `description`, `target`, `method`, `provider` (must be
`local_docker`), `job`, `setup`, `run`, `model`, `dataset`, `training`,
`lora`, `artifacts`.

### `target: cloud`

Recipes consumed by `tuner.handlers.cloud_run_handler`. Top-level keys:
`name`, `description`, `target`, `method`, `provider` (e.g. `hf_jobs`),
`job`, `repo`, `setup`, `run` (with `run.steps` shell commands),
`artifacts`.

### `target: both`

Recipes that work for both local and cloud. Shared fields live at the
top level; runner-specific overrides go in `local:` or `cloud:`
sub-blocks. The loader (`load_recipe(path, runner)`) deep-merges the
matching sub-block into the top level — sub-block wins on conflict —
before returning a flat dict to the handler.

```yaml
target: both
model:
  name: "unsloth/Qwen3-0.6B"
training:
  epochs: 2
local:
  training:
    batch_size: 4
cloud:
  provider: hf_jobs
  job:
    flavor: a10g-small
  run:
    steps: [...]
```

## Discovery

```python
from tuner.discovery.recipes import list_recipes, load_recipe

# All recipes
list_recipes(repo_root)

# Filter by target/method
list_recipes(repo_root, target="local")
list_recipes(repo_root, target="cloud", method="sft")

# Load and normalize for a specific runner
config = load_recipe(path, runner="local")
```

## Running a Recipe

```bash
# CLI
python tuner.py local-run --job-config Trainers/recipes/qwen35_2b_sft_smoke.yaml --yes
python tuner.py cloud-run --job-config Trainers/recipes/battle_of_models_qwen35_2b_sft.yaml --yes

# Interactive TUI: select 'train' or 'cloud' from the main menu and
# pick a recipe from the list.
```

## Method Vocabulary

| Method       | Runner used by                                       |
|--------------|------------------------------------------------------|
| `sft`        | SFT training                                         |
| `kto`        | KTO refinement                                       |
| `grpo`       | GRPO reinforcement                                   |
| `gguf`       | GGUF conversion job                                  |
| `loss-bench` | Loss-only benchmarking on a finished training run    |
| `datagen`    | Synthetic data generation (e.g. SynthChat vault)     |
| `eval`       | Evaluation runs (when migrated)                      |

The discovery module does not enforce this vocabulary; new methods can
be introduced by setting the field and filtering on the same value.
