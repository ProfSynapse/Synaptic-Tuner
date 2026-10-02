# Common Tasks

Detailed command references and decision trees for all major workflows.

---

## 1. Training a Model

**Via CLI (Recommended):**
```bash
./run.sh
# Select: Train -> NVIDIA GPU -> SFT (for initial) or KTO (for refinement)
```

**Direct Python:**
```bash
# SFT (initial training)
cd Trainers/sft
python train_sft.py --model-size 7b

# KTO (refinement)
cd Trainers/kto
python train_kto.py --model-size 7b
```

**Key Difference:**
- **SFT**: Teaches tool-calling from scratch (positive examples only)
- **KTO**: Refines existing model (needs interleaved True/False examples)

### Decision Tree: Training

```
START: User wants to train a model
  |
  v
[1] Check environment ready?
    Run: ./run.sh status
    |
    +-- NOT READY --> Run: ./run.sh doctor --fix --> Retry status
    |
    +-- READY --> Continue
          |
          v
[2] What datasets are available?
    Run: ./run.sh list datasets
    |
    v
[3] Determine training type needed:
    |
    +-- NEW MODEL (learning from scratch)
    |     |
    |     v
    |   USE SFT:
    |   - Needs positive-only examples (label: true)
    |   - Higher learning rate: 2e-4
    |   - Epochs: 2-3 typical
    |   - Command: python Trainers/sft/train_sft.py
    |
    +-- REFINING EXISTING MODEL
          |
          v
        USE KTO:
        - Needs interleaved true/false examples
        - Lower learning rate: 1e-6 to 2e-7
        - Epochs: 1 typical
        - Command: python Trainers/kto/train_kto.py
          |
          v
[4] ALWAYS run with --dry-run first to validate configuration
    |
    v
[5] If dry-run passes, run actual training
```

---

## 1b. Local Docker Training (`local-run`)

`local-run` runs Unsloth SFT/KTO inside Docker on a local GPU. It sidesteps UID/GID permission issues (artifacts land with your user's ownership), keeps the asciimatics dashboard visible inside the container, and can optionally reuse a persistent container so repeat runs skip pip install and HuggingFace model download.

**Quick start:**
```bash
# Smoke test — tiny, verifies the whole path compiles + runs
python tuner.py local-run --job-config Trainers/recipes/qwen35_2b_sft_smoke.yaml

# Real 2-epoch SFT run
python tuner.py local-run --job-config Trainers/recipes/qwen35_2b_sft_2epoch.yaml
```

**Container management flags** (no `--job-config` required):
```bash
python tuner.py local-run --job-config <yaml> --container-status   # Print container state
python tuner.py local-run --job-config <yaml> --stop               # Stop persistent container
python tuner.py local-run --job-config <yaml> --rm-persistent      # Stop + remove persistent container
python tuner.py local-run --job-config <yaml> --yes                # Skip confirmation prompt
```

**Config reference:** The `job.*` YAML keys (`job.user`, `job.tty`, `job.persist`, `job.mount_hf_cache`, `job.mount_pip_cache`, `job.container_name`, `job.stop_timeout`, `job.transfer`, `job.keep_container`) are documented in [`.skills/fine-tuning/reference/training-config.md`](../.skills/fine-tuning/reference/training-config.md). The checked-in recipes with `target: local` in `Trainers/recipes/` are good starting templates.

**Troubleshooting:** See the `local-run` section in [`docs/troubleshooting.md`](troubleshooting.md) for UID/GID, bind-mount, and persistent-container issues.

---

## 1c. Grouped Validation Split

With `split_dataset: true` the SFT, KTO and DPO loaders carve a validation set
out of the training file. The default is a random row split
(`train_test_split(seed=42)`), which puts near-duplicate variants of one
scenario, seed fixture or transcript on both sides and makes validation loss
optimistic. Set `dataset.validation_group_key` to a dot-path into the raw row
and every row sharing that value lands on the same side; `test_size` is then
applied over groups (`ceil(test_size * n_groups)` groups go to validation,
chosen deterministically from the seed).

```yaml
# Trainers/recipes/<recipe>.yaml (local-run forwards it as --validation-group-key)
dataset:
  local_file: Datasets/kto/vault_shared_seed_dynamic_roles_kto_20260316.jsonl
  split_dataset: true
  validation_group_key: scenario_id
```

```bash
# Direct trainer use
cd Trainers/sft && python train_sft.py --split-dataset --validation-group-key metadata.scenario
```

- Missing, null or empty values fail loudly with the row index; fewer than two
  distinct groups is an error. Unset keeps the random split byte-identical.
- `local-run` rejects `validation_group_key` without `split_dataset: true`, and
  the SFT loader rejects it together with `use_preassigned_splits` (prepared
  datasets already carry their split from `prepare-dataset`).
- GRPO does not create a validation split, so it has no group key.

**Recommended keys** (from what the generators actually write):

| Rows produced by | Key | Groups |
|------------------|-----|--------|
| SynthChat `generate` (canonical rollouts) | `metadata.scenario` | One scenario template per group (strict template holdout) |
| SynthChat rollouts, keeping shared-seed variants together | `metadata.environment_seed.seed_id` | One environment seed fixture per group (coarser: several scenarios share a seed) |
| `project_rollout_datasets.py` KTO/SFT/GRPO projections | `scenario_id` (or `metadata.seed_id`) | Template; `metadata.seed_id` also keeps all turn rows of an episode together |
| Transcript distillation (`distill.py`) | `metadata.session_id` | One transcript per group; turn rows of a session share most of their context |
| Transcript distillation DPO projection | `provenance.prompt_key` | All chosen x rejected pairs of one prompt bucket |

The legacy checked-in SFT files (`Datasets/nonthinking_tools_sft_*.jsonl`,
`sft_train_67pct_12.25.jsonl`) carry no per-row group metadata, so a grouped
split is not available for them; regenerate with metadata or keep the random
split.

---

## 2. Uploading to HuggingFace

**Via CLI (Recommended):**
```bash
./run.sh
# Select: Upload -> Choose training run -> Configure save method
```

**Direct Python:**
```bash
cd Trainers/sft  # or kto
python3 .skills/upload-deployment/scripts/upload_model.py \
  ./sft_output/YYYYMMDD_HHMMSS/final_model \
  username/model-name \
  --save-method merged_16bit \
  --create-gguf
```

---

## 3. Generating Synthetic Data

```bash
# Interactive mode
./Tools/run_synth_chat.sh

# Quick test (100 examples)
./Tools/run_synth_chat.sh --quick
```

### Decision Tree: Generating Synthetic Data

```
START: User wants to generate synthetic training data
  |
  v
[1] Check SynthChat configuration:
    Review: synth_chat/config/config.yaml
    |
    v
[2] Is teacher model backend running?
    (Usually needs high-quality model like GPT-4, Claude, etc.)
    |
    +-- NOT AVAILABLE --> Configure OpenRouter or local model
    |
    +-- AVAILABLE --> Continue
          |
          v
[3] Quick test first:
    ./Tools/run_synth_chat.sh --quick
    |
    +-- ERRORS --> Check config and backend connectivity
    |
    +-- SUCCESS --> Continue
          |
          v
[4] Run full generation:
    ./Tools/run_synth_chat.sh
```

---

## 4. Improving Dataset Quality (LM Studio)

**Direct Command:**
```bash
python -m SynthChat.services.rubric_runner \
  --file Datasets/tools_datasets/thinking/agentManager/tools_v1.7.jsonl \
  --output Datasets/tools_datasets/thinking/agentManager/tools_v1.8.jsonl \
  --rubrics system_prompt_format \
  --backend lmstudio \
  --start-line 1 \
  --end-line 3 \
  --max-iterations 3
```

**Options:**
- `--file` - Input JSONL file
- `--output` - Output JSONL file
- `--rubrics` - Comma-separated rubric names (e.g., `system_prompt_format,thinking_quality`)
- `--backend` - `lmstudio`, `ollama`, or `openrouter`
- `--host` - LM Studio host (default: localhost)
- `--port` - LM Studio port (default: 1234)
- `--start-line` / `--end-line` - Line range to process
- `--max-iterations` - Max improvement loops per example
- `--no-interactions` - Disable interaction logging (enabled by default)

**List available rubrics:**
```bash
python -m SynthChat.services.rubric_runner --list
```

**Interactive Menu (Alternative):**
```bash
./run.sh
# Select: [6] Improvement Engine (clean datasets)
```

**How it works:**
1. Loads example from dataset
2. Runs **schema validation** (YAML-driven: xml, json, regex, yaml, code)
3. Passes validation results TO judge prompt
4. Judge sees errors and gives targeted feedback
5. Improver fixes based on feedback
6. Logs interaction to `SynthChat/interactions/` for KTO training

**Checking Interactions:**
```bash
# View latest interactions file
ls -lt SynthChat/interactions/ | head -5

# Inspect judge prompt (shows schema validation results)
cat SynthChat/interactions/interactions_LATEST.jsonl | head -1 | jq '.conversations[1].content'
```

**What Judge Sees:**
```
============================================================
SCHEMA VALIDATION RESULTS
============================================================

system_prompt_format: Schema validation FAILED
   - Missing required XML tag: <vault_structure>
   - Missing field in <selected_workspace>: workflows
```

### Decision Tree: Improving Dataset Quality

```
START: User wants to improve dataset quality
  |
  v
[1] Is LLM backend available?
    Check LM Studio, Ollama, or OpenRouter
    |
    +-- NOT AVAILABLE --> Start backend or configure API
    |
    +-- AVAILABLE --> Continue
          |
          v
[2] List available rubrics:
    Run: python -m SynthChat.services.rubric_runner --list
    |
    v
[3] Validate dataset first:
    Run: python3 .skills/synethetic-data-generation/scripts/validate_syngen.py <dataset_file>
    |
    +-- VALIDATION FAILED --> Fix JSON/format errors first
    |
    +-- VALIDATION PASSED --> Continue
          |
          v
[4] Test with small batch first:
    python -m SynthChat.services.rubric_runner \
      --file <input> --output <output> \
      --rubrics <rubric_names> \
      --start-line 1 --end-line 5
    |
    v
[5] If test passes, run on full dataset
```

---

## 5. Validating Datasets

```bash
python3 .skills/synethetic-data-generation/scripts/validate_syngen.py Datasets/your_dataset.jsonl
```

---

## 5b. Checking Train/Eval Contamination (`check-contamination`)

Before trusting eval numbers, check that eval prompts are not sitting in the
training data. For every eval item the command reports the highest fraction of
its word 8-grams found in any single training row (after NFKC, lowercase,
punctuation stripping and whitespace collapse), plus exact duplicate prompts.

```bash
# Default eval sources (all Evaluator scenarios) from configs/contamination/default.yaml
python tuner.py check-contamination --train-data Datasets/nonthinking_tools_sft_04.22.26.jsonl

# Several datasets, explicit sources, JSON output
python tuner.py check-contamination \
  --train-data Datasets/nonthinking_tools_sft_04.22.26.jsonl \
  --train-data Datasets/kto/vault_shared_seed_dynamic_roles_kto_20260316.jsonl \
  --eval-source Evaluator/config/scenarios/behavior_prompts.yaml \
  --eval-text Datasets/my_holdout.jsonl --threshold 0.5 --json

# Write the training set minus flagged rows (+ <out>.removed.json sidecar)
python tuner.py check-contamination --train-data Datasets/my_sft.jsonl \
  --write-decontaminated scratch/decontam/my_sft.decontaminated.jsonl
```

- Training rows: any trainer format (SFT `messages`/`conversations`,
  `prompt`/`completion`, KTO rows with `label`, prepared `raw_text`, DPO
  `prompt`/`chosen`/`rejected`, prompt-only GRPO), read with the SFT
  preprocessing format detection and tool-call rendering.
- Eval sources: Evaluator scenario YAMLs and prompt sets (`.json`/`.jsonl`)
  loaded with the Evaluator's own loaders, plus plain JSONL text
  (`--eval-text`, field via `--eval-text-field` or `eval.text_sources`).
  `eval.reference_fields` adds reference text (dot-paths into the case
  metadata) as separate items.
- Exit codes: `0` nothing flagged, `2` at least one item at/above `--threshold`
  or exactly duplicated in training, `1` error. A timestamped
  `contamination_report.json` is written under `scratch/contamination/` (the
  host artifact root in host-project mode) or `--report-dir`.
- Knobs (`ngram`, `threshold`, `min_item_tokens`, `top_k`, `train_roles`,
  sources) live in `configs/contamination/default.yaml`; pass
  `--contamination-config` for a project-specific file.
- `--write-decontaminated` never edits the source. For KTO data, rows are
  removed individually, so re-check label balance (the KTO trainer re-interleaves
  at load time).

---

## 6. Evaluating Models

```bash
# Via Evaluator
python -m Evaluator.cli \
  --model your-model-name \
  --prompt-set Evaluator/prompts/tool_prompts.json
```

### Decision Tree: Evaluating a Model

```
START: User wants to evaluate a model
  |
  v
[1] Is LLM backend running?
    |
    +-- LM Studio: Check http://localhost:1234/v1/models
    +-- Ollama: Check http://localhost:11434/api/tags
    +-- OpenRouter: Check OPENROUTER_API_KEY is set
    |
    +-- NOT RUNNING --> Start backend or set API key
    |
    +-- RUNNING --> Continue
          |
          v
[2] Find the model to evaluate:
    Run: ./run.sh list runs
    |
    v
[3] Choose scenario set:
    |
    +-- behavior_prompts: Test general behavior/reasoning
    +-- tool_prompts: Test tool-calling capability
    |
    v
[4] Run evaluation:
    python -m Evaluator.cli --model <model> --prompt-set <scenarios>
```
