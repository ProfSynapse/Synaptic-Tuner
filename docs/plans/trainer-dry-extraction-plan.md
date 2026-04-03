# Trainer DRY Extraction Plan

**Status**: Draft (plan-mode consultation)
**Scope**: Extract duplicated utilities from `Trainers/sft/train_sft.py`, `Trainers/kto/train_kto.py`, and `Trainers/grpo/train_grpo.py` into `Trainers/shared/training_utils.py`
**Files analyzed**:
- `Trainers/sft/train_sft.py` — 1,351 lines
- `Trainers/kto/train_kto.py` — 1,329 lines
- `Trainers/grpo/train_grpo.py` — 468 lines
- `Trainers/shared/__init__.py` — Currently 11 lines (version only)
- `Trainers/shared/ui/training_progress.py` — Existing shared UI

---

## Executive Summary

Five categories of duplicated code exist across the three trainer scripts, totaling approximately 350-400 lines of redundancy. The functions range from perfectly identical (`save_training_lineage`, `setup_wandb`) to structurally similar with meaningful drift (`build_training_lineage`, `extract_previous_log_entries`). Additionally, ~60 lines of environment bootstrap boilerplate (Windows patches, dotenv, torch compile flags) are copy-pasted across all three files.

This plan proposes extracting shared code into two new modules under `Trainers/shared/` and unifying drifted implementations with a strategy that preserves each trainer's specific behavior via parameters rather than forked code.

---

## 1. Duplication Inventory

### 1.1 `setup_wandb()` — Identical (cosmetic drift only)

| Aspect | SFT (L279-296) | KTO (L112-133) | GRPO (L92-104) |
|--------|-----------------|-----------------|-----------------|
| Logic | Check env var, import wandb, login | Same | Same |
| Print format | `[OK] W&B:` / `[WARN] W&B:` | `checkmark W&B:` / `warning W&B:` | `checkmark W&B:` / `warning W&B:` |
| ImportError handling | Combined in `except Exception` | Separate `except ImportError` + `except Exception` | Combined in `except Exception` |
| **Verdict** | **Identical logic. Cosmetic print drift only.** |

**Unification strategy**: Single function with no parameters needed. Normalize print messages.

### 1.2 `extract_previous_log_entries()` — Significant drift

| Aspect | SFT (L299-323) | KTO (L176-244) |
|--------|-----------------|-----------------|
| Signature | `checkpoint_path: str` | `checkpoint_path: str` |
| Path resolution | Glob-based, finds first match | Parses checkpoint name for step number, navigates to run dir |
| Step filtering | None — returns all entries | Filters entries up to resume step |
| Return type | `list or None` | `list` (empty on failure) |
| Lines | 25 | 69 |

GRPO has no equivalent function.

**Unification strategy**: KTO's version is strictly more capable — it filters by step and has better error handling. The SFT version's glob approach could miss entries. **Adopt KTO's implementation as the canonical version.** SFT callers pass checkpoint_path and get correct filtering behavior for free.

### 1.3 `build_training_lineage()` — Structural overlap with parameter drift

| Aspect | SFT (L360-487) | KTO (L247-368) |
|--------|-----------------|-----------------|
| Shared structure | `training_type`, `timestamp`, `run_directory`, `model`, `lora`, `training`, `dataset`, `hardware`, `capacity_profile`, `results` | Same top-level keys |
| Unique to SFT | `evolutionary_stats`, `preprocessing_metadata` params, `packing`, `completion_only_loss`, `max_seq_length`, `filter_desirable` | — |
| Unique to KTO | `final_loss` param, `two_stage_lr`, `beta`, `desirable_weight`, `undesirable_weight`, `use_kto_s`, `max_length`, `max_prompt_length` | — |
| Training type | `"SFT"` | `"KTO"` |
| Lines | 128 | 122 |

**Unification strategy**: Extract a `_build_base_lineage()` helper that constructs the shared skeleton (model, lora, hardware, results, capacity_profile). Each trainer calls it and merges trainer-specific fields. This preserves extensibility — adding a new trainer-specific field only requires changes in that trainer's code, not the shared module.

### 1.4 `save_training_lineage()` — Nearly identical

| Aspect | SFT (L490-514) | KTO (L371-395) |
|--------|-----------------|-----------------|
| Logic | Write JSON, build capacity features, write capacity JSON | Same |
| Print format | `[OK]` prefix | `checkmark` emoji |
| **Verdict** | **Identical logic. Cosmetic print drift only.** |

**Unification strategy**: Single function, normalize print messages.

### 1.5 Environment bootstrap boilerplate — Identical

Duplicated across SFT, KTO, and partially GRPO:

| Boilerplate | SFT lines | KTO lines | GRPO lines |
|-------------|-----------|-----------|------------|
| UTF-8 stdout/stderr (Windows) | L21-28 | L29-36 | — |
| Dotenv loading | L31-35 | L38-43 | L44-49 |
| Torch compile disable env vars | L47-48 | L17-19 | L35-39 |
| Windows compatibility patches | L53-79 | L48-74 | — (exits on Windows) |
| Transformers log suppression | L39-41, L92 | L152-157 | L74-79 |

**Unification strategy**: Extract an `init_trainer_env()` function that handles all of this. Each trainer calls it at module level. GRPO can skip the Windows parts since it exits early on Windows.

### 1.6 Cloud artifact integration — Structural pattern

Both SFT and KTO have near-identical patterns for:
1. Building `run_paths` via `build_run_paths()` (local vs. cloud provider)
2. Writing initial manifest with `status="running"`
3. Post-training `sync_directory_to_hf_bucket()`
4. Final manifest update with `status="completed"` or `status="failed:..."`
5. `publish_final_model_to_hub()` conditional

This accounts for ~80 duplicated lines in each trainer.

**Unification strategy**: Extract a `CloudRunContext` dataclass or context manager that handles setup and teardown. However, this interleaves deeply with each trainer's `run()` function flow, making it harder to extract cleanly. **Defer to Phase 2** — extract after the simpler functions are consolidated.

---

## 2. Proposed Module Structure

```
Trainers/shared/
  __init__.py              (update: re-export from new modules)
  training_utils.py        (NEW: setup_wandb, extract_previous_log_entries,
                             save_training_lineage, build_base_lineage)
  env_bootstrap.py         (NEW: init_trainer_env, apply_windows_patches,
                             setup_utf8_output, load_env_file,
                             disable_torch_compile, suppress_transformers_logging)
  ui/
    training_progress.py   (existing, unchanged)
```

### 2.1 `training_utils.py` — API Design

```python
"""Shared training utilities for SFT, KTO, and GRPO trainers."""

from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import os


def setup_wandb() -> bool:
    """Auto-setup W&B if WANDB_API_KEY is in environment.

    Returns True if W&B login succeeded, False otherwise.
    """
    ...


def extract_previous_log_entries(checkpoint_path: str) -> List[dict]:
    """Extract log entries from a previous run when resuming from checkpoint.

    Parses the checkpoint path to determine the resume step, finds the
    most recent training log file, and returns entries up to that step.

    Args:
        checkpoint_path: Path to checkpoint directory
            (e.g., "output/20251114_135227/checkpoints/checkpoint-50")

    Returns:
        List of log entry dicts (empty list on failure)
    """
    ...


def build_base_lineage(
    training_type: str,
    config,
    train_dataset,
    eval_dataset,
    trainer,
    run_dir: Path,
    args,
    training_time_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """Build the shared portion of training lineage.

    Constructs model, lora, hardware, capacity_profile, and results sections.
    Callers merge trainer-specific fields (e.g., KTO beta, SFT evolutionary).

    Args:
        training_type: "SFT", "KTO", or "GRPO"
        config: Training configuration object (must have .model, .lora, .training)
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset (may be None)
        trainer: Trainer object after training
        run_dir: Path to training run directory
        args: Command-line arguments namespace
        training_time_seconds: Total training time in seconds

    Returns:
        Lineage dict with shared fields populated. Callers extend this.
    """
    ...


def save_training_lineage(lineage: Dict[str, Any], run_dir: Path) -> Path:
    """Save training lineage to JSON file, plus capacity features.

    Args:
        lineage: Training lineage dictionary
        run_dir: Path to training run directory

    Returns:
        Path to saved lineage file
    """
    ...
```

### 2.2 `env_bootstrap.py` — API Design

```python
"""Environment bootstrap for Unsloth-based trainers.

Must be called early — before importing unsloth, torch, or transformers.
"""

import os
import sys


def init_trainer_env(
    *,
    disable_torch_compile: bool = True,
    apply_windows_patches: bool = True,
    load_dotenv: bool = True,
    suppress_transformers: bool = True,
    utf8_output: bool = True,
) -> None:
    """One-call environment bootstrap for all trainers.

    Call this at the top of your trainer script, BEFORE importing
    unsloth, torch, or transformers.

    Args:
        disable_torch_compile: Set TORCH_COMPILE_DISABLE, TORCHDYNAMO_DISABLE, PYTORCH_JIT
        apply_windows_patches: Apply Unsloth Windows compatibility patches
        load_dotenv: Load .env file for API keys
        suppress_transformers: Suppress verbose transformers logging
        utf8_output: Force UTF-8 stdout/stderr on Windows
    """
    ...


def suppress_transformers_logging() -> None:
    """Suppress verbose transformers/trainer logging.

    Safe to call after transformers is imported.
    """
    ...
```

---

## 3. Drift Resolution Details

### 3.1 `extract_previous_log_entries` — Adopting KTO's version

The SFT version (25 lines) takes a simpler approach — glob for any `training_*.jsonl` and return all entries. The KTO version (69 lines) parses the checkpoint step number from the path and filters entries to only those at or below that step. This is the correct behavior: when resuming from step 50, you want entries 0-50, not the full log from a prior run that may have gone to step 200.

**Resolution**: Use KTO's implementation. SFT callers get improved behavior (filtered entries) with no code changes at the call site — both pass `checkpoint_path` and receive a list.

**Behavioral change for SFT**: Previously returned `None` when no entries found; now returns `[]`. The SFT call site (`previous_log_entries = extract_previous_log_entries(...)`) already handles both falsy values, so this is safe.

### 3.2 `build_training_lineage` — Shared skeleton + trainer extensions

The ~70% shared structure becomes `build_base_lineage()`. Each trainer's remaining ~30% is merged in the caller:

**SFT caller** (in `train_sft.py`):
```python
lineage = build_base_lineage("SFT", config, train_dataset, ...)
lineage["training"].update({
    "max_seq_length": config.training.max_seq_length,
    "packing": config.training.packing,
    "completion_only_loss": config.training.completion_only_loss,
})
if preprocessing_metadata:
    lineage["dataset"]["preprocessing"] = dict(preprocessing_metadata)
if hasattr(config, 'evolutionary') and config.evolutionary.enabled:
    lineage["evolutionary"] = { ... }
```

**KTO caller** (in `train_kto.py`):
```python
lineage = build_base_lineage("KTO", config, train_dataset, ...)
lineage["training"].update({
    "max_length": config.training.max_length,
    "max_prompt_length": config.training.max_prompt_length,
    "beta": config.training.beta,
    "desirable_weight": config.training.desirable_weight,
    "undesirable_weight": config.training.undesirable_weight,
    "use_kto_s": config.training.use_kto_s,
})
if config.training.use_two_stage_lr:
    lineage["training"]["two_stage_lr"] = { ... }
if final_loss is not None:
    lineage["results"]["final_loss"] = final_loss
```

This design follows the Open/Closed Principle: the shared function is closed for modification, but open for extension via the caller's merge.

### 3.3 Config attribute access compatibility

SFT and KTO use dataclass configs (`config.model.model_name`), while GRPO uses plain dicts (`config['model']['model_name']`). The shared functions must handle this.

**Resolution**: `build_base_lineage` and `save_training_lineage` only receive already-extracted values (primitives), so they are config-agnostic. `setup_wandb` and `extract_previous_log_entries` don't touch config at all. No compatibility issue.

For `build_base_lineage` specifically, the caller constructs the lineage dict in their own code and passes in the values — the shared function doesn't need to know about config format.

**Revised approach**: Actually, looking more closely at the shared fields, the most practical design is to have `build_base_lineage` accept the values it needs as explicit keyword arguments rather than a config object:

```python
def build_base_lineage(
    training_type: str,
    *,
    base_model: str,
    max_seq_length: int,
    load_in_4bit: bool,
    dtype: str,
    lora_r: int,
    lora_alpha: float,
    lora_dropout: float,
    target_modules: list,
    bias: str,
    batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    num_epochs: int,
    max_steps: int,
    warmup_ratio: float,
    lr_scheduler: str,
    optimizer: str,
    max_grad_norm: float,
    gradient_checkpointing: bool,
    fp16: bool,
    bf16: bool,
    seed: int,
    dataset_source: str,
    train_size: int,
    eval_size: int,
    run_dir: Path,
    trainer,
    training_time_seconds: Optional[float] = None,
) -> Dict[str, Any]:
```

This is verbose but config-format-agnostic and self-documenting. Each caller maps their config format to these kwargs at the call site.

**Alternative** (simpler but less type-safe): Accept pre-built sub-dicts:

```python
def build_base_lineage(
    training_type: str,
    model_info: dict,
    lora_info: dict,
    training_info: dict,
    dataset_info: dict,
    run_dir: Path,
    trainer,
    training_time_seconds: Optional[float] = None,
) -> Dict[str, Any]:
```

**Recommendation**: Use the sub-dict approach. It keeps the signature manageable, each trainer builds the sub-dicts from their own config format, and the shared function just assembles them with hardware/capacity/results. This matches the existing lineage structure directly.

---

## 4. Implementation Roadmap

### Phase 1: Environment bootstrap extraction (lowest risk, highest reuse)

**Target**: `Trainers/shared/env_bootstrap.py`

| Step | Action | Risk |
|------|--------|------|
| 1a | Create `env_bootstrap.py` with `init_trainer_env()` | Low — pure env var setting |
| 1b | Replace boilerplate in `train_sft.py` with `from shared.env_bootstrap import init_trainer_env; init_trainer_env()` | Low — exact same behavior |
| 1c | Same for `train_kto.py` | Low |
| 1d | Same for `train_grpo.py` (skip Windows patches) | Low |
| 1e | Verify all three trainers still import and run (dry-run mode) | — |

**Estimated reduction**: ~60 lines removed from each of SFT and KTO, ~20 from GRPO.

**Ordering constraint**: `init_trainer_env()` must be called BEFORE any torch/unsloth/transformers imports. This is a module-level call, not inside a function.

### Phase 2: Simple function extraction (low risk, high confidence)

**Target**: `Trainers/shared/training_utils.py`

| Step | Action | Risk |
|------|--------|------|
| 2a | Extract `setup_wandb()` — use GRPO's concise version as base, normalize prints | Low — identical logic |
| 2b | Extract `save_training_lineage()` — normalize print messages | Low — identical logic |
| 2c | Extract `extract_previous_log_entries()` — adopt KTO's version | Medium — SFT gets new filtering behavior |
| 2d | Update imports in all three trainers | Low |
| 2e | Verify dry-run mode for SFT, KTO, GRPO | — |

**Estimated reduction**: ~80 lines removed from SFT, ~110 from KTO, ~15 from GRPO.

### Phase 3: Lineage builder refactoring (medium risk, architectural change)

**Target**: `build_base_lineage()` in `Trainers/shared/training_utils.py`

| Step | Action | Risk |
|------|--------|------|
| 3a | Implement `build_base_lineage()` with sub-dict API | Low — new code, no existing behavior changed yet |
| 3b | Refactor SFT `build_training_lineage()` to call `build_base_lineage()` + merge SFT-specific fields | Medium — must verify lineage JSON output is identical |
| 3c | Refactor KTO `build_training_lineage()` similarly | Medium — same verification needed |
| 3d | Verify lineage JSON output matches original for both trainers | — |

**Estimated reduction**: ~80 lines removed from SFT, ~80 from KTO.

### Phase 4: Tier preset loading extraction (low-medium risk)

**Target**: `apply_tier_preset()` in `Trainers/shared/training_utils.py`

The tier preset loading code is nearly identical between SFT (L713-743) and KTO (L730-760):
1. Load YAML from `configs/tiers/{tier}.yaml`
2. Iterate keys against a `_tier_config_map` dict
3. Map each key to `(section, attr)` on the config dataclass
4. Special-case `max_steps` (goes to `args`, not config)

The only difference is the key mappings themselves (SFT includes `use_rslora`, `target_modules`; KTO includes `use_dora` and different training keys).

| Step | Action | Risk |
|------|--------|------|
| 4a | Extract `apply_tier_preset(config, tier_name, tier_config_map, args, configs_dir)` | Low — mechanical extraction |
| 4b | Each trainer defines its own `_tier_config_map` dict and calls `apply_tier_preset()` | Low — the map is the only variation point |
| 4c | Verify tier preset application produces identical config values | — |

**Estimated reduction**: ~25 lines removed from each of SFT and KTO.

### Phase 5: GRPO lineage support (medium risk)

**Target**: Add GRPO support to `build_base_lineage()` flow

GRPO currently bypasses the lineage flow entirely — it uses `register_grpo_run()` from `shared/experiment_tracking/adapters.py` to create a `RunRecord` directly from ad-hoc dict construction, without producing a `training_lineage.json` file.

| Step | Action | Risk |
|------|--------|------|
| 5a | Add `build_base_lineage("GRPO", ...)` call to `train_grpo.py` post-training, producing `training_lineage.json` | Medium — GRPO uses plain dicts not dataclasses, so the sub-dict API is essential here |
| 5b | Add GRPO-specific lineage fields (reward functions, GRPO-specific hyperparams) via `.update()` | Low |
| 5c | Update `register_grpo_run()` to read from `training_lineage.json` like the SFT/KTO adapters, rather than constructing ad-hoc | Medium — must preserve RunRecord field mapping |
| 5d | Verify GRPO runs produce identical RunRecords before and after | — |

**Note**: This unifies all three trainers onto the same lineage flow: `build_base_lineage()` → `save_training_lineage()` → adapter reads JSON → `RunRecord`. Currently only SFT and KTO follow this path.

### Phase 6 (Future): Cloud artifact context extraction

This is deferred and not part of the immediate plan. The cloud artifact setup/teardown pattern (~80 lines duplicated) would benefit from a `CloudRunContext` abstraction, but it interleaves deeply with each trainer's `run()` function and would require more careful design.

---

## 5. Testing Strategy

### No existing tests

None of the five duplicated functions have test coverage today. The refactoring should add tests as part of extraction.

### Proposed test file: `tests/trainers/shared/test_training_utils.py`

| Test | What it verifies |
|------|------------------|
| `test_setup_wandb_no_key` | Returns False when WANDB_API_KEY not set |
| `test_setup_wandb_with_key` | Returns True when key set (mock wandb.login) |
| `test_setup_wandb_import_error` | Returns False when wandb not installed |
| `test_extract_log_entries_valid_checkpoint` | Parses step number, filters entries correctly |
| `test_extract_log_entries_no_logs` | Returns empty list when no log files exist |
| `test_extract_log_entries_invalid_path` | Returns empty list for malformed paths |
| `test_save_training_lineage_writes_json` | Creates training_lineage.json with correct content |
| `test_save_training_lineage_writes_capacity` | Creates capacity_features.json when features available |
| `test_build_base_lineage_structure` | Output dict has all expected top-level keys |
| `test_build_base_lineage_results_from_trainer` | Extracts final_step, total_epochs from trainer state |
| `test_build_base_lineage_grpo` | GRPO-specific fields merge correctly with base lineage |
| `test_apply_tier_preset_maps_keys` | Tier YAML keys map to correct config attributes |
| `test_apply_tier_preset_max_steps_to_args` | `max_steps` routes to args, not config |
| `test_apply_tier_preset_unknown_keys_ignored` | Unknown keys in tier YAML don't crash |

### Proposed test file: `tests/trainers/shared/test_env_bootstrap.py`

| Test | What it verifies |
|------|------------------|
| `test_init_trainer_env_sets_env_vars` | TORCH_COMPILE_DISABLE, PYTORCH_JIT, etc. are set |
| `test_init_trainer_env_utf8_windows` | UTF-8 output wrappers applied on Windows (mock sys.platform) |
| `test_init_trainer_env_dotenv_optional` | Gracefully handles missing python-dotenv |

---

## 6. Migration Checklist

For each trainer, after refactoring:

- [ ] `python train_sft.py --dry-run` succeeds
- [ ] `python train_kto.py --dry-run` succeeds
- [ ] `python train_grpo.py --dry-run` succeeds
- [ ] Lineage JSON output from SFT matches pre-refactoring output (diff test)
- [ ] Lineage JSON output from KTO matches pre-refactoring output (diff test)
- [ ] `setup_wandb()` still returns False when no key (env test)
- [ ] Checkpoint resume with `extract_previous_log_entries()` filters correctly
- [ ] Cloud provider path (`--cloud-provider hf_jobs`) still works for SFT and KTO
- [ ] Tier preset application produces identical config values for SFT and KTO
- [ ] GRPO produces `training_lineage.json` matching expected schema
- [ ] GRPO `register_grpo_run()` produces identical RunRecords before and after
- [ ] All new tests pass
- [ ] No circular imports between `Trainers/shared/` and `Trainers/sft/`, `Trainers/kto/`, `Trainers/grpo/`

---

## 7. Import Path Considerations

The trainers run in two environments:
1. **Local**: `python Trainers/sft/train_sft.py` — sys.path includes `Trainers/sft/src` and repo root
2. **Cloud (HF Jobs)**: The training script runs in a container where the repo is cloned fresh

Both environments already have `sys.path.insert(0, str(Path(__file__).parent.parent.parent))` (repo root), which means `from shared.training_utils import ...` would resolve to `{repo_root}/shared/training_utils.py` — **not** `Trainers/shared/training_utils.py`.

**Critical path issue**: The existing `shared/` at repo root contains `llm/`, `judge/`, `upload/`, `validation/`, etc. The `Trainers/shared/` is a separate package. Currently, trainers import from both:
- `from shared.cloud_artifacts import ...` (repo root `shared/`)
- `from src.data_loader import ...` (local `src/`)

The new `Trainers/shared/training_utils.py` needs to be importable. Two options:

**Option A (Recommended)**: Relative path adjustment — trainers already add their own `src` to path. Add `Trainers/` to path as well:
```python
sys.path.insert(0, str(Path(__file__).parent.parent))  # Trainers/
from shared.training_utils import setup_wandb, ...
```
But this **conflicts** with existing `from shared.cloud_artifacts import ...` which resolves to `{repo_root}/shared/`.

**Option B (Safer)**: Use a different module name to avoid collision:
```python
# In Trainers/shared/__init__.py or a renamed package
from trainers_shared.training_utils import ...
```
This requires renaming `Trainers/shared/` to `trainers_shared/` which breaks the existing `Trainers/shared/ui/` import path.

**Option C (Simplest, Recommended)**: Keep `Trainers/shared/` as-is. Import with explicit path manipulation already done in the trainers:
```python
# Each trainer already does:
sys.path.insert(0, str(Path(__file__).parent / "src"))         # src/
sys.path.insert(0, str(Path(__file__).parent.parent.parent))   # repo root

# Add Trainers/ to path:
sys.path.insert(0, str(Path(__file__).parent.parent))          # Trainers/
```

Wait — this still causes `shared` to be ambiguous (both `{repo_root}/shared/` and `Trainers/shared/` on the path).

**Option D (Correct fix)**: Use a package-qualified import. The trainers should import from `Trainers/shared/` using a path that distinguishes it from the repo-root `shared/`:

```python
# In each trainer, before imports:
_trainers_dir = Path(__file__).parent.parent
sys.path.insert(0, str(_trainers_dir))

# Then import as:
from shared.training_utils import setup_wandb  # resolves to Trainers/shared/
```

**But** this only works if `Trainers/` is **earlier** in sys.path than the repo root. Since `sys.path.insert(0, ...)` prepends, the last `insert(0, ...)` wins for shadowed names. We need `Trainers/` to be inserted **after** the repo root insert:

```python
sys.path.insert(0, str(Path(__file__).parent.parent.parent))   # repo root
sys.path.insert(0, str(Path(__file__).parent.parent))           # Trainers/ (wins for 'shared')
sys.path.insert(0, str(Path(__file__).parent / "src"))          # src/
```

This makes `from shared.xxx` resolve to `Trainers/shared/xxx` first, shadowing `{repo_root}/shared/xxx`. But the trainers **also need** `from shared.cloud_artifacts import ...` (repo root).

**Final Resolution**: The existing imports `from shared.cloud_artifacts import ...` resolve to the repo-root `shared/` because of the repo root path insertion. If we also put `Trainers/` on the path, `shared` becomes ambiguous.

**The clean fix**: Use a different package name for trainer-specific shared code. Rename `Trainers/shared/` to... no, that's disruptive.

**Practical approach**: Put the new utilities in the **repo-root** `shared/` package instead, which is already importable from all trainers:

```
shared/
  training_utils.py     (NEW — used by all trainers)
  env_bootstrap.py      (NEW — used by all trainers)
  cloud_artifacts.py    (existing)
  training_capacity.py  (existing)
  ...
```

This is the path of least resistance — all trainers already import from `shared.*`, the path is already configured, and there's no ambiguity. The `Trainers/shared/` directory can remain for UI-specific code.

**Recommendation**: Place new shared modules in `shared/` (repo root), not `Trainers/shared/`. This avoids all import path issues.

---

## 8. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Import path collision between `shared/` locations | Medium | High — breaks imports | Place new modules in repo-root `shared/` (Section 7 resolution) |
| `extract_previous_log_entries` behavior change for SFT | Low | Medium — different log entries returned on resume | SFT gets _improved_ behavior (filtered by step); verify with checkpoint resume test |
| `build_base_lineage` missing trainer-specific fields | Low | Medium — incomplete lineage | Diff test: compare output JSON before/after refactoring |
| Cloud environment import failures | Low | High — training job crashes | Test with `--dry-run` in simulated cloud env; HF Jobs clones fresh repo |
| `init_trainer_env()` call order sensitivity | Medium | High — torch compile not disabled | Document and enforce: must be called before any torch import |
| Tier config map drift between SFT/KTO | Low | Low — different keys are intentional | Each trainer owns its own map; shared function handles iteration |
| GRPO dict-based config vs dataclass API | Medium | Medium — sub-dict API mismatch | Sub-dict API is config-format-agnostic by design; GRPO builds dicts directly |
| GRPO `register_grpo_run` RunRecord field mapping | Low | Medium — experiment tracking regression | Diff test: compare RunRecord output before/after |

---

## 9. Expected Outcome

| Metric | Before | After |
|--------|--------|-------|
| `train_sft.py` lines | 1,351 | ~1,055 (-295) |
| `train_kto.py` lines | 1,329 | ~1,035 (-295) |
| `train_grpo.py` lines | 468 | ~420 (-48) |
| Shared utility lines | 0 | ~300 (new) |
| Net line change | 3,148 total | ~2,810 total (-338 net, but shared code is tested) |
| Test coverage of shared functions | 0% | ~90% |
| Number of places to update `setup_wandb` | 3 | 1 |
| Number of places to update lineage format | 2 | 1 (base) + 3 (extensions, now including GRPO) |
| Number of places to update tier loading | 2 | 1 (shared) + 2 (config maps) |

---

## 10. Implementation Order Summary

```
Phase 1: env_bootstrap.py (lowest risk)
  └── init_trainer_env() — consolidate env vars, Windows patches, dotenv, logging

Phase 2: training_utils.py — simple extractions (low risk)
  ├── setup_wandb()
  ├── save_training_lineage()
  └── extract_previous_log_entries()

Phase 3: training_utils.py — lineage refactoring (medium risk)
  └── build_base_lineage() with sub-dict API

Phase 4: training_utils.py — tier preset extraction (low-medium risk)
  └── apply_tier_preset() — shared YAML loading + config mapping

Phase 5: GRPO lineage unification (medium risk)
  ├── Add build_base_lineage("GRPO", ...) to train_grpo.py
  └── Update register_grpo_run() to read from training_lineage.json

Phase 6 (Future): Cloud artifact context (deferred)
  └── CloudRunContext for run_paths/manifest/sync pattern
```

Each phase is independently deployable and testable. If any phase introduces issues, it can be reverted without affecting the others.
