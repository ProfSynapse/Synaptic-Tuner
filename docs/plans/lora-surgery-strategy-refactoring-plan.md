# LoRA Surgery Strategy Pattern Refactoring Plan

## Executive Summary

`shared/evolutionary/lora_surgery.py` (1,054 lines) bundles 8 distinct surgical operations, 12 utility functions, 3 dataclasses, and the orchestrator class into a single file. This plan decomposes it using the Strategy pattern, mirroring the existing `shared/evolutionary/strategies/` pattern already established for gradient modification strategies. The `LoRASurgeon` becomes a thin orchestrator delegating to registered `SurgeryOperation` strategy objects.

---

## 1. Current Architecture Analysis

### File Structure (1,054 lines)

| Section | Lines | Purpose |
|---------|-------|---------|
| Dataclasses (`SurgeryConfig`, `OperationResult`, `SurgeryResult`) | 58-170 | Configuration and result types |
| Helper functions (12 functions) | 172-288 | Weight I/O, adapter management, key parsing |
| `LoRASurgeon` class | 294-1029 | Orchestrator + 8 operation methods + internal helpers |
| `_find_lora_pairs` (standalone) | 1031-1054 | Pair matching (outside class) |

### The 8 Operations

All operations share an identical signature and lifecycle:

```python
async def operation_name(self, adapter_path: str, baseline_score: float) -> OperationResult
```

**Common lifecycle**:
1. Load weights from adapter directory (or check preconditions)
2. For each variant parameter (from `SurgeryConfig`):
   a. Copy adapter to work directory
   b. Modify weights according to the operation's logic
   c. Evaluate modified adapter via `self._evaluate()`
   d. Track best score
3. Return `OperationResult` with best variant info

**Per-operation specifics**:

| Operation | Lines | Iterates Over | Weight Modification | Special Dependencies |
|-----------|-------|---------------|--------------------|--------------------|
| `alpha_sweep` | ~50 | `config.alpha_multipliers` | Config-only (no weight changes) | None |
| `layer_scaling` | ~68 | `layer_indices x config.layer_scales` | Multiply layer weights by scale | `_get_layer_indices` |
| `module_ablation` | ~57 | `module_types` | Zero module-type weights | `_get_module_types` |
| `checkpoint_interpolation` | ~68 | `config.blend_ratios` | Linear interpolation of two checkpoints | `other_checkpoint_path` config |
| `dare_drop_rescale` | ~60 | `config.dare_drop_rates` | Random mask + rescale | None |
| `metrics_weighted_merge` | ~57 | Single variant | Softmax-weighted checkpoint merge | `checkpoint_paths/scores` config |
| `svd_rank_reduction` | ~104 | `config.svd_rank_fractions` | Truncated SVD on LoRA pairs | `_find_lora_pairs` |
| `attention_mlp_ablation` | ~86 | 2 variants (attn/mlp) | Zero attention or MLP weights | `_is_attention_key`, `_is_mlp_key` |

### Existing Pattern Precedent

`shared/evolutionary/strategies/` already implements a Strategy pattern:
- `BaseStrategy` (ABC with `generate_candidates` abstract method)
- Concrete strategies in separate files (`gradient_noise.py`, `antithetic_noise.py`, etc.)
- `get_strategy()` factory function in `__init__.py`
- `CandidateGenerator` orchestrates strategy execution

The surgery refactoring should follow this established convention.

### External Consumers

| Consumer | Imports | Impact |
|----------|---------|--------|
| `tuner/handlers/surgery_handler.py` | `LoRASurgeon`, `SurgeryConfig` | Must remain importable from same path |
| `tests/test_karpathy_integration.py` | `LoRASurgeon`, `SurgeryConfig`, `SurgeryResult` | Must remain importable from same path |
| `tests/test_lora_surgery.py` | Everything (class + 12 private helpers) | Needs updated imports |
| `shared/evolutionary/__init__.py` | Lazy imports of `LoRASurgeon`, `SurgeryConfig`, `SurgeryResult`, `OperationResult` | Must update lazy import path |

---

## 2. Target Architecture

### Directory Layout

```
shared/evolutionary/
  surgery/                          # NEW package
    __init__.py                     # Public API + get_operation() factory
    base.py                         # SurgeryOperation protocol/ABC
    surgeon.py                      # LoRASurgeon orchestrator (thin)
    config.py                       # SurgeryConfig, OperationResult, SurgeryResult
    utils.py                        # All 12+ helper functions
    operations/                     # NEW package
      __init__.py                   # Re-exports all operations
      alpha_sweep.py                # AlphaSweepOperation
      layer_scaling.py              # LayerScalingOperation
      module_ablation.py            # ModuleAblationOperation
      checkpoint_interpolation.py   # CheckpointInterpolationOperation
      dare_drop_rescale.py          # DAREDropRescaleOperation
      metrics_weighted_merge.py     # MetricsWeightedMergeOperation
      svd_rank_reduction.py         # SVDRankReductionOperation
      attention_mlp_ablation.py     # AttentionMLPAblationOperation
  lora_surgery.py                   # KEPT as backward-compat re-export shim
```

### Component Design

#### 2.1 `SurgeryOperation` Base (Protocol)

```python
# shared/evolutionary/surgery/base.py

from __future__ import annotations
from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from .config import OperationResult, SurgeryConfig

class SurgeryOperation(Protocol):
    """Protocol for surgery operations (Strategy pattern)."""

    name: str

    async def execute(
        self,
        adapter_path: str,
        baseline_score: float,
        work_dir: str,
        config: SurgeryConfig,
        evaluate_fn,  # async callable: str -> float
    ) -> OperationResult:
        """Run the operation and return the result."""
        ...
```

**Design decisions**:
- Use `Protocol` (not ABC) for structural typing consistency with `EvalBackend`
- Pass `evaluate_fn` as a callable rather than the full `LoRASurgeon` — operations should not know about the orchestrator
- Pass `work_dir` and `config` explicitly — operations are stateless; the orchestrator owns state
- Each operation is responsible only for its weight modification logic and variant iteration

#### 2.2 Concrete Operations (one per file)

Each operation file contains a single class implementing `SurgeryOperation`. Example structure:

```python
# shared/evolutionary/surgery/operations/alpha_sweep.py

from __future__ import annotations
from ..config import OperationResult, SurgeryConfig
from ..utils import _copy_adapter, _load_adapter_config, _save_adapter_config

class AlphaSweepOperation:
    """Modify adapter_config.json lora_alpha, no weight changes."""

    name = "alpha_sweep"

    async def execute(
        self,
        adapter_path: str,
        baseline_score: float,
        work_dir: str,
        config: SurgeryConfig,
        evaluate_fn,
    ) -> OperationResult:
        # ... operation logic extracted from LoRASurgeon.alpha_sweep ...
```

#### 2.3 Operation Registry

```python
# shared/evolutionary/surgery/__init__.py

from .config import SurgeryConfig, OperationResult, SurgeryResult
from .surgeon import LoRASurgeon
from .base import SurgeryOperation

# Static registry — all operations known at import time
_OPERATION_REGISTRY: dict[str, type[SurgeryOperation]] = {}

def register_operation(cls):
    """Decorator to register an operation class."""
    _OPERATION_REGISTRY[cls.name] = cls
    return cls

def get_operation(name: str) -> SurgeryOperation:
    """Get an operation instance by name."""
    if name not in _OPERATION_REGISTRY:
        raise ValueError(f"Unknown surgery operation: {name}. Available: {list(_OPERATION_REGISTRY.keys())}")
    return _OPERATION_REGISTRY[name]()

def list_operations() -> list[str]:
    """Return names of all registered operations."""
    return sorted(_OPERATION_REGISTRY.keys())
```

**Static vs dynamic registration**: Use a decorator-based static registry. All 8 operations are decorated with `@register_operation` at module level. The `operations/__init__.py` imports all operation modules to trigger registration. This is simpler than plugin-style discovery and matches the scale of this codebase (8 known operations, not an open-ended plugin system).

#### 2.4 `LoRASurgeon` (Thin Orchestrator)

The refactored `LoRASurgeon` drops from ~735 lines to ~120 lines. It retains:
- `__init__`, `__aenter__`, `__aexit__`, `cleanup`
- `run_surgery()` — now iterates over operation names, calls `get_operation(name).execute()`
- `_evaluate()` — wraps eval backend call (passed to operations as `evaluate_fn`)
- `_save_report()` — writes JSON report

```python
# shared/evolutionary/surgery/surgeon.py

class LoRASurgeon:
    """Thin orchestrator that delegates to SurgeryOperation strategies."""

    async def run_surgery(self) -> SurgeryResult:
        # ... setup ...
        for operation_name in self.config.operations:
            try:
                operation = get_operation(operation_name)
            except ValueError:
                logger.warning("Unknown operation: %s, skipping", operation_name)
                continue

            result = await operation.execute(
                adapter_path=best_adapter,
                baseline_score=best_score,
                work_dir=self._work_dir,
                config=self.config,
                evaluate_fn=self._evaluate,
            )
            # ... improvement check, same as current ...
```

#### 2.5 Backward Compatibility Shim

```python
# shared/evolutionary/lora_surgery.py (KEPT — becomes re-export shim)

"""Backward compatibility — import from shared.evolutionary.surgery instead."""

from shared.evolutionary.surgery import (
    LoRASurgeon,
    SurgeryConfig,
    OperationResult,
    SurgeryResult,
)
from shared.evolutionary.surgery.utils import (
    _check_dependencies,
    _copy_adapter,
    _find_lora_pairs,
    _find_safetensor_files,
    _get_layer_indices,
    _get_module_types,
    _is_attention_key,
    _is_mlp_key,
    _load_adapter_config,
    _load_all_weights,
    _save_adapter_config,
    _save_all_weights,
    _softmax,
)

__all__ = [
    "LoRASurgeon", "SurgeryConfig", "OperationResult", "SurgeryResult",
    # Helpers re-exported for backward compat
    "_check_dependencies", "_copy_adapter", "_find_lora_pairs",
    "_find_safetensor_files", "_get_layer_indices", "_get_module_types",
    "_is_attention_key", "_is_mlp_key", "_load_adapter_config",
    "_load_all_weights", "_save_adapter_config", "_save_all_weights", "_softmax",
]
```

This ensures `tests/test_lora_surgery.py` and external consumers continue to work unchanged during migration.

---

## 3. Utility Extraction

### `shared/evolutionary/surgery/utils.py`

All 12 helper functions move here, renamed from private (`_` prefix) to public within the module. The backward-compat shim in `lora_surgery.py` re-exports them with their original `_` prefixed names.

| Function | Current Location | Consumers |
|----------|-----------------|-----------|
| `_check_dependencies` | Module-level | `LoRASurgeon.__init__`, `_load_all_weights`, `_save_all_weights` |
| `_load_adapter_config` | Module-level | 5 operations, tests |
| `_save_adapter_config` | Module-level | 2 operations, tests |
| `_find_safetensor_files` | Module-level | `_load_all_weights` |
| `_load_all_weights` | Module-level | 6 operations, tests |
| `_save_all_weights` | Module-level | 6 operations, tests |
| `_copy_adapter` | Module-level | All operations, `run_surgery`, tests |
| `_get_layer_indices` | Module-level | `layer_scaling`, tests |
| `_get_module_types` | Module-level | `module_ablation`, tests |
| `_is_attention_key` | Module-level | `attention_mlp_ablation`, tests |
| `_is_mlp_key` | Module-level | `attention_mlp_ablation`, tests |
| `_softmax` | Module-level | `metrics_weighted_merge`, tests |
| `_find_lora_pairs` | After class | `svd_rank_reduction`, tests |

---

## 4. Test Refactoring Strategy

### Current Test Structure (828 lines)

| Test Class | Tests | Tests Against |
|------------|-------|---------------|
| `TestSurgeryConfig` | 4 | `SurgeryConfig` dataclass |
| `TestHelpers` | 11 | All 12 helper functions |
| `TestAlphaSweep` | 2 | `alpha_sweep` operation |
| `TestLayerScaling` | 3 | `layer_scaling` operation |
| `TestModuleAblation` | 1 | `module_ablation` operation |
| `TestCheckpointInterpolation` | 3 | `checkpoint_interpolation` operation |
| `TestDAREDropRescale` | 2 | `dare_drop_rescale` operation |
| `TestSVDRankReduction` | 2 | `svd_rank_reduction` operation |
| `TestAttentionMLPAblation` | 1 | `attention_mlp_ablation` operation |
| `TestMetricsWeightedMerge` | 2 | `metrics_weighted_merge` operation |
| `TestSurgeryLoop` | 4 | `run_surgery()` orchestration |
| `TestOperationOrdering` | 2 | Operation execution order |
| `TestEdgeCases` | 2 | Edge cases |

### Migration Approach

**Phase 1 (backward-compat shim)**: No test changes needed. The shim re-exports everything from `lora_surgery.py`, so all existing imports continue to work.

**Phase 2 (incremental test migration)**: As a separate follow-up task, update test imports to point to new locations:
- `TestSurgeryConfig` → import from `shared.evolutionary.surgery.config`
- `TestHelpers` → import from `shared.evolutionary.surgery.utils`
- Operation tests → can test operations directly via `SurgeryOperation.execute()`
- `TestSurgeryLoop` / `TestOperationOrdering` → still test `LoRASurgeon`

The test migration is low-priority because the shim makes it non-breaking.

---

## 5. Implementation Roadmap

### Step 1: Create package skeleton + utils
**Files created**: `surgery/__init__.py`, `surgery/base.py`, `surgery/config.py`, `surgery/utils.py`, `surgery/operations/__init__.py`
**What moves**: Dataclasses to `config.py`, helpers to `utils.py`, `SurgeryOperation` protocol to `base.py`
**Risk**: Low. No behavior changes, only code movement.
**Verification**: Run existing tests against backward-compat shim.

### Step 2: Extract operations one at a time
**Order** (simplest first, each validates the pattern works):
1. `alpha_sweep` — simplest (config-only, no weight loading)
2. `module_ablation` — straightforward weight zeroing
3. `dare_drop_rescale` — simple weight transformation
4. `layer_scaling` — nested loops but straightforward
5. `attention_mlp_ablation` — two fixed variants
6. `checkpoint_interpolation` — depends on external checkpoint config
7. `metrics_weighted_merge` — depends on multi-checkpoint config
8. `svd_rank_reduction` — most complex (SVD math, pair matching, rank update)

**Per operation**:
1. Create `surgery/operations/<name>.py` with the operation class
2. Decorate with `@register_operation`
3. Import in `surgery/operations/__init__.py`
4. Remove method from `LoRASurgeon`
5. Run tests — they should pass via the backward-compat shim

### Step 3: Refactor LoRASurgeon to thin orchestrator
**What changes**: Remove `_OPERATION_METHODS` dict, replace `getattr` dispatch with `get_operation()` calls. Remove now-empty operation methods.
**File**: `surgery/surgeon.py`
**Risk**: Medium. The dispatch mechanism changes, but behavior should be identical.
**Verification**: `TestSurgeryLoop` and `TestOperationOrdering` validate orchestration behavior.

### Step 4: Convert lora_surgery.py to backward-compat shim
**What changes**: Replace 1,054-line file with ~30-line re-export module.
**Risk**: Low. All tests should already pass from steps 1-3.

### Step 5 (optional follow-up): Migrate test imports
**What changes**: Update `tests/test_lora_surgery.py` imports to new paths.
**Risk**: Low. Purely cosmetic — the shim already works.

---

## 6. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Circular imports between `surgery/` submodules | Medium | Build breaks | Use `TYPE_CHECKING` guards; keep `utils.py` dependency-free |
| Async context lost in operation extraction | Low | Runtime errors | Operations receive `evaluate_fn` as async callable; test each extraction |
| `SurgeryConfig` field access patterns differ per operation | Low | Subtle bugs | Each operation receives the full config; no API change |
| Backward-compat shim missed import | Low | Import errors | Automated grep verification: all symbols in `tests/test_lora_surgery.py` import block must be re-exported |
| `__init__.py` lazy import in `shared/evolutionary/` breaks | Medium | Import errors | Update lazy import to point to `surgery/` package |
| Registry not populated if operations not imported | Medium | Runtime "unknown operation" errors | `operations/__init__.py` eagerly imports all operation modules |

---

## 7. Reasoning Chain

The refactoring follows from identifying that the `LoRASurgeon` class violates Single Responsibility: it is simultaneously an orchestrator (sequencing operations, tracking state, saving reports) and a collection of 8 unrelated algorithm implementations. The Strategy pattern is the natural fit because:

1. All operations share an identical interface (`adapter_path`, `baseline_score` in; `OperationResult` out).
2. Operations are selected at runtime from a config list — classic Strategy selection.
3. The existing `shared/evolutionary/strategies/` package already established the exact same decomposition pattern in this codebase, so the approach is idiomatic rather than novel.

The decision to use `Protocol` over `ABC` follows the codebase convention (`EvalBackend` is a `Protocol`). The decision to use static registration over dynamic discovery reflects the bounded set of operations — there's no plugin extensibility requirement. The backward-compat shim preserves all existing imports, making the refactoring fully non-breaking and allowing incremental test migration.

The extraction order (simplest first) reduces risk: `alpha_sweep` validates the full pattern end-to-end with the least complex operation, giving confidence before tackling `svd_rank_reduction` which has the most intricate weight manipulation logic.

---

## 8. Estimated File Sizes After Refactoring

| File | Estimated Lines | Notes |
|------|----------------|-------|
| `surgery/__init__.py` | ~60 | Registry + public API |
| `surgery/base.py` | ~25 | Protocol definition |
| `surgery/config.py` | ~120 | 3 dataclasses (moved as-is) |
| `surgery/utils.py` | ~130 | 13 helper functions |
| `surgery/surgeon.py` | ~120 | Thin orchestrator |
| `surgery/operations/__init__.py` | ~20 | Imports all operations |
| `surgery/operations/alpha_sweep.py` | ~60 | |
| `surgery/operations/layer_scaling.py` | ~80 | |
| `surgery/operations/module_ablation.py` | ~65 | |
| `surgery/operations/checkpoint_interpolation.py` | ~80 | |
| `surgery/operations/dare_drop_rescale.py` | ~70 | |
| `surgery/operations/metrics_weighted_merge.py` | ~70 | |
| `surgery/operations/svd_rank_reduction.py` | ~110 | |
| `surgery/operations/attention_mlp_ablation.py` | ~95 | |
| `lora_surgery.py` (shim) | ~30 | Backward-compat re-exports |
| **Total** | ~1,135 | Slight increase from boilerplate, but each file is focused |

All files well under the 500-600 line maintainability guideline.
