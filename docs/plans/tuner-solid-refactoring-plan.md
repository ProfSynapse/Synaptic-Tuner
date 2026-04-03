# Tuner Module SOLID Refactoring Plan

## Overview

Two files in the `tuner/` module violate SOLID principles and exceed recommended size limits:

| File | Lines | Violation | Classes/Concerns |
|------|-------|-----------|-----------------|
| `tuner/handlers/experiment_handler.py` | 946 | 4 classes in 1 file (SRP) | HFTrainingStageRunner, HFEvalStageRunner, HFLossStageRunner, ExperimentHandler |
| `tuner/backends/training/cloud/hf_jobs_backend.py` | 990 | God class (SRP) | 30+ methods spanning 6 distinct concerns |

This plan proposes incremental extraction into focused modules, preserving all existing APIs.

---

## File 1: experiment_handler.py — Stage Runner Extraction

### Current Structure

```
experiment_handler.py (946 lines)
├── _optional_backend_value()         # helper (lines 42-47)
├── HFTrainingStageRunner             # lines 50-318  (~269 lines)
├── HFEvalStageRunner                 # lines 321-423 (~103 lines)
├── HFLossStageRunner                 # lines 426-763 (~338 lines)
└── ExperimentHandler(BaseHandler)    # lines 766-946 (~181 lines)
```

### Analysis

Each stage runner is already a self-contained class with:
- Its own `__init__(*, repo_root, tracking_service)`
- A `run(spec, experiment, previous)` method conforming to the `StageRunner` Protocol
- Private helpers scoped to its stage
- No cross-references between runners (Training -> Eval -> Loss is sequenced by ExperimentOrchestrator, not by the runners themselves)

`ExperimentHandler` instantiates all three runners and passes them to `ExperimentOrchestrator`. It does not call runner internals directly.

**This is a clean, low-risk extraction.** The boundaries already exist; they just need separate files.

### Proposed Structure

```
tuner/handlers/
├── experiment_handler.py              # ExperimentHandler only (~200 lines)
├── stages/
│   ├── __init__.py                    # Re-exports all runners
│   ├── _util.py                       # _optional_backend_value helper
│   ├── hf_training_stage.py           # HFTrainingStageRunner (~270 lines)
│   ├── hf_eval_stage.py              # HFEvalStageRunner (~105 lines)
│   └── hf_loss_stage.py             # HFLossStageRunner (~340 lines)
```

### Dependency Map

```
ExperimentHandler (experiment_handler.py)
    ├── imports HFTrainingStageRunner  from .stages
    ├── imports HFEvalStageRunner      from .stages
    ├── imports HFLossStageRunner      from .stages
    └── passes them to ExperimentOrchestrator (shared/experiment_tracking/)

Each stage runner imports:
    ├── shared.experiment_tracking (Experiment, ExperimentSpec, StageResult, etc.)
    ├── shared.utilities.env (get_hf_token)
    ├── tuner.cloud (CloudJobSpec, HFJobExecutor, etc.)
    ├── tuner.core.exceptions (CloudProviderError)
    └── tuner.backends.registry (TrainingBackendRegistry) — Training + Loss only
```

No circular dependencies. Each runner only depends on shared infrastructure and `tuner.cloud`.

### Blast Radius

| Consumer | Current Import | Migration |
|----------|---------------|-----------|
| `tests/cloud/test_experiment_handler.py` | `from tuner.handlers.experiment_handler import HFEvalStageRunner, HFLossStageRunner, HFTrainingStageRunner, StageResult` | Update to `from tuner.handlers.stages import ...` (or keep backward-compat re-export) |
| `tuner/cli/router.py` | `from tuner.handlers.experiment_handler import ExperimentHandler` | No change needed |
| `tests/cloud/test_hardware_planner.py` | `from tuner.handlers.experiment_handler import ExperimentHandler` | No change needed |

**Mitigation**: Add backward-compatible re-exports in `experiment_handler.py` during migration, remove in a follow-up PR.

### Migration Steps

1. Create `tuner/handlers/stages/` package with `__init__.py`
2. Move `_optional_backend_value` to `stages/_util.py`
3. Move `HFTrainingStageRunner` to `stages/hf_training_stage.py`
4. Move `HFEvalStageRunner` to `stages/hf_eval_stage.py`
5. Move `HFLossStageRunner` to `stages/hf_loss_stage.py`
6. Update `stages/__init__.py` to re-export all runners
7. Update `experiment_handler.py` imports to use `from .stages import ...`
8. Add backward-compat re-exports in `experiment_handler.py` for external consumers
9. Update test imports to use `from tuner.handlers.stages import ...`
10. Run tests to verify

---

## File 2: hf_jobs_backend.py — Concern Extraction

### Current Structure

```
hf_jobs_backend.py (990 lines)
└── HFJobsBackend(ITrainingBackend)
    ├── INTERFACE (ITrainingBackend contract)
    │   ├── name property
    │   ├── get_available_methods()
    │   ├── validate_environment()
    │   ├── load_config(method) -> CloudTrainingConfig
    │   └── execute(config, python_path) -> ExecuteResult
    │
    ├── COMMAND BUILDING (~200 lines)
    │   ├── _build_training_command()        # lines 750-914
    │   ├── _build_env_grpo_steps()          # lines 916-952
    │   ├── _build_artifact_prefix()         # lines 743-748
    │   ├── _config_filename_for_method()
    │   └── _script_name_for_method()
    │
    ├── JOB WATCHING (~175 lines)
    │   ├── _should_use_remote_dashboard()   # lines 461-468
    │   ├── _watch_job_with_remote_dashboard() # lines 470-562
    │   └── _update_dashboard_from_local_log() # lines 564-576
    │
    ├── BUCKET MANAGEMENT (~40 lines)
    │   ├── _ensure_hf_bucket()              # lines 578-594
    │   ├── _build_remote_run_uri()          # lines 596-600
    │   └── _sync_bucket_path()              # lines 607-613
    │
    ├── ARTIFACT DOWNLOAD (~65 lines)
    │   ├── _download_completed_run()        # lines 623-637
    │   ├── _recover_completed_run_from_bucket() # lines 639-658
    │   ├── _run_dir_has_completion_artifacts() # lines 615-621
    │   └── _local_download_run_dir()        # lines 602-605
    │
    ├── POST-TRAINING ACTIONS (~65 lines)
    │   ├── _handle_post_training_actions()  # lines 676-725
    │   ├── _print_completion_summary()      # lines 660-674
    │   └── _finalize_completed_job()        # lines 726-741
    │
    └── CONFIG LOADING (~115 lines)
        └── load_config()                    # lines 187-305

+ _parse_timeout() module-level helper      # lines 955-990
```

### Analysis

Unlike experiment_handler.py, these concerns are **more intertwined**:
- `execute()` calls command building, bucket management, job watching, artifact recovery, and finalization
- `_watch_job_with_remote_dashboard()` calls `_update_dashboard_from_local_log()` AND bucket sync
- `_finalize_completed_job()` calls both `_print_completion_summary()` and `_handle_post_training_actions()`

However, the concern groups have **clear cohesion within** and **loose coupling between**:
- Command building only needs config + repo structure
- Job watching only needs HF hub API + dashboard
- Bucket/artifact operations only need HF hub + file system
- Post-training actions only need config + local paths + UI

### Proposed Structure: Mixin Extraction

Since `HFJobsBackend` must remain a single class for `ITrainingBackend` compatibility and the registry system, use **mixin classes** to separate concerns while keeping one unified class:

```
tuner/backends/training/cloud/
├── hf_jobs_backend.py                 # HFJobsBackend (facade, ~150 lines)
│                                      # Inherits from mixins + ITrainingBackend
├── _hf_command_builder.py             # HFCommandBuilderMixin (~220 lines)
│                                      # _build_training_command, _build_env_grpo_steps,
│                                      # _build_artifact_prefix, method/config name helpers
├── _hf_job_watcher.py                 # HFJobWatcherMixin (~180 lines)
│                                      # _watch_job_with_remote_dashboard,
│                                      # _update_dashboard_from_local_log,
│                                      # _should_use_remote_dashboard
├── _hf_bucket_ops.py                  # HFBucketOpsMixin (~120 lines)
│                                      # _ensure_hf_bucket, _build_remote_run_uri,
│                                      # _sync_bucket_path, _download_completed_run,
│                                      # _recover_completed_run_from_bucket,
│                                      # _run_dir_has_completion_artifacts,
│                                      # _local_download_run_dir
├── _hf_post_training.py              # HFPostTrainingMixin (~70 lines)
│                                      # _handle_post_training_actions,
│                                      # _print_completion_summary,
│                                      # _finalize_completed_job
├── base_cloud.py                      # (unchanged)
├── modal_backend.py                   # (unchanged)
└── runpod_backend.py                  # (unchanged)
```

The facade class becomes:

```python
class HFJobsBackend(
    HFCommandBuilderMixin,
    HFJobWatcherMixin,
    HFBucketOpsMixin,
    HFPostTrainingMixin,
    ITrainingBackend,
):
    """HuggingFace Jobs training backend."""

    def __init__(self, repo_root: Path):
        self.repo_root = Path(repo_root)
        self.show_post_training_actions = True
        self.last_artifact_prefix: Optional[str] = None
        self.last_bucket_id: Optional[str] = None
        self.last_job_id: Optional[str] = None

    # name, get_available_methods, validate_environment, load_config, execute
    # ~150 lines of interface methods + execute orchestration
```

### Why Mixins Over Composition

| Approach | Pros | Cons |
|----------|------|------|
| **Mixins** | Zero API change; no constructor changes; tests work as-is; methods still access `self.repo_root` etc. naturally | Mixins can be harder to reason about if abused; Python MRO considerations |
| **Composition** | Cleaner separation; explicit dependencies | Requires refactoring `self.xyz` access patterns; constructor changes; more test changes |
| **Strategy pattern** | Each concern is testable independently | Heavy refactor; requires dependency injection framework-level changes |

Mixins are the pragmatic choice here because:
1. All methods currently access `self.repo_root`, `self.last_*` etc. — mixins preserve this naturally
2. External API (`HFJobsBackend` class) stays identical
3. Tests need zero import changes
4. The underscore-prefixed (`_`) leading convention already signals internal concerns

### Alternative Considered: Standalone Helper Modules

Instead of mixins, extract to standalone functions that receive explicit parameters:

```python
# _hf_command_builder.py
def build_training_command(config: CloudTrainingConfig, repo_root: Path, timestamp: str) -> str: ...
```

**Trade-off**: Cleaner dependency graph but requires converting ~30 methods from `self.xyz` access to explicit parameter passing. Higher effort, more changes to tests. This could be a Phase 2 refactor after mixins prove the boundaries are correct.

### Blast Radius

| Consumer | Current Import | Migration |
|----------|---------------|-----------|
| `tests/cloud/test_hf_jobs_backend.py` | `from tuner.backends.training.cloud.hf_jobs_backend import HFJobsBackend, _parse_timeout` | No change — `HFJobsBackend` stays in same module |
| `tests/cloud/test_cloud_deps.py` | `from tuner.backends.training.cloud.hf_jobs_backend import HFJobsBackend` | No change |
| `tuner/backends/training/cloud/__init__.py` | `from .hf_jobs_backend import HFJobsBackend` | No change |
| `tuner/backends/registry.py` | `from tuner.backends.training.cloud import HFJobsBackend` | No change |

**Zero external API changes.** All consumers import `HFJobsBackend` which remains the same class in the same module. The mixins are internal implementation detail.

### Migration Steps

1. Create `_hf_command_builder.py` with `HFCommandBuilderMixin`; move command-building methods
2. Create `_hf_job_watcher.py` with `HFJobWatcherMixin`; move dashboard/watching methods
3. Create `_hf_bucket_ops.py` with `HFBucketOpsMixin`; move bucket and artifact methods
4. Create `_hf_post_training.py` with `HFPostTrainingMixin`; move finalization methods
5. Update `HFJobsBackend` to inherit from all mixins
6. Keep `load_config()`, `execute()`, `validate_environment()` in the facade (these are the orchestration)
7. Keep `_parse_timeout()` as module-level in `hf_jobs_backend.py` (simple utility)
8. Run tests to verify

---

## Test File Strategy

### test_experiment_handler.py (958 lines)

After stage runner extraction, split tests to match:

```
tests/cloud/
├── test_experiment_handler.py         # ExperimentHandler tests only (~30 lines)
├── stages/
│   ├── __init__.py
│   ├── test_hf_training_stage.py     # Training runner tests (~400 lines)
│   ├── test_hf_eval_stage.py         # Eval runner tests (~250 lines)
│   └── test_hf_loss_stage.py         # Loss runner tests (~300 lines)
```

### test_hf_jobs_backend.py (580 lines)

No split needed for the mixin refactor — tests import `HFJobsBackend` which remains the same class. Future phase could add targeted mixin tests, but this is not required since the class-level tests already exercise all paths.

---

## Implementation Roadmap

### PR 1: Stage Runner Extraction (experiment_handler.py)
- **Scope**: Create `tuner/handlers/stages/` package, move 3 runner classes
- **Risk**: Low — classes are already independent
- **Tests**: Update imports in `test_experiment_handler.py`; add backward-compat re-exports
- **Estimated complexity**: ~2 hours

### PR 2: HFJobsBackend Mixin Extraction (hf_jobs_backend.py)
- **Scope**: Create 4 mixin modules, refactor `HFJobsBackend` to compose them
- **Risk**: Low-Medium — need to verify MRO and `self` attribute access works correctly across mixins
- **Tests**: No import changes needed; add a smoke test verifying MRO resolution
- **Estimated complexity**: ~3 hours
- **Depends on**: Independent of PR 1 (can run in parallel)

### PR 3 (Optional): Test File Reorganization
- **Scope**: Split `test_experiment_handler.py` to mirror source structure
- **Risk**: Low — mechanical import updates
- **Depends on**: PR 1

### PR 4 (Future): Mixin-to-Composition Migration
- **Scope**: Convert mixins to composed helper objects with explicit dependency injection
- **Risk**: Medium — requires constructor changes and parameter threading
- **Depends on**: PR 2 proves boundaries are correct
- **Deferred**: Only pursue if mixins cause maintenance issues

---

## Reasoning Chain

1. **Identified natural boundaries**: experiment_handler.py already has 4 classes with zero cross-references — the file is literally 4 modules concatenated into one. This is the easy win.

2. **Chose extraction over restructuring for stage runners**: Since the `StageRunner` Protocol already defines the interface, and `ExperimentOrchestrator` already depends on abstractions (Protocol), the runners can move without any design changes.

3. **Chose mixins for HFJobsBackend over composition**: The God class pattern in hf_jobs_backend.py has 30+ methods that all access `self.repo_root` and `self.last_*` state. Mixins preserve this access pattern with zero refactoring of method signatures. Composition would require threading state through constructors or adding a shared context object — higher effort with no immediate benefit.

4. **Preserved external API completely**: Both refactors maintain identical import paths for all external consumers. The `__init__.py` files and re-exports ensure backward compatibility.

5. **Ordered PRs by risk**: Stage runner extraction (PR 1) is near-zero risk and delivers the bigger clarity win. HFJobsBackend mixins (PR 2) are slightly higher risk due to MRO considerations but still low because all methods are `_private` and only called by `execute()`.

6. **Deferred composition**: Converting mixins to composed objects is the "proper" SOLID solution but doubles the effort. The mixin step proves the boundaries are correct, and composition can follow if needed.

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Circular imports in stages package | Low | Medium | Runners have no cross-references; shared imports go through `tuner.cloud` |
| MRO issues with mixins | Low | Medium | All mixins are leaf classes with no inheritance hierarchy; tested with MRO smoke test |
| Test breakage from import changes | Medium | Low | Backward-compat re-exports; mechanical fix |
| Hidden coupling between HF backend methods | Low | Medium | Thorough read of all methods confirms clean concern boundaries |
| `_parse_timeout` ambiguity | Very Low | Very Low | Keep in `hf_jobs_backend.py` as module-level; it has no class dependency |
