# Tuner Discovery Services Implementation

**Date:** 2025-11-30
**Status:** Completed
**Phase:** Backend Coding (PACT Framework)

---

## Executive Summary

This document summarizes the implementation of the discovery services layer for the Synaptic Tuner CLI refactoring project. The discovery layer provides a clean, testable interface for enumerating resources (training runs, checkpoints, models, and prompt sets) needed by the tuner handlers.

**Implementation Scope:**
- Created 5 new Python modules totaling ~650 lines of code
- Migrated discovery logic from monolithic `tuner.py` (lines 220-780)
- Implemented consistent patterns following SOLID principles
- Provided comprehensive documentation and examples

---

## Files Implemented

### 1. `/mnt/f/Code/Toolset-Training/tuner/discovery/__init__.py`
**Purpose:** Package initialization and exports
**Lines of Code:** 26
**Key Exports:**
- `TrainingRunDiscovery`
- `CheckpointDiscovery`
- `ModelDiscovery`
- `PromptSetDiscovery`

### 2. `/mnt/f/Code/Toolset-Training/tuner/discovery/training_runs.py`
**Purpose:** Discover training runs for a trainer type
**Lines of Code:** 122
**Key Features:**
- Scans `Trainers/rtx3090_{trainer_type}/{trainer_type}_output_rtx3090/`
- Filters runs with `final_model` or `checkpoints` directories
- Sorts by modification time (newest first)
- Supports configurable limit

**Pattern Migrated From:** `tuner.py` lines 220-234 (`list_training_runs` function)

**Usage Example:**
```python
from tuner.discovery import TrainingRunDiscovery

discovery = TrainingRunDiscovery(repo_root=Path('/path/to/repo'))
sft_runs = discovery.discover('sft', limit=10)
```

### 3. `/mnt/f/Code/Toolset-Training/tuner/discovery/checkpoints.py`
**Purpose:** Discover checkpoints and load training metrics
**Lines of Code:** 206
**Key Features:**
- `load_metrics()`: Parses `logs/training_*.jsonl` to extract step metrics
- `discover()`: Finds all checkpoints in a training run
- Returns `CheckpointInfo` objects with path, step, metrics, and is_final flag
- Sorts checkpoints (final model first, then by step number)

**Pattern Migrated From:** `tuner.py` lines 330-364 (`_load_checkpoint_metrics` and `_detect_training_type` functions)

**Usage Example:**
```python
from tuner.discovery import CheckpointDiscovery

discovery = CheckpointDiscovery()
checkpoints = discovery.discover(run_dir)

# Find best checkpoint by score
best = max(checkpoints, key=lambda cp: cp.score('kto'))
```

### 4. `/mnt/f/Code/Toolset-Training/tuner/discovery/models.py`
**Purpose:** Discover available models from evaluation backends
**Lines of Code:** 105
**Key Features:**
- Queries evaluation backends (Ollama, LM Studio) via registry
- Returns list of model names/identifiers
- Graceful error handling (returns empty list on failure)
- Unified interface across different backends

**Pattern Migrated From:** `tuner.py` lines 705-746 (`_list_ollama_models` and `_list_lmstudio_models` functions)

**Usage Example:**
```python
from tuner.discovery import ModelDiscovery

discovery = ModelDiscovery()
ollama_models = discovery.discover('ollama')
lms_models = discovery.discover('lmstudio')
```

### 5. `/mnt/f/Code/Toolset-Training/tuner/discovery/prompt_sets.py`
**Purpose:** Discover and parse prompt sets for evaluation
**Lines of Code:** 191
**Key Features:**
- Scans `Evaluator/prompts/` for known prompt set JSON files
- Parses JSON to count prompts (supports list and dict formats)
- Returns tuples of (name, description, count)
- Maintains display order from `KNOWN_PROMPT_SETS`

**Pattern Migrated From:** `tuner.py` lines 749-780 (`_list_prompt_sets` function)

**Usage Example:**
```python
from tuner.discovery import PromptSetDiscovery

discovery = PromptSetDiscovery()
prompt_sets = discovery.discover()

for name, description, count in prompt_sets:
    print(f"{name}: {description} ({count} prompts)")
```

---

## Architecture Alignment

The discovery layer implementation follows the architecture specified in `/mnt/f/Code/Toolset-Training/docs/tuner_architecture.md` section 4.3 (Discovery Services).

### Key Design Patterns

1. **Service Locator Pattern**
   - Discovery services find resources independently
   - Handlers query discovery services as needed
   - No tight coupling between handlers and resource locations

2. **Dependency Injection**
   - All services accept `repo_root` parameter (defaults to auto-detection)
   - Enables testing with custom paths
   - No hardcoded absolute paths

3. **Graceful Degradation**
   - Empty lists returned when resources not found
   - No exceptions thrown for missing directories
   - Consistent error handling across all services

4. **Single Responsibility**
   - Each service has one clear purpose
   - TrainingRunDiscovery: Find runs
   - CheckpointDiscovery: Load metrics and find checkpoints
   - ModelDiscovery: Query backends for models
   - PromptSetDiscovery: Parse prompt set files

### Integration Points

**Imports from Core:**
- `tuner.core.config.CheckpointInfo` (used by CheckpointDiscovery)

**Imports from Backends:**
- `tuner.backends.registry.EvaluationBackendRegistry` (used by ModelDiscovery)

**Used By (Future Integration):**
- `tuner.handlers.upload_handler` (will use TrainingRunDiscovery and CheckpointDiscovery)
- `tuner.handlers.eval_handler` (will use ModelDiscovery and PromptSetDiscovery)

---

## Testing Recommendations

The following tests should be implemented to verify the discovery services:

### Unit Tests

**File:** `tests/test_discovery/test_training_runs.py`

```python
def test_discover_sft_runs():
    """Test discovering SFT training runs."""
    discovery = TrainingRunDiscovery(repo_root=test_repo_root)
    runs = discovery.discover('sft', limit=5)
    assert len(runs) <= 5
    assert all(r.is_dir() for r in runs)

def test_discover_with_no_output_dir():
    """Test discovery when output directory doesn't exist."""
    discovery = TrainingRunDiscovery(repo_root=Path('/nonexistent'))
    runs = discovery.discover('sft')
    assert runs == []

def test_discover_filters_runs_without_models():
    """Test that runs without final_model or checkpoints are excluded."""
    # Setup mock directory structure
    # Assert only valid runs are returned
    pass
```

**File:** `tests/test_discovery/test_checkpoints.py`

```python
def test_load_metrics():
    """Test loading metrics from training logs."""
    metrics = CheckpointDiscovery.load_metrics(test_run_dir)
    assert isinstance(metrics, dict)
    assert all(isinstance(k, int) for k in metrics.keys())

def test_discover_checkpoints():
    """Test discovering checkpoints in a run."""
    checkpoints = CheckpointDiscovery.discover(test_run_dir)
    assert len(checkpoints) > 0
    assert any(cp.is_final for cp in checkpoints)

def test_checkpoint_score_kto():
    """Test checkpoint scoring for KTO training."""
    cp = CheckpointInfo(
        path=Path('/fake'),
        step=100,
        metrics={'kl': 0.1, 'rewards/margins': 0.3},
        is_final=False
    )
    score = cp.score('kto')
    assert score == 3.0  # 0.3 / 0.1
```

**File:** `tests/test_discovery/test_models.py`

```python
def test_discover_ollama_models():
    """Test discovering Ollama models."""
    models = ModelDiscovery.discover('ollama')
    # Should return list (may be empty if Ollama not running)
    assert isinstance(models, list)

def test_discover_invalid_backend():
    """Test discovery with invalid backend name."""
    models = ModelDiscovery.discover('nonexistent')
    assert models == []
```

**File:** `tests/test_discovery/test_prompt_sets.py`

```python
def test_discover_prompt_sets():
    """Test discovering prompt sets."""
    discovery = PromptSetDiscovery(repo_root=test_repo_root)
    prompt_sets = discovery.discover()
    assert len(prompt_sets) > 0
    assert all(len(ps) == 3 for ps in prompt_sets)  # (name, desc, count)

def test_count_prompts_list_format():
    """Test counting prompts in list format."""
    data = ["prompt1", "prompt2", "prompt3"]
    count = PromptSetDiscovery._count_prompts(data)
    assert count == 3

def test_count_prompts_dict_format():
    """Test counting prompts in dict format."""
    data = {"prompts": ["p1", "p2"], "metadata": {}}
    count = PromptSetDiscovery._count_prompts(data)
    assert count == 2
```

### Integration Tests

**File:** `tests/integration/test_discovery_integration.py`

```python
def test_full_upload_workflow():
    """Test full upload workflow using discovery services."""
    # 1. Discover training runs
    run_discovery = TrainingRunDiscovery(repo_root=REPO_ROOT)
    runs = run_discovery.discover('sft', limit=1)
    assert len(runs) > 0

    # 2. Discover checkpoints in the run
    checkpoint_discovery = CheckpointDiscovery()
    checkpoints = checkpoint_discovery.discover(runs[0])
    assert len(checkpoints) > 0

    # 3. Select best checkpoint
    best = max(checkpoints, key=lambda cp: cp.score('sft'))
    assert best is not None

def test_full_eval_workflow():
    """Test full evaluation workflow using discovery services."""
    # 1. Discover models
    models = ModelDiscovery.discover('ollama')
    # (May be empty if Ollama not running - that's OK)

    # 2. Discover prompt sets
    prompt_discovery = PromptSetDiscovery(repo_root=REPO_ROOT)
    prompt_sets = prompt_discovery.discover()
    assert len(prompt_sets) > 0
```

---

## Verification Instructions for Test Engineer

To verify the discovery services implementation, the test engineer should:

### 1. Code Review
- [ ] Review all 5 implementation files for adherence to SOLID principles
- [ ] Verify comprehensive docstrings and examples in all classes/methods
- [ ] Check error handling patterns (graceful degradation)
- [ ] Confirm consistent naming conventions

### 2. Manual Testing

**Training Run Discovery:**
```bash
cd /mnt/f/Code/Toolset-Training
python3 -c "
from pathlib import Path
from tuner.discovery import TrainingRunDiscovery

discovery = TrainingRunDiscovery(repo_root=Path.cwd())
sft_runs = discovery.discover('sft', limit=10)
print(f'Found {len(sft_runs)} SFT runs')
for run in sft_runs[:3]:
    print(f'  - {run.name}')

kto_runs = discovery.discover('kto', limit=10)
print(f'Found {len(kto_runs)} KTO runs')
"
```

**Checkpoint Discovery:**
```bash
python3 -c "
from pathlib import Path
from tuner.discovery import TrainingRunDiscovery, CheckpointDiscovery

# Find latest SFT run
run_discovery = TrainingRunDiscovery(repo_root=Path.cwd())
runs = run_discovery.discover('sft', limit=1)

if runs:
    run_dir = runs[0]
    print(f'Analyzing run: {run_dir.name}')

    # Load metrics
    metrics = CheckpointDiscovery.load_metrics(run_dir)
    print(f'Loaded metrics for {len(metrics)} steps')

    # Discover checkpoints
    checkpoints = CheckpointDiscovery.discover(run_dir)
    print(f'Found {len(checkpoints)} checkpoints')

    for cp in checkpoints[:3]:
        name = 'final_model' if cp.is_final else f'checkpoint-{cp.step}'
        loss = cp.metrics.get('loss', 'N/A')
        print(f'  {name}: loss={loss}')
"
```

**Model Discovery:**
```bash
python3 -c "
from tuner.discovery import ModelDiscovery

# Discover Ollama models
ollama_models = ModelDiscovery.discover('ollama')
print(f'Ollama models: {len(ollama_models)}')
for model in ollama_models[:5]:
    print(f'  - {model}')

# Discover LM Studio models
lms_models = ModelDiscovery.discover('lmstudio')
print(f'LM Studio models: {len(lms_models)}')
"
```

**Prompt Set Discovery:**
```bash
python3 -c "
from pathlib import Path
from tuner.discovery import PromptSetDiscovery

discovery = PromptSetDiscovery(repo_root=Path.cwd())
prompt_sets = discovery.discover()

print(f'Found {len(prompt_sets)} prompt sets')
for name, description, count in prompt_sets:
    print(f'  {name}: {description} ({count} prompts)')
"
```

### 3. Unit Test Execution

Once unit tests are implemented:

```bash
cd /mnt/f/Code/Toolset-Training

# Run discovery service tests
pytest tests/test_discovery/ -v

# Run with coverage
pytest tests/test_discovery/ --cov=tuner.discovery --cov-report=html
```

### 4. Integration Testing

Test full workflows that combine multiple discovery services:

```bash
# Test upload workflow (requires actual training runs)
pytest tests/integration/test_discovery_integration.py::test_full_upload_workflow -v

# Test eval workflow
pytest tests/integration/test_discovery_integration.py::test_full_eval_workflow -v
```

### 5. Performance Testing

Verify discovery services perform well even with many training runs:

```python
import time
from pathlib import Path
from tuner.discovery import TrainingRunDiscovery, CheckpointDiscovery

# Test training run discovery performance
start = time.time()
discovery = TrainingRunDiscovery(repo_root=Path.cwd())
runs = discovery.discover('sft', limit=100)
elapsed = time.time() - start
print(f"Discovered {len(runs)} runs in {elapsed:.2f}s")
assert elapsed < 1.0, "Discovery should complete in under 1 second"

# Test checkpoint discovery performance
if runs:
    start = time.time()
    checkpoints = CheckpointDiscovery.discover(runs[0])
    elapsed = time.time() - start
    print(f"Discovered {len(checkpoints)} checkpoints in {elapsed:.2f}s")
    assert elapsed < 0.5, "Checkpoint discovery should complete in under 0.5 seconds"
```

---

## Known Limitations

1. **Training Run Discovery:**
   - Only supports RTX 3090 output directories (`rtx3090_{type}`)
   - Mac MLX runs are not currently supported
   - Assumes standard directory naming convention (`YYYYMMDD_HHMMSS`)

2. **Checkpoint Discovery:**
   - Assumes JSONL log format with `step` field
   - Only reads first log file found (should be only one)
   - Final model has empty metrics (no associated step)

3. **Model Discovery:**
   - Depends on backend availability (no offline mode)
   - No caching (queries backend every time)
   - Limited error messages on failure (just returns empty list)

4. **Prompt Set Discovery:**
   - Only discovers known prompt sets from `KNOWN_PROMPT_SETS`
   - New prompt sets require code changes
   - No dynamic discovery of unknown prompt sets

---

## Future Enhancements

### Phase 1: Mac Training Run Support
Add support for discovering Mac MLX training runs:
- Extend `TrainingRunDiscovery.discover()` to handle Mac output directories
- Add platform parameter: `discover(trainer_type, platform='rtx')`

### Phase 2: Caching
Implement caching for expensive discovery operations:
- Cache model lists from backends (invalidate after 5 minutes)
- Cache checkpoint metrics (invalidate when log file changes)

### Phase 3: Dynamic Prompt Set Discovery
Remove hardcoded `KNOWN_PROMPT_SETS`:
- Scan directory for all `*.json` files
- Parse first few lines to detect prompt set format
- Auto-generate descriptions from file metadata

### Phase 4: Error Reporting
Improve error handling and reporting:
- Return `(success, data, error_message)` tuples instead of just data
- Log warnings for malformed files
- Provide actionable error messages to users

---

## Dependencies

**Python Standard Library:**
- `json`: Parsing JSONL logs and prompt set JSON files
- `pathlib`: Cross-platform path handling
- `typing`: Type hints (List, Tuple, Dict)

**Internal Dependencies:**
- `tuner.core.config`: CheckpointInfo dataclass
- `tuner.backends.registry`: EvaluationBackendRegistry

**No External Dependencies Required**

---

## Summary

The discovery services layer has been successfully implemented following the architecture specification. All services provide:

- **Clean interfaces**: Simple, focused methods with clear responsibilities
- **Comprehensive documentation**: Docstrings, examples, and inline comments
- **Error handling**: Graceful degradation with empty lists on failure
- **Testability**: Static methods and dependency injection enable easy testing
- **Consistency**: Uniform patterns across all discovery services

The implementation is ready for integration with the handlers layer and comprehensive testing.

---

**Next Steps for Orchestrator:**
1. Have test engineer review this document
2. Have test engineer execute verification instructions (section 6)
3. Have test engineer implement unit tests (section 5)
4. Proceed to next phase: Handlers implementation

**Test Engineer Instructions:**
Read this document (`/mnt/f/Code/Toolset-Training/docs/tuner_discovery_implementation.md`) and execute the verification steps in section 6 to validate the discovery services implementation.
