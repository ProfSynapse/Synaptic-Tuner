# GPU Cloud Provider Integration Architecture

**Date**: 2026-02-20
**Phase**: ARCHITECT
**Upstream**: [GPU Cloud Providers Research](../preparation/gpu-cloud-providers-research.md)

---

## 1. Executive Summary

This document specifies the architecture for integrating three GPU cloud providers (HuggingFace Jobs, Modal, RunPod) into the Toolset-Training CLI. The design follows the existing patterns established by the `tuner/` package: a new `CloudTrainHandler` routes to provider-specific backends registered in the existing `TrainingBackendRegistry`, reusing the same `ITrainingBackend` interface used by `RTXBackend` and `MacBackend`.

**Key design decisions**:

1. **Extend, don't replace**: Cloud providers are new training backends, registered alongside `rtx` and `mac` in the existing registry. No new abstraction layer.
2. **Handler-based CLI integration**: A new `CloudTrainHandler` sits alongside `TrainHandler` in the menu, keeping cloud training logically separate from local training without duplicating menu infrastructure.
3. **Provider files live in `tuner/backends/training/cloud/`**: Follows the existing backend module pattern rather than creating a top-level `cloud/` directory.
4. **Thin provider wrappers**: Each provider backend is a single file implementing `ITrainingBackend`. Complexity stays in the provider SDKs where it belongs.
5. **Cloud config extends existing YAML**: A `cloud.yaml` config file alongside `config.yaml` in each trainer directory holds provider-specific settings (GPU type, timeout, etc.).

---

## 2. System Context

```
+-------------------+       +---------------------------+       +-------------------+
|                   |       |                           |       |                   |
|   User (CLI)      +------>+   Synaptic Tuner CLI      +------>+   Local GPU       |
|                   |       |   (tuner.py / run.sh)     |       |   (RTX 3090)      |
+-------------------+       |                           |       +-------------------+
                            |   tuner/handlers/         |
                            |   tuner/backends/         |       +-------------------+
                            |                           +------>+   HuggingFace     |
                            |                           |       |   Jobs API        |
                            |                           |       +-------------------+
                            |                           |
                            |                           |       +-------------------+
                            |                           +------>+   Modal           |
                            |                           |       |   Platform        |
                            |                           |       +-------------------+
                            |                           |
                            |                           |       +-------------------+
                            |                           +------>+   RunPod           |
                            |                           |       |   API             |
                            +---------------------------+       +-------------------+
                                        |
                                        v
                            +---------------------------+
                            |   HuggingFace Hub         |
                            |   (model output target)   |
                            +---------------------------+
```

All three cloud providers ultimately push trained model artifacts to HuggingFace Hub. The CLI orchestrates job submission, status polling, and log streaming locally.

---

## 3. Component Architecture

### 3.1 File Structure

New files are shown with `[NEW]`. Existing files shown for context.

```
tuner/
  backends/
    training/
      __init__.py                        # [MODIFY] Add cloud backend exports
      base.py                            # [EXISTING] Re-exports ITrainingBackend
      rtx_backend.py                     # [EXISTING]
      mac_backend.py                     # [EXISTING]
      cloud/                             # [NEW] Cloud provider backends
        __init__.py                      # [NEW] Export all cloud backends
        base_cloud.py                    # [NEW] Shared cloud backend utilities
        hf_jobs_backend.py              # [NEW] HuggingFace Jobs backend
        modal_backend.py                # [NEW] Modal backend
        runpod_backend.py               # [NEW] RunPod backend
    registry.py                          # [MODIFY] Register cloud backends
  handlers/
    cloud_train_handler.py              # [NEW] Cloud training handler
    main_menu_handler.py                # [MODIFY] Add cloud training menu option
  cli/
    parser.py                            # [MODIFY] Add 'cloud' command
    router.py                            # [MODIFY] Route 'cloud' command
  core/
    config.py                            # [MODIFY] Add CloudTrainingConfig
    exceptions.py                        # [MODIFY] Add CloudProviderError

Trainers/
  cloud/                                 # [NEW] Cloud-specific training assets
    cloud_config.yaml                    # [NEW] Default cloud training configuration
    train_modal.py                       # [NEW] Modal wrapper script (standalone runnable)
    setup_cloud.sh                       # [NEW] Cloud provider setup script
    README.md                            # [NEW] Cloud training documentation
```

### 3.2 Component Descriptions

#### `tuner/backends/training/cloud/base_cloud.py`

Shared utilities for all cloud backends. Not an abstract class -- just helper functions:
- `load_cloud_config(trainer_dir)` -- Load `cloud_config.yaml` alongside regular `config.yaml`
- `resolve_dataset_path(config)` -- Convert relative dataset paths to absolute
- `format_job_status(status_dict)` -- Consistent status formatting for display
- `poll_job_until_done(check_fn, interval, timeout)` -- Generic polling loop with log streaming

#### `tuner/backends/training/cloud/hf_jobs_backend.py`

Implements `ITrainingBackend` for HuggingFace Jobs.

- **`validate_environment()`**: Checks `HF_TOKEN` is set, `huggingface_hub` version >= 0.34.0 (Jobs support), and user has Pro/Team access.
- **`load_config(method)`**: Loads standard training config + cloud overlay from `cloud_config.yaml`.
- **`execute(config, python_path)`**: Calls `huggingface_hub.run_job()` to submit training script as a UV script with inline deps. Polls `inspect_job()` and streams `fetch_job_logs()` until completion.
- **`get_available_methods()`**: Returns `['sft', 'kto']` (same as local, since scripts run unchanged).

#### `tuner/backends/training/cloud/modal_backend.py`

Implements `ITrainingBackend` for Modal.

- **`validate_environment()`**: Checks `modal` package is installed, token is configured (`MODAL_TOKEN_ID` + `MODAL_TOKEN_SECRET` or OAuth via `~/.modal.toml`).
- **`load_config(method)`**: Loads training config + cloud overlay.
- **`execute(config, python_path)`**: Runs `modal run Trainers/cloud/train_modal.py` as a subprocess with config flags. Modal handles the remote execution.
- **`get_available_methods()`**: Returns `['sft', 'kto']`.

#### `tuner/backends/training/cloud/runpod_backend.py`

Implements `ITrainingBackend` for RunPod.

- **`validate_environment()`**: Checks `RUNPOD_API_KEY` is set, `runpod` package is installed.
- **`load_config(method)`**: Loads training config + cloud overlay.
- **`execute(config, python_path)`**: Creates pod via `runpod.create_pod()`, waits for it to start, streams pod logs, waits for training completion marker, then terminates pod.
- **`get_available_methods()`**: Returns `['sft', 'kto']`.

#### `tuner/handlers/cloud_train_handler.py`

New handler for cloud training workflow. Follows the same pattern as `TrainHandler`:

1. Present provider selection menu (HF Jobs / Modal / RunPod)
2. Get provider backend from registry
3. Validate credentials/environment
4. Select training method (SFT / KTO)
5. Load and display config (including cloud-specific: GPU type, timeout, cost estimate)
6. Confirm with user
7. Submit job and stream logs

#### `Trainers/cloud/train_modal.py`

Standalone Modal wrapper script. Can be run directly (`modal run train_modal.py`) or via the CLI. Contains:
- Modal `App` definition with GPU/volume/image configuration
- `@app.function(gpu=...)` decorated training function that imports and runs existing training logic
- `@app.local_entrypoint()` for CLI invocation
- Persistent Volume definitions for model and checkpoint caching

#### `Trainers/cloud/cloud_config.yaml`

Default cloud training configuration overlay:

```yaml
# Cloud Training Configuration
# Merged with standard config.yaml when running cloud training

cloud:
  # Default provider (can be overridden per-run)
  default_provider: hf_jobs

  # HuggingFace Jobs settings
  hf_jobs:
    flavor: a10g-small           # GPU hardware flavor
    timeout: 4h                  # Job timeout
    image: pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

  # Modal settings
  modal:
    gpu: L40S                    # GPU type
    timeout_hours: 6             # Function timeout
    volumes:
      model_cache: model-cache   # Volume name for model caching
      checkpoints: checkpoints   # Volume name for checkpoints

  # RunPod settings
  runpod:
    gpu_type_id: NVIDIA RTX A6000
    gpu_count: 1
    volume_gb: 50
    image: runpod/pytorch:2.4.0-cuda12.4
    cloud_type: COMMUNITY        # COMMUNITY or SECURE

  # Common settings
  push_to_hub: true              # Push results to HF Hub on completion
  hub_repo: null                 # Target HF repo (prompted if null)
```

#### `Trainers/cloud/setup_cloud.sh`

Setup script for cloud provider dependencies:

```bash
#!/bin/bash
# Install cloud provider SDKs (optional, not required for local training)
pip install --upgrade huggingface_hub  # HF Jobs (v0.34.0+)
pip install modal                       # Modal
pip install runpod                      # RunPod
```

---

## 4. Interface Definitions

### 4.1 ITrainingBackend (Existing -- No Change)

The existing `ITrainingBackend` interface at `tuner/core/interfaces.py` is sufficient. Cloud backends implement the same four methods:

```python
class ITrainingBackend(ABC):
    @property
    def name(self) -> str: ...
    def get_available_methods(self) -> List[str]: ...
    def load_config(self, method: str) -> TrainingConfig: ...
    def execute(self, config: TrainingConfig, python_path: str) -> int: ...
    def validate_environment(self) -> Tuple[bool, str]: ...
```

**Design rationale**: Adding a separate `ICloudProvider` interface was considered but rejected. The existing interface already covers what cloud backends need: environment validation, config loading, and execution. The `execute()` method returning an exit code works equally well for "submit job and poll until done" as it does for "run subprocess locally." Keeping one interface means the registry, handler, and error handling patterns all stay the same.

### 4.2 CloudTrainingConfig (New Dataclass)

Extends `TrainingConfig` with cloud-specific fields:

```python
@dataclass
class CloudTrainingConfig(TrainingConfig):
    """Configuration for cloud training runs."""
    provider: str                    # 'hf_jobs', 'modal', 'runpod'
    gpu_type: str                    # Provider-specific GPU identifier
    timeout_hours: float             # Maximum job duration
    cloud_image: str                 # Docker image for the job
    push_to_hub: bool = True         # Push results to HF Hub
    hub_repo: Optional[str] = None   # Target HF repo ID

    # Provider-specific (optional)
    hf_flavor: Optional[str] = None          # HF Jobs hardware flavor
    modal_volumes: Optional[dict] = None     # Modal volume mappings
    runpod_volume_gb: Optional[int] = None   # RunPod persistent volume size
```

### 4.3 CloudProviderError (New Exception)

```python
class CloudProviderError(BackendError):
    """Exception raised when a cloud provider operation fails.

    Covers: authentication failures, job submission errors,
    timeout exceeded, provider API errors.
    """
    pass
```

Inherits from `BackendError`, so existing error handling in handlers catches it without changes.

---

## 5. Data Flow

### 5.1 Job Submission Flow (All Providers)

```
User selects "Cloud Training"
    |
    v
CloudTrainHandler.handle()
    |
    +-- Select provider (HF Jobs / Modal / RunPod)
    |
    +-- TrainingBackendRegistry.get("hf_jobs" | "modal" | "runpod")
    |
    +-- backend.validate_environment()
    |     |
    |     +-- Check env vars (HF_TOKEN, MODAL_TOKEN_*, RUNPOD_API_KEY)
    |     +-- Check SDK installed and version
    |     +-- Check account access (HF Pro, Modal auth, RunPod credits)
    |
    +-- Select method (SFT / KTO)
    |
    +-- backend.load_config(method)
    |     |
    |     +-- Load Trainers/rtx3090_{method}/configs/config.yaml (training params)
    |     +-- Load Trainers/cloud/cloud_config.yaml (cloud overlay)
    |     +-- Merge into CloudTrainingConfig
    |
    +-- Display config + cost estimate
    |
    +-- User confirms
    |
    +-- backend.execute(config, python_path)
          |
          +-- [Provider-specific job submission]
          +-- Poll status / stream logs
          +-- Return exit code (0 = success)
```

### 5.2 Provider-Specific Execution Patterns

#### HuggingFace Jobs

```
execute()
    |
    +-- Build UV script with PEP 723 inline deps header
    |     deps: unsloth, trl, transformers, datasets, peft
    |
    +-- huggingface_hub.run_job(
    |       image=config.cloud_image,
    |       command=["python", "train_{method}.py"],
    |       flavor=config.hf_flavor,
    |       secrets={"HF_TOKEN": os.environ["HF_TOKEN"]},
    |       timeout=config.timeout_hours,
    |   )
    |
    +-- Poll inspect_job(job_id) every 30s
    |     |
    |     +-- Stream fetch_job_logs(job_id) to console
    |
    +-- On COMPLETED: return 0
    +-- On ERROR: print error details, return 1
```

**Code sync strategy**: HF Jobs can run UV scripts with inline dependencies. The training script can be uploaded as-is, or the repo can be cloned inside the job via the command. For simplicity, use the command-based approach: `git clone {repo_url} && cd Trainers/rtx3090_{method} && python train_{method}.py`. This avoids file upload complexity.

**Output retrieval**: Training scripts already push to HF Hub when `HF_TOKEN` is available. The cloud job inherits this behavior. No additional output retrieval needed.

#### Modal

```
execute()
    |
    +-- Build modal run command:
    |     modal run Trainers/cloud/train_modal.py \
    |       --method {method} \
    |       --model-name {model_name} \
    |       --dataset {dataset} \
    |       --gpu {gpu_type}
    |
    +-- subprocess.run(cmd) -- Modal CLI handles remote execution
    |     |
    |     +-- train_modal.py @app.function runs remotely
    |     +-- Logs stream to local console via Modal CLI
    |
    +-- Return subprocess exit code
```

**Code sync strategy**: Modal's decorator pattern means the `train_modal.py` script runs locally but the decorated function executes remotely. Modal serializes the function and its dependencies. The wrapper imports training logic from the existing codebase.

**Output retrieval**: Modal Volumes persist outputs. The wrapper script downloads results from the Volume after training completes, or pushes directly to HF Hub from within the remote function.

#### RunPod

```
execute()
    |
    +-- runpod.create_pod(
    |       name="toolset-{method}-{timestamp}",
    |       image_name=config.cloud_image,
    |       gpu_type_id=config.gpu_type,
    |       volume_in_gb=config.runpod_volume_gb,
    |       docker_args=startup_command,
    |   )
    |
    +-- Wait for pod status == "RUNNING"
    |
    +-- Poll pod logs via RunPod API
    |     |
    |     +-- Look for "TRAINING_COMPLETE" marker in output
    |
    +-- On completion:
    |     +-- If push_to_hub: training script pushes to HF Hub from pod
    |     +-- runpod.terminate_pod(pod_id)
    |
    +-- Return 0 on success, 1 on error
```

**Code sync strategy**: RunPod pods have internet access. The startup command:
1. Installs dependencies (`pip install unsloth trl datasets peft`)
2. Clones the repo (`git clone {repo_url} /workspace/repo`)
3. Runs the training script (`cd /workspace/repo/Trainers/rtx3090_{method} && python train_{method}.py`)

The clone URL can be the user's GitHub repo URL (if public) or use a GitHub token (if private). For private repos, `GH_TOKEN` or a deploy key can be passed as an environment variable.

**Output retrieval**: Training scripts push models to HF Hub using `HF_TOKEN`, which is passed as a pod environment variable. This is the same mechanism used locally. No additional sync needed.

---

## 6. Environment Variables Schema

### 6.1 New Variables

| Variable | Provider | Required | Format | Description |
|----------|----------|----------|--------|-------------|
| `RUNPOD_API_KEY` | RunPod | Yes (for RunPod) | `rp_...` | RunPod API key from [runpod.io/console/user/settings](https://www.runpod.io/console/user/settings) |
| `MODAL_TOKEN_ID` | Modal | Optional* | `ak-...` | Modal token ID |
| `MODAL_TOKEN_SECRET` | Modal | Optional* | `as-...` | Modal token secret |
| `CLOUD_REPO_URL` | All | Optional | URL | Git repo URL for code sync (defaults to origin remote) |

*Modal can authenticate via browser OAuth (`modal setup`) which stores tokens in `~/.modal.toml`. Environment variables are an alternative for CI/CD environments.

### 6.2 Existing Variables (Used by Cloud)

| Variable | Already Exists | Used By |
|----------|----------------|---------|
| `HF_TOKEN` | Yes | HF Jobs (auth + job secrets), all providers (Hub upload) |
| `WANDB_API_KEY` | Yes | Optional metrics logging from cloud jobs |

### 6.3 Updated .env Template

```bash
# === Existing ===
HF_TOKEN=hf_your_token_here
OPENROUTER_API_KEY=sk-or-...
WANDB_API_KEY=your_wandb_key

# === Cloud Training (Optional) ===
# RunPod - Get key from https://www.runpod.io/console/user/settings
RUNPOD_API_KEY=rp_your_key_here

# Modal - Alternative to browser OAuth (modal setup)
# Get tokens from https://modal.com/settings
MODAL_TOKEN_ID=ak-your_id_here
MODAL_TOKEN_SECRET=as-your_secret_here

# Repo URL for cloud code sync (auto-detected from git remote if not set)
# CLOUD_REPO_URL=https://github.com/username/Toolset-Training.git
```

---

## 7. CLI Integration

### 7.1 Command Structure

Add `cloud` as a new top-level command alongside `train`, `eval`, `synthchat`, and `modelops`:

```
python tuner.py                    # Interactive menu (includes Cloud Training option)
python tuner.py cloud              # Cloud training submenu
python tuner.py cloud --json       # JSON mode (returns provider status)
```

### 7.2 Parser Changes (`tuner/cli/parser.py`)

Add `cloud` to the `choices` list:

```python
parser.add_argument(
    "command",
    nargs="?",
    choices=["train", "cloud", "eval", "synthchat", "modelops", "status", "doctor", "list"],
    help="Command to run (optional, defaults to interactive menu)"
)
```

### 7.3 Router Changes (`tuner/cli/router.py`)

Add cloud handler import and routing:

```python
from tuner.handlers.cloud_train_handler import CloudTrainHandler

handlers = {
    'train': TrainHandler,
    'cloud': CloudTrainHandler,
    'eval': EvalHandler,
    'synthchat': SynthChatHandler,
    'modelops': ModelOpsHandler,
}
```

### 7.4 Main Menu Changes (`tuner/handlers/main_menu_handler.py`)

Add cloud training option to the menu:

```python
menu_options = [
    ("train", f"{BOX['star']} Training - Train models locally (SFT, KTO, GRPO)"),
    ("cloud", f"{BOX['bullet']} Cloud Training - Train on GPU cloud (HF Jobs, Modal, RunPod)"),
    ("eval", f"{BOX['bullet']} Evaluation - Run benchmarks against a model"),
    ("synthchat", f"{BOX['bullet']} SynthChat - Generate + improve training data"),
    ("modelops", f"{BOX['bullet']} Model Ops - Run, merge, convert, upload"),
]
```

### 7.5 Cloud Training Handler Flow

```
CloudTrainHandler.handle()
    |
    v
+-------------------------------------------+
|  CLOUD TRAINING                           |
|  Select your cloud GPU provider           |
|                                           |
|  [1] * HuggingFace Jobs (HF_TOKEN ready)  |
|  [2] . Modal (requires setup)             |
|  [3] . RunPod (requires API key)          |
|  [0]   Back                               |
+-------------------------------------------+
    |
    v (user selects HF Jobs)
+-------------------------------------------+
|  HUGGINGFACE JOBS                         |
|  Select training method                   |
|                                           |
|  [1] . SFT - Supervised Fine-Tuning       |
|  [2] . KTO - Preference Learning          |
|  [0]   Back                               |
+-------------------------------------------+
    |
    v (user selects SFT)
+-------------------------------------------+
|  Cloud Training Configuration             |
|                                           |
|  Provider:     HuggingFace Jobs           |
|  Method:       SFT                        |
|  Model:        LFM2.5-1.2B-Instruct      |
|  Dataset:      nonthinking_tools_sft...   |
|  GPU:          A10G (24GB)                |
|  Timeout:      4 hours                    |
|  Est. Cost:    ~$4.00                     |
|                                           |
|  Start cloud training? (y/n)              |
+-------------------------------------------+
    |
    v (user confirms)
    [Submit job, stream logs, report result]
```

### 7.6 Registry Changes (`tuner/backends/registry.py`)

Register cloud backends conditionally (only when SDKs are available):

```python
# Cloud backends (optional - registered only if SDK is available)
try:
    from tuner.backends.training.cloud import HFJobsBackend
    TrainingBackendRegistry.register("hf_jobs", HFJobsBackend)
except ImportError:
    pass  # huggingface_hub not installed or too old

try:
    from tuner.backends.training.cloud import ModalBackend
    TrainingBackendRegistry.register("modal", ModalBackend)
except ImportError:
    pass  # modal not installed

try:
    from tuner.backends.training.cloud import RunPodBackend
    TrainingBackendRegistry.register("runpod", RunPodBackend)
except ImportError:
    pass  # runpod not installed
```

This ensures that the CLI works even when cloud SDKs are not installed -- the providers simply don't appear in the menu.

---

## 8. Technology Decisions

### ADR-001: Reuse ITrainingBackend Instead of New Interface

**Decision**: Cloud providers implement the existing `ITrainingBackend` interface.

**Alternatives considered**:
- **New `ICloudProvider` interface**: Would provide cloud-specific methods (`submit_job`, `get_status`, `fetch_results`, `cancel_job`). Rejected because it introduces an unnecessary abstraction layer. The four existing methods (`validate_environment`, `load_config`, `execute`, `get_available_methods`) cover the cloud use case. `execute()` already returns an exit code which maps to "submit, poll, return status."
- **Strategy pattern with `CloudTrainingStrategy`**: Would separate job lifecycle management from the backend. Rejected as over-engineering for three providers with relatively simple APIs.

**Rationale**: The existing interface is thin and sufficient. Cloud backends just implement `execute()` differently (HTTP API calls instead of local subprocess). The handler, registry, and error handling all work unchanged.

### ADR-002: Cloud Backends in `tuner/backends/training/cloud/` Not Top-Level `cloud/`

**Decision**: Place cloud backends under `tuner/backends/training/cloud/`.

**Alternatives considered**:
- **Top-level `cloud/` directory**: Conceptually separate from tuner. Rejected because it would need its own import paths and wouldn't integrate with the existing registry pattern.
- **`Trainers/cloud/` only**: Mixed training scripts and backend code. Rejected because backend logic belongs in `tuner/backends/`.

**Rationale**: Cloud providers are training backends. They belong in the training backends module. `Trainers/cloud/` holds only the Modal wrapper script and cloud config (deployment assets, not backend logic).

### ADR-003: Conditional Registration via Try/Except

**Decision**: Cloud backends are registered only if their SDK is importable.

**Alternatives considered**:
- **Always register, fail at runtime**: Register all backends, raise `CloudProviderError` if SDK missing. Rejected because it shows unavailable options in the menu.
- **Feature flags in config**: Add `cloud_providers_enabled` to config. Rejected as unnecessary indirection.

**Rationale**: Try/except at import time is simple, Pythonic, and means the menu only shows providers the user can actually use. Zero configuration needed.

### ADR-004: Code Sync via Git Clone (HF Jobs, RunPod)

**Decision**: Cloud jobs clone the repo from Git rather than uploading files.

**Alternatives considered**:
- **File upload**: Upload training scripts via provider API. Rejected because it requires managing file dependencies and the full repo context.
- **Docker image with code baked in**: Build custom image per run. Rejected as too slow for iterative development.
- **rsync to RunPod pod**: Direct file copy. Considered as future enhancement for private repos without GitHub.

**Rationale**: Git clone is simple, reproducible, and works with the existing repo structure. Private repos can use `GH_TOKEN` or deploy keys. The repo is small enough that clone time is negligible.

### ADR-005: Separate Handler for Cloud Training

**Decision**: Create `CloudTrainHandler` rather than extending `TrainHandler`.

**Alternatives considered**:
- **Extend `TrainHandler` with cloud option**: Add cloud as a third platform alongside RTX and Mac. Rejected because the workflow is different enough (provider selection, cost estimation, job polling) that it would complicate the existing handler.
- **Subcommand under `train`**: `python tuner.py train --cloud`. Rejected because it conflates local and cloud workflows in one handler.

**Rationale**: Cloud training has a distinct workflow (provider selection, credential checks, cost estimates, async job management) that warrants its own handler. This keeps `TrainHandler` focused and cloud-specific logic contained.

---

## 9. Security Architecture

### 9.1 Credential Management

| Credential | Storage | Exposure Risk | Mitigation |
|------------|---------|---------------|------------|
| `HF_TOKEN` | `.env` file (gitignored) | Could be committed to git | `.gitignore` already includes `.env` |
| `RUNPOD_API_KEY` | `.env` file (gitignored) | Same as above | Same mitigation |
| `MODAL_TOKEN_ID/SECRET` | `.env` or `~/.modal.toml` | Same as above | Modal OAuth preferred (no env vars) |
| `GH_TOKEN` (optional) | `.env` file (gitignored) | Needed for private repo clone | Use read-only deploy keys if possible |

### 9.2 Credential Passing to Cloud Jobs

- **HF Jobs**: Uses `secrets` parameter in `run_job()` -- credentials are encrypted at rest and only exposed inside the job container.
- **Modal**: Uses `modal.Secret` objects -- credentials stored in Modal's secret manager, not in code.
- **RunPod**: Passed as pod environment variables via `env` parameter in `create_pod()`. Encrypted in transit (HTTPS API), but visible inside the pod.

### 9.3 Security Principles

1. **Never log credentials**: All `execute()` methods must ensure API keys are not printed in logs or error messages.
2. **Validate credentials before submission**: `validate_environment()` checks token format (prefix matching) without revealing the full token.
3. **No credentials in config files**: `cloud_config.yaml` references env var names, not values.
4. **Timeout enforcement**: All providers must set explicit timeouts to prevent runaway costs.

---

## 10. Deployment Architecture

### 10.1 Dependency Management

Cloud provider SDKs are optional dependencies. Two approaches:

**Option A: requirements-cloud.txt** (Recommended)

```
# requirements-cloud.txt
# Optional: Install for cloud training support
huggingface_hub>=0.34.0    # HF Jobs
modal                       # Modal
runpod                      # RunPod
```

Install with: `pip install -r requirements-cloud.txt`

**Option B: Extras in setup.py/pyproject.toml**

```toml
[project.optional-dependencies]
cloud = ["huggingface_hub>=0.34.0", "modal", "runpod"]
```

Install with: `pip install -e ".[cloud]"`

Recommendation: Use `requirements-cloud.txt` for simplicity, matching the existing `requirements.txt` pattern. The `setup_cloud.sh` script wraps this.

### 10.2 Doctor Command Integration

The existing `doctor` command should check cloud provider status:

```
Cloud Training Providers:
  [OK] HuggingFace Jobs: HF_TOKEN set, huggingface_hub 0.35.1
  [--] Modal: Not installed (pip install modal)
  [!!] RunPod: RUNPOD_API_KEY not set
```

This is informational only -- cloud providers are optional.

---

## 11. Implementation Guidelines

### 11.1 For the Cloud Base Utilities (`base_cloud.py`)

```python
# tuner/backends/training/cloud/base_cloud.py

import time
import yaml
from pathlib import Path
from typing import Callable, Optional


def load_cloud_config(cloud_config_path: Path) -> dict:
    """Load cloud_config.yaml and return the cloud section."""
    if not cloud_config_path.exists():
        return {}
    with open(cloud_config_path) as f:
        config = yaml.safe_load(f)
    return config.get('cloud', {})


def resolve_repo_url() -> str:
    """Get the repo URL for code sync, from env or git remote."""
    import os
    import subprocess
    url = os.environ.get('CLOUD_REPO_URL')
    if url:
        return url
    result = subprocess.run(
        ['git', 'remote', 'get-url', 'origin'],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        return result.stdout.strip()
    raise ValueError("No CLOUD_REPO_URL set and no git remote origin found")


def poll_until_done(
    check_fn: Callable[[], Optional[str]],
    interval: int = 30,
    timeout_seconds: int = 14400,
) -> str:
    """
    Poll check_fn until it returns a terminal status.

    check_fn should return:
      - None if still running (keep polling)
      - "COMPLETED" on success
      - "ERROR: <message>" on failure
    """
    elapsed = 0
    while elapsed < timeout_seconds:
        status = check_fn()
        if status is not None:
            return status
        time.sleep(interval)
        elapsed += interval
    return "ERROR: Timeout exceeded"
```

### 11.2 For the HF Jobs Backend

Key implementation detail: HF Jobs supports "UV scripts" which declare dependencies inline using PEP 723 metadata. This is the cleanest pattern:

```python
# The submitted script starts with:
# /// script
# dependencies = [
#   "unsloth",
#   "trl>=0.15",
#   "datasets",
#   "peft",
# ]
# ///
```

The backend should generate this UV script wrapper dynamically, embedding the training script content or referencing it via git clone.

### 11.3 For the Modal Backend

The `Trainers/cloud/train_modal.py` file must be a standalone script. The Modal backend's `execute()` simply runs it as a subprocess:

```python
def execute(self, config: CloudTrainingConfig, python_path: str) -> int:
    cmd = [
        "modal", "run",
        str(self.repo_root / "Trainers" / "cloud" / "train_modal.py"),
        "--method", config.method,
        "--model-name", config.model_name,
        "--gpu", config.gpu_type,
    ]
    process = subprocess.Popen(cmd)
    return process.wait()
```

### 11.4 For the RunPod Backend

The startup command must be carefully constructed:

```python
def _build_startup_command(self, config: CloudTrainingConfig) -> str:
    repo_url = resolve_repo_url()
    return (
        f"pip install unsloth trl datasets peft && "
        f"git clone {repo_url} /workspace/repo && "
        f"cd /workspace/repo/Trainers/rtx3090_{config.method} && "
        f"HF_TOKEN={os.environ.get('HF_TOKEN', '')} "
        f"python train_{config.method}.py"
    )
```

**Important**: `HF_TOKEN` must be passed as an environment variable to the pod, not embedded in the command string (which would appear in logs). Use RunPod's `env` parameter:

```python
pod = runpod.create_pod(
    ...,
    env={"HF_TOKEN": os.environ.get("HF_TOKEN", "")},
)
```

### 11.5 Error Handling Patterns

All cloud backends should handle these failure modes:

| Failure Mode | Detection | Response |
|-------------|-----------|----------|
| Invalid credentials | `validate_environment()` returns `(False, msg)` | Show error, suggest fix |
| Job submission fails | API exception during `execute()` | Catch, wrap in `CloudProviderError` |
| Job times out | Polling exceeds timeout | Terminate/cancel job, return 1 |
| Job crashes | Status = ERROR/FAILED | Print logs, return 1 |
| Network error during polling | Connection exception | Retry 3 times, then fail |
| Provider API rate limit | HTTP 429 | Exponential backoff |

---

## 12. Implementation Roadmap

### Phase 1: Foundation (All Providers Share)

| Step | Task | Files | Dependencies |
|------|------|-------|--------------|
| 1.1 | Add `CloudTrainingConfig` dataclass | `tuner/core/config.py` | None |
| 1.2 | Add `CloudProviderError` exception | `tuner/core/exceptions.py` | None |
| 1.3 | Create `tuner/backends/training/cloud/` package with `base_cloud.py` | New directory + files | 1.1 |
| 1.4 | Create `Trainers/cloud/cloud_config.yaml` | New file | None |
| 1.5 | Create `CloudTrainHandler` | `tuner/handlers/cloud_train_handler.py` | 1.1, 1.2 |
| 1.6 | Wire CLI: parser, router, main menu | `tuner/cli/parser.py`, `router.py`, `main_menu_handler.py` | 1.5 |

### Phase 2: HuggingFace Jobs (First Provider)

| Step | Task | Files | Dependencies |
|------|------|-------|--------------|
| 2.1 | Implement `HFJobsBackend` | `tuner/backends/training/cloud/hf_jobs_backend.py` | Phase 1 |
| 2.2 | Register in registry | `tuner/backends/registry.py` | 2.1 |
| 2.3 | Test: validate environment, submit test job | Manual testing | 2.2 |

### Phase 3: Modal (Second Provider)

| Step | Task | Files | Dependencies |
|------|------|-------|--------------|
| 3.1 | Create `train_modal.py` wrapper | `Trainers/cloud/train_modal.py` | None |
| 3.2 | Implement `ModalBackend` | `tuner/backends/training/cloud/modal_backend.py` | Phase 1, 3.1 |
| 3.3 | Register in registry | `tuner/backends/registry.py` | 3.2 |
| 3.4 | Test: validate environment, run Modal job | Manual testing | 3.3 |

### Phase 4: RunPod (Third Provider)

| Step | Task | Files | Dependencies |
|------|------|-------|--------------|
| 4.1 | Implement `RunPodBackend` | `tuner/backends/training/cloud/runpod_backend.py` | Phase 1 |
| 4.2 | Register in registry | `tuner/backends/registry.py` | 4.1 |
| 4.3 | Test: validate environment, create/terminate pod | Manual testing | 4.2 |

### Phase 5: Polish

| Step | Task | Files | Dependencies |
|------|------|-------|--------------|
| 5.1 | Create `setup_cloud.sh` | `Trainers/cloud/setup_cloud.sh` | None |
| 5.2 | Create `requirements-cloud.txt` | Root directory | None |
| 5.3 | Add cloud status to `doctor` command | `tuner/handlers/doctor_handler.py` | Phase 2-4 |
| 5.4 | Create `Trainers/cloud/README.md` | New file | Phase 2-4 |
| 5.5 | Update root `CLAUDE.md` with cloud training docs | `CLAUDE.md` | Phase 2-4 |

### Development Order

Phases 1-5 are sequential. Within Phase 1, steps 1.1-1.4 can be done in parallel. Phases 2, 3, and 4 can be developed in parallel after Phase 1 completes, as the three providers are independent.

### Milestones

| Milestone | Deliverable | Acceptance Criteria |
|-----------|-------------|---------------------|
| M1: Cloud CLI Ready | Menu shows "Cloud Training" option, handler skeleton works | `python tuner.py cloud` shows provider menu |
| M2: HF Jobs Works | Submit and monitor HF Jobs training | User can submit SFT job from CLI and see logs |
| M3: Modal Works | Submit and monitor Modal training | User can submit SFT job via Modal from CLI |
| M4: RunPod Works | Submit and monitor RunPod training | User can submit SFT job on RunPod from CLI |
| M5: Production Ready | Documentation, doctor integration, error handling | All providers documented and tested |

---

## 13. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Training scripts need modifications for cloud** | Low | High | HF Jobs runs scripts as-is; Modal wrapper imports existing logic; RunPod clones and runs. Validated by research. |
| **LFM2.5 bnb-4bit crash in cloud** | Certain | High | Cloud config must inherit `load_in_4bit: false` from existing config.yaml. Document in README. |
| **RunPod pod lifecycle management complexity** | Medium | Medium | Implement robust error handling with auto-terminate on failure. Use `finally` blocks. |
| **Modal API breaking changes** | Low | Medium | Pin `modal` SDK version in `requirements-cloud.txt`. |
| **Cost overrun from forgotten jobs** | Medium | Medium | Enforce mandatory timeout in all providers. Add cost estimate to confirmation prompt. |
| **Private repo code sync fails** | Medium | Low | Fallback instructions for GH_TOKEN. Detect and warn in `validate_environment()`. |
| **Provider SDK import breaks local-only users** | Low | High | Conditional registration (try/except import) ensures local-only users are unaffected. |

---

## 14. Testing Strategy

### Unit Tests

- Test `load_cloud_config()` with valid/invalid/missing YAML
- Test `CloudTrainingConfig` construction and field validation
- Test each backend's `validate_environment()` with mocked env vars
- Test `CloudTrainHandler` menu flow with mocked backends

### Integration Tests

- Test HF Jobs: submit a minimal 1-step training job, verify completion
- Test Modal: run a minimal function, verify output
- Test RunPod: create pod, verify it starts, terminate

### Manual Smoke Tests

- Full CLI flow: `python tuner.py cloud` through each provider
- Verify log streaming works for each provider
- Verify model push to HF Hub works from cloud jobs
- Test error paths: invalid credentials, job timeout, provider unavailable
