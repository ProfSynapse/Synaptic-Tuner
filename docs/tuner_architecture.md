# Synaptic Tuner CLI Architecture

**Version:** 1.0
**Date:** 2025-11-30
**Status:** Proposed Architecture

---

## Executive Summary

This document specifies a modular, maintainable architecture for the Synaptic Tuner CLI (`tuner.py`), transforming it from a 1026-line monolithic script into a well-structured, testable application following SOLID principles. The architecture aligns with existing patterns in `Trainers/shared/upload/`, ensuring consistency across the codebase.

**Current Pain Points:**
- 1026 lines in a single file
- Duplicated UI fallback code (~70 lines)
- Tightly coupled menu logic and business operations
- Difficult to test individual components
- Hard to extend with new training methods or backends
- Copy-pasted configuration parsing across different menus

**Proposed Solution:**
- Modular architecture with clear separation of concerns
- Interface-based design for extensibility
- Shared utilities eliminate duplication
- Business logic separated from UI presentation
- Easy to add new training methods, backends, or evaluation tools

---

## Table of Contents

1. [System Context](#1-system-context)
2. [Component Architecture](#2-component-architecture)
3. [Data Architecture](#3-data-architecture)
4. [API Specifications](#4-api-specifications)
5. [Technology Decisions](#5-technology-decisions)
6. [Implementation Guidelines](#6-implementation-guidelines)
7. [Migration Strategy](#7-migration-strategy)
8. [Risk Assessment](#8-risk-assessment)

---

## 1. System Context

### 1.1 External Dependencies

```
┌─────────────────────────────────────────────────────────────┐
│                    Synaptic Tuner CLI                       │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   Training   │  │    Upload    │  │  Evaluation  │    │
│  │   Backends   │  │   Backend    │  │   Backends   │    │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘    │
│         │                  │                  │            │
└─────────┼──────────────────┼──────────────────┼────────────┘
          │                  │                  │
          ▼                  ▼                  ▼
┌─────────────────┐  ┌──────────────┐  ┌──────────────┐
│ RTX SFT/KTO     │  │ HuggingFace  │  │   Ollama     │
│ Mac MLX LoRA    │  │     Hub      │  │  LM Studio   │
└─────────────────┘  └──────────────┘  └──────────────┘

External Systems:
- Conda environments (unsloth_latest)
- YAML config files (training parameters)
- JSONL training logs (checkpoint metrics)
- HuggingFace API (model upload)
- Ollama/LM Studio HTTP APIs (inference)
- Filesystem (training runs, checkpoints, datasets)
```

### 1.2 System Boundaries

**In Scope:**
- CLI argument parsing and menu orchestration
- Training workflow orchestration (RTX SFT/KTO, Mac MLX)
- Upload workflow orchestration (delegates to `Trainers/shared/upload/`)
- Evaluation workflow orchestration
- Configuration loading and validation
- Checkpoint discovery and selection
- Model/prompt set discovery
- Cross-platform environment detection

**Out of Scope:**
- Actual training logic (handled by `train_sft.py`, `train_kto.py`, `main.py`)
- Model upload strategies (handled by `Trainers/shared/upload/`)
- Evaluation execution (handled by `Evaluator/`)
- Dataset generation (separate system)

---

## 2. Component Architecture

### 2.1 High-Level Architecture

```
tuner/
├── __init__.py                    # Package initialization
├── __main__.py                    # Entry point (python -m tuner)
├── cli/                           # CLI layer
│   ├── __init__.py
│   ├── main.py                    # Main CLI entry point
│   ├── parser.py                  # Argument parser
│   └── router.py                  # Command router
├── core/                          # Core business logic
│   ├── __init__.py
│   ├── interfaces.py              # Abstract interfaces
│   ├── config.py                  # Configuration models
│   └── exceptions.py              # Custom exceptions
├── handlers/                      # Menu/command handlers
│   ├── __init__.py
│   ├── base.py                    # Base handler class
│   ├── train_handler.py           # Training menu
│   ├── upload_handler.py          # Upload menu
│   ├── eval_handler.py            # Evaluation menu
│   └── pipeline_handler.py        # Pipeline orchestration
├── backends/                      # Backend discovery/interaction
│   ├── __init__.py
│   ├── registry.py                # Backend registry
│   ├── training/
│   │   ├── __init__.py
│   │   ├── base.py                # ITrainingBackend
│   │   ├── rtx_backend.py         # RTX SFT/KTO backend
│   │   └── mac_backend.py         # Mac MLX backend
│   └── evaluation/
│       ├── __init__.py
│       ├── base.py                # IEvaluationBackend
│       ├── ollama_backend.py      # Ollama backend
│       └── lmstudio_backend.py    # LM Studio backend
├── discovery/                     # Resource discovery
│   ├── __init__.py
│   ├── training_runs.py           # Find training runs
│   ├── checkpoints.py             # Find checkpoints
│   ├── models.py                  # Find models (Ollama/LMS)
│   └── prompt_sets.py             # Find prompt sets
├── ui/                            # UI components (delegates to shared)
│   ├── __init__.py
│   ├── menu.py                    # Menu rendering
│   ├── table.py                   # Table rendering
│   └── prompts.py                 # User input prompts
└── utils/                         # Utilities
    ├── __init__.py
    ├── environment.py             # Environment detection
    ├── conda.py                   # Conda path finding
    └── validation.py              # Input validation

Integration Points:
- Trainers/shared/ui/              # UI theme and console
- Trainers/shared/utilities/       # Path and env utilities
- Trainers/shared/upload/          # Upload orchestration
- Evaluator/                       # Evaluation execution
```

### 2.2 Component Interaction

```
┌─────────────────────────────────────────────────────────────┐
│                      CLI Layer                              │
│  ┌──────────┐     ┌──────────┐     ┌──────────────┐       │
│  │  Parser  │────►│  Router  │────►│   Handler    │       │
│  └──────────┘     └──────────┘     │  (Train/     │       │
│                                     │   Upload/    │       │
│                                     │   Eval)      │       │
│                                     └──────┬───────┘       │
└────────────────────────────────────────────┼───────────────┘
                                             │
                                             ▼
┌─────────────────────────────────────────────────────────────┐
│                   Business Logic Layer                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Backend    │  │  Discovery   │  │    Config    │     │
│  │   Registry   │  │   Services   │  │   Loaders    │     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
└─────────┼──────────────────┼──────────────────┼─────────────┘
          │                  │                  │
          ▼                  ▼                  ▼
┌─────────────────┐  ┌──────────────┐  ┌──────────────┐
│ Training/Eval   │  │ Filesystem   │  │ YAML/JSON    │
│   Backends      │  │   Scanner    │  │   Parsers    │
└─────────────────┘  └──────────────┘  └──────────────┘

Data Flow:
1. User invokes command → Parser → Router
2. Router selects Handler based on command
3. Handler uses Discovery to find resources
4. Handler uses Backend to execute operations
5. Backend delegates to external systems
6. Results flow back through Handler → UI
```

### 2.3 Detailed Component Descriptions

#### 2.3.1 CLI Layer

**Purpose:** Handle command-line interaction and route to appropriate handlers.

**Components:**
- `parser.py`: Parse command-line arguments
- `router.py`: Route commands to handlers
- `main.py`: Entry point, orchestrate CLI flow

**Responsibilities:**
- Validate CLI arguments
- Show help/usage information
- Route to appropriate handler
- Handle top-level errors

#### 2.3.2 Core Layer

**Purpose:** Define interfaces, configuration models, and exceptions.

**Components:**
- `interfaces.py`: Abstract base classes (protocols)
- `config.py`: Configuration dataclasses
- `exceptions.py`: Custom exceptions

**Responsibilities:**
- Define contracts for all components
- Type-safe configuration models
- Centralized exception hierarchy

#### 2.3.3 Handlers Layer

**Purpose:** Implement menu logic and workflow orchestration.

**Components:**
- `base.py`: Base handler with common functionality
- `train_handler.py`: Training menu and workflow
- `upload_handler.py`: Upload menu and workflow
- `eval_handler.py`: Evaluation menu and workflow
- `pipeline_handler.py`: Full pipeline orchestration

**Responsibilities:**
- Present menus to user
- Gather user input
- Validate selections
- Orchestrate backend calls
- Display results

#### 2.3.4 Backends Layer

**Purpose:** Abstract interaction with external training/evaluation systems.

**Components:**
- `registry.py`: Backend registration and retrieval
- `training/`: Training backend implementations
- `evaluation/`: Evaluation backend implementations

**Responsibilities:**
- Abstract platform differences
- Execute training/evaluation
- Report progress
- Handle errors

#### 2.3.5 Discovery Layer

**Purpose:** Find and enumerate available resources.

**Components:**
- `training_runs.py`: Find training run directories
- `checkpoints.py`: Find and analyze checkpoints
- `models.py`: List available models from backends
- `prompt_sets.py`: Find and parse prompt sets

**Responsibilities:**
- Scan filesystem for resources
- Parse metadata (logs, configs)
- Return structured resource lists

#### 2.3.6 UI Layer

**Purpose:** Render menus, tables, and prompts (thin wrapper over `Trainers/shared/ui/`).

**Components:**
- `menu.py`: Menu rendering
- `table.py`: Table rendering
- `prompts.py`: User input prompts

**Responsibilities:**
- Delegate to `Trainers/shared/ui/`
- Add CLI-specific UI helpers
- Maintain graceful fallbacks

#### 2.3.7 Utils Layer

**Purpose:** Cross-cutting utilities.

**Components:**
- `environment.py`: Environment detection (WSL/Linux/Windows)
- `conda.py`: Conda environment discovery
- `validation.py`: Input validation helpers

**Responsibilities:**
- Platform detection
- Path resolution
- Input sanitization

---

## 3. Data Architecture

### 3.1 Configuration Models

```python
# tuner/core/config.py

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

@dataclass
class TrainingConfig:
    """Configuration for a training run."""
    method: str                    # "sft", "kto", "mlx"
    platform: str                  # "rtx", "mac"
    config_path: Path              # Path to YAML config
    trainer_dir: Path              # Trainer directory
    model_name: str                # From config
    dataset_file: str              # From config
    epochs: int                    # From config
    batch_size: int                # From config
    learning_rate: float           # From config

    @classmethod
    def from_yaml(cls, yaml_path: Path, method: str, platform: str):
        """Load from YAML config file."""
        # Implementation parses YAML and extracts fields
        pass

@dataclass
class CheckpointInfo:
    """Information about a training checkpoint."""
    path: Path
    step: int
    metrics: Dict[str, float]      # loss, kl, margin, lr, etc.
    is_final: bool                 # True if final_model

    def score(self, training_type: str) -> float:
        """Calculate quality score based on training type."""
        if training_type == "kto":
            kl = self.metrics.get('kl', 0)
            margin = self.metrics.get('rewards/margins', 0)
            return margin / kl if kl > 0 else 0
        else:
            return -self.metrics.get('loss', float('inf'))

@dataclass
class UploadConfig:
    """Configuration for model upload."""
    model_path: Path
    repo_id: str
    save_method: str               # "merged_16bit", "merged_4bit", "lora"
    create_gguf: bool
    hf_token: str

@dataclass
class EvalConfig:
    """Configuration for evaluation."""
    backend: str                   # "ollama", "lmstudio"
    model: str
    prompt_set: Path
    prompt_count: int
    temperature: float = 0.2
    top_p: float = 0.9
    max_tokens: int = 1024
```

### 3.2 Data Flow Diagrams

#### Training Workflow

```
User Input
    │
    ▼
┌───────────────┐
│ Select        │
│ Platform      │  rtx / mac
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ Select        │
│ Method        │  sft / kto / mlx
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ Load YAML     │
│ Config        │  Parse training parameters
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ Display       │
│ Config        │  Show to user for confirmation
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ Execute       │
│ Training      │  subprocess.run([conda_python, "train_*.py"])
└───────┬───────┘
        │
        ▼
    Results
```

#### Upload Workflow

```
User Input
    │
    ▼
┌───────────────┐
│ Select        │
│ Model Type    │  sft / kto
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ List Training │
│ Runs          │  Scan sft_output_rtx3090/ or kto_output_rtx3090/
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ Select Run    │  User picks from timestamped directories
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ Load          │
│ Checkpoint    │  Parse logs/training_*.jsonl
│ Metrics       │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ Display       │
│ Checkpoint    │  Show metrics table with scores
│ Table         │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ Select        │
│ Checkpoint    │  User picks checkpoint or final_model
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ Configure     │
│ Upload        │  Repo ID, save method, GGUF option
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ Delegate to   │
│ Upload CLI    │  Call Trainers/shared/upload/cli/upload_cli.py
└───────┬───────┘
        │
        ▼
    Results
```

#### Evaluation Workflow

```
User Input
    │
    ▼
┌───────────────┐
│ Select        │
│ Backend       │  ollama / lmstudio
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ List Models   │  Query backend API
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ Select Model  │  User picks from available models
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ List Prompt   │
│ Sets          │  Scan Evaluator/prompts/*.json
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ Select Prompt │
│ Set           │  User picks prompt set
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ Execute       │
│ Evaluation    │  Call python -m Evaluator.cli
└───────┬───────┘
        │
        ▼
    Results
```

---

## 4. API Specifications

### 4.1 Core Interfaces

```python
# tuner/core/interfaces.py

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Dict, Any
from .config import TrainingConfig, CheckpointInfo

class ITrainingBackend(ABC):
    """
    Interface for training backend implementations.

    Implementations:
    - RTXBackend: NVIDIA GPU training (SFT/KTO via Unsloth)
    - MacBackend: Apple Silicon training (MLX LoRA)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend identifier (e.g., 'rtx', 'mac')."""
        pass

    @abstractmethod
    def get_available_methods(self) -> List[str]:
        """
        Get available training methods for this backend.

        Returns:
            List of method names (e.g., ['sft', 'kto'] or ['mlx'])
        """
        pass

    @abstractmethod
    def load_config(self, method: str) -> TrainingConfig:
        """
        Load configuration for a training method.

        Args:
            method: Training method ('sft', 'kto', 'mlx')

        Returns:
            Parsed training configuration
        """
        pass

    @abstractmethod
    def execute(self, config: TrainingConfig, python_path: str) -> int:
        """
        Execute training.

        Args:
            config: Training configuration
            python_path: Path to Python interpreter (conda env)

        Returns:
            Exit code (0 = success)
        """
        pass

    @abstractmethod
    def validate_environment(self) -> tuple[bool, str]:
        """
        Validate that backend environment is available.

        Returns:
            Tuple of (is_valid, error_message)
        """
        pass


class IEvaluationBackend(ABC):
    """
    Interface for evaluation backend implementations.

    Implementations:
    - OllamaBackend: Ollama local inference
    - LMStudioBackend: LM Studio local inference
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend identifier (e.g., 'ollama', 'lmstudio')."""
        pass

    @abstractmethod
    def list_models(self) -> List[str]:
        """
        List available models from this backend.

        Returns:
            List of model names
        """
        pass

    @abstractmethod
    def validate_connection(self) -> tuple[bool, str]:
        """
        Validate backend is running and accessible.

        Returns:
            Tuple of (is_connected, error_message)
        """
        pass

    @property
    @abstractmethod
    def default_host(self) -> str:
        """Default host for this backend."""
        pass

    @property
    @abstractmethod
    def default_port(self) -> int:
        """Default port for this backend."""
        pass


class IHandler(ABC):
    """
    Interface for command handlers.

    Implementations:
    - TrainHandler: Training workflow
    - UploadHandler: Upload workflow
    - EvalHandler: Evaluation workflow
    - PipelineHandler: Full pipeline
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Handler identifier."""
        pass

    @abstractmethod
    def handle(self) -> int:
        """
        Execute handler workflow.

        Returns:
            Exit code (0 = success)
        """
        pass

    @abstractmethod
    def can_handle_direct_mode(self) -> bool:
        """
        Whether this handler supports direct CLI invocation.

        Returns:
            True if can be invoked as `tuner.py <command>`
        """
        pass


class IDiscoveryService(ABC):
    """
    Interface for resource discovery services.

    Implementations:
    - TrainingRunDiscovery: Find training runs
    - CheckpointDiscovery: Find and analyze checkpoints
    - ModelDiscovery: List models from backends
    - PromptSetDiscovery: Find prompt sets
    """

    @abstractmethod
    def discover(self, **filters) -> List[Any]:
        """
        Discover resources matching filters.

        Args:
            **filters: Discovery-specific filters

        Returns:
            List of discovered resources
        """
        pass
```

### 4.2 Backend Implementations

#### RTX Training Backend

```python
# tuner/backends/training/rtx_backend.py

import yaml
import subprocess
from pathlib import Path
from typing import List
from .base import ITrainingBackend
from tuner.core.config import TrainingConfig

class RTXBackend(ITrainingBackend):
    """NVIDIA RTX training backend (SFT/KTO via Unsloth)."""

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root

    @property
    def name(self) -> str:
        return "rtx"

    def get_available_methods(self) -> List[str]:
        return ["sft", "kto"]

    def load_config(self, method: str) -> TrainingConfig:
        """Load configuration from YAML."""
        trainer_dir = self.repo_root / "Trainers" / f"rtx3090_{method}"
        config_path = trainer_dir / "configs" / "config.yaml"

        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Extract relevant fields
        return TrainingConfig(
            method=method,
            platform="rtx",
            config_path=config_path,
            trainer_dir=trainer_dir,
            model_name=config.get('model', {}).get('model_name', 'Unknown'),
            dataset_file=config.get('dataset', {}).get('local_file', 'Unknown'),
            epochs=config.get('training', {}).get('num_train_epochs', 1),
            batch_size=config.get('training', {}).get('per_device_train_batch_size', 4),
            learning_rate=config.get('training', {}).get('learning_rate', 0),
        )

    def execute(self, config: TrainingConfig, python_path: str) -> int:
        """Execute training script."""
        cmd = [python_path, f"train_{config.method}.py"]
        return subprocess.run(cmd, cwd=str(config.trainer_dir)).returncode

    def validate_environment(self) -> tuple[bool, str]:
        """Validate CUDA is available."""
        try:
            import torch
            if torch.cuda.is_available():
                return True, ""
            else:
                return False, "CUDA not available"
        except ImportError:
            return False, "PyTorch not installed"
```

#### Ollama Evaluation Backend

```python
# tuner/backends/evaluation/ollama_backend.py

import subprocess
from typing import List
from .base import IEvaluationBackend

class OllamaBackend(IEvaluationBackend):
    """Ollama evaluation backend."""

    @property
    def name(self) -> str:
        return "ollama"

    def list_models(self) -> List[str]:
        """List models via ollama CLI."""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                return []

            models = []
            lines = result.stdout.strip().split("\n")
            for line in lines[1:]:  # Skip header
                if line.strip():
                    parts = line.split()
                    if parts:
                        models.append(parts[0])
            return models
        except Exception:
            return []

    def validate_connection(self) -> tuple[bool, str]:
        """Check if Ollama is running."""
        try:
            models = self.list_models()
            if models:
                return True, ""
            else:
                return False, "Ollama running but no models found"
        except Exception as e:
            return False, f"Ollama not accessible: {e}"

    @property
    def default_host(self) -> str:
        return "127.0.0.1"

    @property
    def default_port(self) -> int:
        return 11434
```

### 4.3 Discovery Services

#### Checkpoint Discovery

```python
# tuner/discovery/checkpoints.py

import json
from pathlib import Path
from typing import List, Dict
from tuner.core.config import CheckpointInfo

class CheckpointDiscovery:
    """Discover and analyze training checkpoints."""

    @staticmethod
    def load_metrics(run_dir: Path) -> Dict[int, Dict]:
        """Load training metrics from logs."""
        metrics = {}
        logs_dir = run_dir / "logs"

        if not logs_dir.exists():
            return metrics

        log_files = list(logs_dir.glob("training_*.jsonl"))
        if not log_files:
            return metrics

        try:
            with open(log_files[0]) as f:
                for line in f:
                    entry = json.loads(line)
                    step = entry.get("step", 0)
                    metrics[step] = entry
        except Exception:
            pass

        return metrics

    @staticmethod
    def discover(run_dir: Path) -> List[CheckpointInfo]:
        """
        Discover all checkpoints in a training run.

        Args:
            run_dir: Training run directory

        Returns:
            List of checkpoint info objects
        """
        checkpoints = []
        metrics = CheckpointDiscovery.load_metrics(run_dir)

        # Final model
        final_model = run_dir / "final_model"
        if final_model.exists():
            checkpoints.append(CheckpointInfo(
                path=final_model,
                step=-1,
                metrics={},
                is_final=True
            ))

        # Individual checkpoints
        checkpoints_dir = run_dir / "checkpoints"
        if checkpoints_dir.exists():
            for cp_dir in sorted(checkpoints_dir.iterdir()):
                if cp_dir.is_dir() and cp_dir.name.startswith("checkpoint-"):
                    step = int(cp_dir.name.split("-")[1])
                    checkpoints.append(CheckpointInfo(
                        path=cp_dir,
                        step=step,
                        metrics=metrics.get(step, {}),
                        is_final=False
                    ))

        return checkpoints
```

### 4.4 Handler Example

```python
# tuner/handlers/train_handler.py

from tuner.core.interfaces import IHandler
from tuner.backends.registry import TrainingBackendRegistry
from tuner.ui.menu import print_menu, print_header, print_config
from tuner.ui.prompts import confirm
from tuner.utils.conda import get_conda_python

class TrainHandler(IHandler):
    """Handler for training workflow."""

    @property
    def name(self) -> str:
        return "train"

    def can_handle_direct_mode(self) -> bool:
        return True

    def handle(self) -> int:
        """Execute training workflow."""
        print_header("TRAINING", "Select your platform and training method")

        # Select platform
        platform_choice = print_menu([
            ("rtx", "NVIDIA GPU (RTX 3090 / CUDA) - SFT or KTO"),
            ("mac", "Apple Silicon (M1/M2/M3) - MLX LoRA"),
        ], "Select platform:")

        if not platform_choice:
            return 0

        # Get backend
        backend = TrainingBackendRegistry.get(platform_choice)

        # Validate environment
        is_valid, error = backend.validate_environment()
        if not is_valid:
            print(f"Error: {error}")
            return 1

        # Select method
        methods = backend.get_available_methods()
        method_options = [(m, f"{m.upper()} training") for m in methods]

        if len(methods) > 1:
            method = print_menu(method_options, "Select training method:")
            if not method:
                return 0
        else:
            method = methods[0]

        # Load configuration
        try:
            config = backend.load_config(method)
        except Exception as e:
            print(f"Error loading config: {e}")
            return 1

        # Display configuration
        print_config({
            "Platform": platform_choice.upper(),
            "Method": method.upper(),
            "Model": config.model_name,
            "Dataset": Path(config.dataset_file).name,
            "Epochs": str(config.epochs),
            "Batch Size": str(config.batch_size),
            "Learning Rate": str(config.learning_rate),
        }, "Training Configuration")

        # Confirm
        if not confirm("Start training with this configuration?"):
            print("Training cancelled.")
            return 0

        # Execute
        python = get_conda_python()
        return backend.execute(config, python)
```

---

## 5. Technology Decisions

### 5.1 Programming Language & Framework

**Python 3.8+**
- Rationale: Existing codebase is Python, no reason to change
- Existing dependencies: PyYAML, rich (optional), subprocess

### 5.2 Design Patterns

**Strategy Pattern** (from existing upload framework)
- Backends implement `ITrainingBackend` / `IEvaluationBackend`
- Allows easy addition of new backends without changing handlers

**Registry Pattern** (from existing upload framework)
- `TrainingBackendRegistry`, `EvaluationBackendRegistry`
- Centralized backend registration and retrieval

**Command Pattern** (handlers)
- Each handler implements `IHandler.handle()`
- Router dispatches to appropriate handler

**Service Locator Pattern** (discovery services)
- Discovery services find resources independently
- Handlers query discovery services as needed

### 5.3 Dependency Management

**Minimal External Dependencies:**
- `PyYAML`: YAML config parsing (already used)
- `rich`: Optional UI enhancement (already used)
- Standard library: `subprocess`, `pathlib`, `json`, `argparse`

**Internal Dependencies:**
- `Trainers/shared/ui/`: UI components (already available)
- `Trainers/shared/utilities/`: Path and env utilities (already available)
- `Trainers/shared/upload/`: Upload orchestration (already available)

### 5.4 Configuration Strategy

**YAML for Training Configuration** (existing pattern)
- Training parameters in `configs/config.yaml`
- Handlers read YAML, display to user, pass to backends

**Dataclasses for Runtime Configuration** (existing pattern in upload)
- Type-safe configuration models
- Validation at construction time
- Easy serialization

### 5.5 Error Handling Strategy

**Custom Exception Hierarchy:**
```python
# tuner/core/exceptions.py

class TunerError(Exception):
    """Base exception for tuner errors."""
    pass

class ConfigurationError(TunerError):
    """Configuration loading/parsing failed."""
    pass

class BackendError(TunerError):
    """Backend execution failed."""
    pass

class DiscoveryError(TunerError):
    """Resource discovery failed."""
    pass

class ValidationError(TunerError):
    """Validation failed."""
    pass
```

**Error Handling Pattern:**
- Exceptions bubble up to handler
- Handler displays user-friendly message
- Handler returns non-zero exit code
- No silent failures

---

## 6. Implementation Guidelines

### 6.1 Code Organization Principles

1. **Single Responsibility**: Each module has one clear purpose
2. **Open/Closed**: Easy to add new backends/handlers without modifying existing code
3. **Dependency Inversion**: Depend on interfaces (`ITrainingBackend`), not concrete implementations
4. **Separation of Concerns**: UI separate from business logic, discovery separate from execution
5. **DRY**: No duplicated UI fallback code, shared utilities for common operations

### 6.2 Naming Conventions

**Files:**
- `snake_case.py` for all Python files
- `*_backend.py` for backend implementations
- `*_handler.py` for command handlers
- `*_discovery.py` for discovery services

**Classes:**
- `PascalCase` for class names
- `I*` prefix for interfaces (e.g., `ITrainingBackend`)
- `*Backend` suffix for backends
- `*Handler` suffix for handlers
- `*Discovery` suffix for discovery services

**Functions/Methods:**
- `snake_case` for functions and methods
- Verbs for actions (e.g., `execute`, `discover`, `load_config`)
- Predicates start with `is_` or `can_` (e.g., `is_valid`, `can_handle`)

### 6.3 Testing Strategy

**Unit Tests:**
```python
# tests/backends/test_rtx_backend.py

import pytest
from tuner.backends.training.rtx_backend import RTXBackend

def test_get_available_methods():
    backend = RTXBackend(Path("/fake/repo"))
    methods = backend.get_available_methods()
    assert "sft" in methods
    assert "kto" in methods

def test_load_config_missing_file():
    backend = RTXBackend(Path("/fake/repo"))
    with pytest.raises(FileNotFoundError):
        backend.load_config("sft")
```

**Integration Tests:**
```python
# tests/handlers/test_train_handler.py

import pytest
from unittest.mock import Mock, patch
from tuner.handlers.train_handler import TrainHandler

@patch('tuner.handlers.train_handler.print_menu')
@patch('tuner.handlers.train_handler.TrainingBackendRegistry.get')
def test_train_handler_user_cancels(mock_registry, mock_menu):
    # User selects platform then cancels
    mock_menu.side_effect = ["rtx", None]

    handler = TrainHandler()
    exit_code = handler.handle()

    assert exit_code == 0  # Graceful exit
```

### 6.4 Documentation Standards

**Docstrings:**
- All public classes, methods, functions
- Google-style docstrings
- Include Args, Returns, Raises

**Example:**
```python
def load_config(self, method: str) -> TrainingConfig:
    """
    Load configuration for a training method.

    Args:
        method: Training method ('sft', 'kto', 'mlx')

    Returns:
        Parsed training configuration

    Raises:
        ConfigurationError: If config file is missing or invalid
        FileNotFoundError: If config file doesn't exist
    """
    pass
```

### 6.5 Logging Strategy

**Console Logging** (via `Trainers/shared/ui/`)
- Info: Use `print_info()`
- Success: Use `print_success()`
- Error: Use `print_error()`
- Headers: Use `print_header()`

**No File Logging at CLI Level:**
- Training/upload/eval components handle their own logging
- CLI focuses on user interaction

### 6.6 Performance Considerations

**Lazy Loading:**
- Don't load backends until needed
- Don't scan filesystem until user requests it

**Caching:**
- Cache conda Python path (find once per invocation)
- Cache environment detection (detect once)

**Subprocess Management:**
- Use `subprocess.run()` for training/upload/eval
- Don't capture output for long-running processes (let it stream)
- Proper error code propagation

---

## 7. Migration Strategy

### 7.1 Phase 1: Core Infrastructure (Week 1)

**Goals:**
- Establish directory structure
- Implement core interfaces
- Set up basic CLI entry point

**Tasks:**
1. Create directory structure: `tuner/`
2. Implement `tuner/core/interfaces.py`
3. Implement `tuner/core/config.py`
4. Implement `tuner/core/exceptions.py`
5. Create `tuner/__main__.py` entry point
6. Create `tuner/cli/parser.py` and `tuner/cli/router.py`
7. Verify imports work: `python -m tuner --help`

**Success Criteria:**
- `python -m tuner --help` shows help text
- Core interfaces compile without errors
- No change to existing `tuner.py` yet (keep as backup)

### 7.2 Phase 2: Backend Abstractions (Week 1-2)

**Goals:**
- Implement backend abstractions
- Migrate training backends

**Tasks:**
1. Implement `tuner/backends/training/base.py`
2. Implement `tuner/backends/training/rtx_backend.py`
3. Implement `tuner/backends/training/mac_backend.py`
4. Implement `tuner/backends/registry.py`
5. Unit test backends
6. Implement `tuner/backends/evaluation/base.py`
7. Implement `tuner/backends/evaluation/ollama_backend.py`
8. Implement `tuner/backends/evaluation/lmstudio_backend.py`

**Success Criteria:**
- Backends can load configs from YAML
- Backends can list available methods
- Backend validation works
- All unit tests pass

### 7.3 Phase 3: Discovery Services (Week 2)

**Goals:**
- Implement resource discovery
- Migrate checkpoint/model discovery logic

**Tasks:**
1. Implement `tuner/discovery/training_runs.py`
2. Implement `tuner/discovery/checkpoints.py`
3. Implement `tuner/discovery/models.py`
4. Implement `tuner/discovery/prompt_sets.py`
5. Unit test discovery services
6. Integration test with real training runs

**Success Criteria:**
- Discovery finds all training runs
- Checkpoint metrics parsed correctly
- Model lists match `ollama list` output
- Prompt sets discovered with counts

### 7.4 Phase 4: Handlers (Week 2-3)

**Goals:**
- Implement command handlers
- Migrate menu logic

**Tasks:**
1. Implement `tuner/handlers/base.py`
2. Implement `tuner/handlers/train_handler.py`
3. Implement `tuner/handlers/upload_handler.py`
4. Implement `tuner/handlers/eval_handler.py`
5. Implement `tuner/handlers/pipeline_handler.py`
6. Integration test each handler
7. Connect handlers to router

**Success Criteria:**
- `python -m tuner train` works end-to-end
- `python -m tuner upload` works end-to-end
- `python -m tuner eval` works end-to-end
- `python -m tuner pipeline` works end-to-end

### 7.5 Phase 5: UI Layer (Week 3)

**Goals:**
- Implement UI components
- Remove duplicated fallback code

**Tasks:**
1. Implement `tuner/ui/menu.py`
2. Implement `tuner/ui/table.py`
3. Implement `tuner/ui/prompts.py`
4. Delegate to `Trainers/shared/ui/` where possible
5. Test graceful degradation without rich

**Success Criteria:**
- UI works with rich installed
- UI works without rich (fallback)
- No duplicated UI code
- Consistent styling with upload CLI

### 7.6 Phase 6: Utilities (Week 3)

**Goals:**
- Implement utilities
- Migrate environment detection

**Tasks:**
1. Implement `tuner/utils/environment.py`
2. Implement `tuner/utils/conda.py`
3. Implement `tuner/utils/validation.py`
4. Unit test all utilities
5. Use utilities in handlers

**Success Criteria:**
- Environment detection works (WSL/Linux/Windows)
- Conda Python found correctly
- Validation helpers tested

### 7.7 Phase 7: Integration & Testing (Week 4)

**Goals:**
- End-to-end testing
- Fix bugs
- Update documentation

**Tasks:**
1. Test all workflows on WSL
2. Test all workflows on native Linux
3. Test all workflows on Windows (if supported)
4. Test all workflows on Mac
5. Fix discovered bugs
6. Update CLAUDE.md
7. Create migration guide

**Success Criteria:**
- All workflows tested on all platforms
- Zero regressions from old `tuner.py`
- Documentation updated
- Migration guide complete

### 7.8 Phase 8: Cutover (Week 4)

**Goals:**
- Replace old `tuner.py`
- Archive old implementation

**Tasks:**
1. Rename `tuner.py` → `tuner_legacy.py`
2. Create new `tuner.py` wrapper:
   ```python
   # tuner.py
   if __name__ == "__main__":
       from tuner.cli.main import main
       main()
   ```
3. Update `run.sh` and `run.ps1` (no changes needed)
4. Update CLAUDE.md to reference new structure
5. Archive `tuner_legacy.py` after verification

**Success Criteria:**
- `python tuner.py` works (imports from package)
- `python -m tuner` works (package entry point)
- All wrappers work (`./run.sh`, `./run.ps1`)
- Legacy file archived

### 7.9 Rollback Plan

If critical issues found after cutover:

1. **Immediate:** Rename `tuner_legacy.py` → `tuner.py`
2. **Root Cause:** Identify failing scenario
3. **Fix:** Patch new implementation
4. **Test:** Verify fix on all platforms
5. **Re-cutover:** Swap back when stable

**Rollback Triggers:**
- Training doesn't execute
- Upload fails silently
- Evaluation crashes
- Critical platform incompatibility

---

## 8. Risk Assessment

### 8.1 Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Regression in existing workflows | Medium | High | Comprehensive integration tests, phased rollout |
| Platform-specific bugs (Windows) | Medium | Medium | Test on all platforms, maintain fallbacks |
| Conda environment detection fails | Low | High | Extensive testing, clear error messages |
| YAML parsing incompatibility | Low | Medium | Validate against existing configs |
| Subprocess execution differences | Low | Medium | Test on all platforms, match existing behavior |

### 8.2 Organizational Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| User confusion during migration | Low | Low | Keep CLI interface identical, update docs |
| Documentation lag | Medium | Low | Update docs in parallel with code |
| Testing gaps | Medium | Medium | Define test coverage requirements upfront |

### 8.3 Dependency Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `rich` library changes | Low | Low | Maintain fallback, pin version |
| `PyYAML` incompatibility | Low | Medium | Pin version, test against existing configs |
| Training script API changes | Low | High | Coordinate with training script owners |

### 8.4 Mitigation Strategies

**Regression Prevention:**
- Automated integration tests
- Manual smoke tests on all platforms
- Side-by-side comparison during migration

**Error Handling:**
- Clear error messages with remediation steps
- Graceful degradation when components unavailable
- Validate environment before execution

**Testing:**
- Unit tests for all business logic
- Integration tests for end-to-end workflows
- Platform-specific test matrix

**Documentation:**
- Update CLAUDE.md with new architecture
- Create migration guide for developers
- Add inline code documentation

---

## Appendix A: Sample Code Patterns

### A.1 Backend Registration

```python
# tuner/backends/registry.py

from typing import Dict, Type
from .training.base import ITrainingBackend
from .training.rtx_backend import RTXBackend
from .training.mac_backend import MacBackend

class TrainingBackendRegistry:
    """Registry for training backends."""

    _backends: Dict[str, Type[ITrainingBackend]] = {
        "rtx": RTXBackend,
        "mac": MacBackend,
    }

    @classmethod
    def register(cls, name: str, backend: Type[ITrainingBackend]):
        """Register a new backend."""
        cls._backends[name] = backend

    @classmethod
    def get(cls, name: str, **kwargs) -> ITrainingBackend:
        """Get a backend instance."""
        if name not in cls._backends:
            raise ValueError(f"Unknown backend: {name}")
        return cls._backends[name](**kwargs)

    @classmethod
    def list(cls) -> list[str]:
        """List available backends."""
        return list(cls._backends.keys())
```

### A.2 CLI Entry Point

```python
# tuner/cli/main.py

import sys
from .parser import create_parser
from .router import route_command

def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    try:
        exit_code = route_command(args)
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

### A.3 Router

```python
# tuner/cli/router.py

from argparse import Namespace
from tuner.handlers.train_handler import TrainHandler
from tuner.handlers.upload_handler import UploadHandler
from tuner.handlers.eval_handler import EvalHandler
from tuner.handlers.pipeline_handler import PipelineHandler
from tuner.handlers.main_menu_handler import MainMenuHandler

def route_command(args: Namespace) -> int:
    """Route command to appropriate handler."""
    command = getattr(args, 'command', None)

    handlers = {
        'train': TrainHandler,
        'upload': UploadHandler,
        'eval': EvalHandler,
        'pipeline': PipelineHandler,
    }

    if command and command in handlers:
        handler = handlers[command]()
        return handler.handle()
    else:
        # No command = interactive menu
        handler = MainMenuHandler()
        return handler.handle()
```

---

## Appendix B: Directory Structure (Complete)

```
tuner/
├── __init__.py
├── __main__.py                    # Entry: python -m tuner
│
├── cli/                           # CLI layer
│   ├── __init__.py
│   ├── main.py                    # Main entry point
│   ├── parser.py                  # Argument parser
│   └── router.py                  # Command router
│
├── core/                          # Core abstractions
│   ├── __init__.py
│   ├── interfaces.py              # Abstract interfaces
│   ├── config.py                  # Configuration dataclasses
│   └── exceptions.py              # Custom exceptions
│
├── handlers/                      # Command handlers
│   ├── __init__.py
│   ├── base.py                    # Base handler
│   ├── train_handler.py           # Training workflow
│   ├── upload_handler.py          # Upload workflow
│   ├── eval_handler.py            # Evaluation workflow
│   ├── pipeline_handler.py        # Full pipeline
│   └── main_menu_handler.py       # Interactive main menu
│
├── backends/                      # Backend abstractions
│   ├── __init__.py
│   ├── registry.py                # Backend registry
│   ├── training/
│   │   ├── __init__.py
│   │   ├── base.py                # ITrainingBackend
│   │   ├── rtx_backend.py         # RTX SFT/KTO
│   │   └── mac_backend.py         # Mac MLX
│   └── evaluation/
│       ├── __init__.py
│       ├── base.py                # IEvaluationBackend
│       ├── ollama_backend.py      # Ollama
│       └── lmstudio_backend.py    # LM Studio
│
├── discovery/                     # Resource discovery
│   ├── __init__.py
│   ├── training_runs.py           # Find training runs
│   ├── checkpoints.py             # Find/analyze checkpoints
│   ├── models.py                  # List models from backends
│   └── prompt_sets.py             # Find prompt sets
│
├── ui/                            # UI components
│   ├── __init__.py
│   ├── menu.py                    # Menu rendering
│   ├── table.py                   # Table rendering
│   └── prompts.py                 # User input prompts
│
└── utils/                         # Utilities
    ├── __init__.py
    ├── environment.py             # Environment detection
    ├── conda.py                   # Conda path finding
    └── validation.py              # Input validation

tests/                             # Test suite
├── __init__.py
├── test_cli/
│   ├── test_parser.py
│   └── test_router.py
├── test_backends/
│   ├── test_rtx_backend.py
│   ├── test_mac_backend.py
│   ├── test_ollama_backend.py
│   └── test_lmstudio_backend.py
├── test_discovery/
│   ├── test_training_runs.py
│   ├── test_checkpoints.py
│   ├── test_models.py
│   └── test_prompt_sets.py
├── test_handlers/
│   ├── test_train_handler.py
│   ├── test_upload_handler.py
│   ├── test_eval_handler.py
│   └── test_pipeline_handler.py
└── test_utils/
    ├── test_environment.py
    ├── test_conda.py
    └── test_validation.py
```

---

## Appendix C: Key Differences from Current Implementation

| Aspect | Current (`tuner.py`) | Proposed Architecture |
|--------|---------------------|----------------------|
| **Structure** | Monolithic 1026 lines | Modular package |
| **UI Fallback** | Duplicated 70 lines | Shared via `tuner/ui/` |
| **Training Logic** | Inline in `_train_rtx()` | `RTXBackend.execute()` |
| **Checkpoint Discovery** | `_load_checkpoint_metrics()` | `CheckpointDiscovery.discover()` |
| **Model Listing** | `_list_ollama_models()` | `OllamaBackend.list_models()` |
| **Config Loading** | Inline YAML parsing | `TrainingBackend.load_config()` |
| **Error Handling** | Try/catch with prints | Custom exception hierarchy |
| **Testability** | Hard to test (subprocess calls inline) | Easy to mock interfaces |
| **Extensibility** | Add code to `tuner.py` | Implement interface, register backend |
| **Dependencies** | Implicit (imports at top) | Explicit (dependency injection) |

---

## Appendix D: Success Metrics

### Code Quality Metrics

- **Cyclomatic Complexity**: Max 10 per function (down from ~30 in current implementation)
- **Test Coverage**: Minimum 80% for business logic
- **Module Coupling**: Low (measured via dependency graphs)
- **Code Duplication**: Zero (eliminate UI fallback duplication)

### Functional Metrics

- **Regression Rate**: 0% (all existing workflows must work)
- **Error Clarity**: 100% of errors have actionable messages
- **Platform Compatibility**: Works on WSL, Linux, Mac (Windows best-effort)
- **Performance**: No degradation in execution time

### Developer Experience Metrics

- **Time to Add Backend**: < 2 hours (implement interface, register)
- **Time to Add Handler**: < 4 hours (implement workflow, tests)
- **Time to Fix Bug**: < 1 hour (isolated components)
- **Time to Onboard**: < 1 day (clear architecture, documentation)

---

## Conclusion

This architecture transforms the Synaptic Tuner CLI from a monolithic script into a maintainable, extensible application. By following SOLID principles and aligning with existing patterns in `Trainers/shared/upload/`, we achieve:

1. **Separation of Concerns**: UI, business logic, and external integrations are isolated
2. **Extensibility**: New backends, handlers, or discovery services can be added without modifying existing code
3. **Testability**: All business logic can be unit tested via interface mocking
4. **Maintainability**: Small, focused modules are easier to understand and modify
5. **Consistency**: Patterns match existing upload framework

The phased migration strategy ensures zero downtime and provides clear rollback points if issues arise. With comprehensive testing and documentation updates, this architecture will support the project's evolution for years to come.

---

**Next Steps:**
1. Review this architecture with stakeholders
2. Get approval for Phase 1 (Core Infrastructure)
3. Begin implementation with `tuner/core/interfaces.py`
4. Set up CI/CD for automated testing
5. Schedule weekly architecture reviews during migration
