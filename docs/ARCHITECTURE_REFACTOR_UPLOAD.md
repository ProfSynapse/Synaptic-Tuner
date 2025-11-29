# Upload System Architecture Analysis & Refactoring Recommendations

**Date:** 2025-11-29
**Scope:** Model upload functionality across SFT and KTO trainers
**Goal:** Apply SOLID principles and DRY practices with improved folder organization

---

## Executive Summary

The current upload system has **significant code duplication** between `rtx3090_sft` and `rtx3090_kto` trainers. Both trainers contain nearly identical copies of:
- Python upload scripts (`upload_to_hf.py`) - **1,165 lines duplicated**
- Shell scripts (`upload_model.sh`) - **204 lines duplicated**
- PowerShell scripts (`upload_model.ps1`) - **240 lines duplicated**

**Total duplication:** ~1,600+ lines of code across 6 files performing identical functions.

### Key Violations Identified

1. **DRY Violation:** Complete duplication of upload logic between trainers
2. **Single Responsibility Violation:** `upload_to_hf.py` handles 10+ responsibilities
3. **Open/Closed Violation:** Adding new model formats requires modifying core logic
4. **No abstraction:** Hard dependencies on specific implementations (Unsloth, HuggingFace)
5. **Poor modularity:** 1,165-line monolithic script with mixed concerns

### Recommended Approach

Create a **shared upload framework** in `Trainers/shared/` with:
- Strategy pattern for different save methods
- Plugin architecture for GGUF converters
- Dependency injection for platform-specific operations
- Clear separation of concerns across focused modules

**Estimated Impact:**
- Reduce code by 60-70% (~1,000+ lines eliminated)
- Improve testability and maintainability
- Enable easy addition of new formats/trainers
- Single source of truth for upload logic

---

## Current State Analysis

### 1. Code Duplication Overview

#### Identical Files Between Trainers

| File | SFT Path | KTO Path | Lines | Duplication |
|------|----------|----------|-------|-------------|
| `upload_to_hf.py` | `rtx3090_sft/src/` | `rtx3090_kto/src/` | 1,165 | 100% identical |
| `upload_model.sh` | `rtx3090_sft/` | `rtx3090_kto/` | 204 | 99% identical (line 20, 48) |
| `upload_model.ps1` | `rtx3090_sft/` | `rtx3090_kto/` | 240 | 99% identical (line 5, 48) |

**Only differences:**
- Shell scripts: Output directory name (`sft_output_rtx3090` vs `kto_output_rtx3090`)
- PowerShell scripts: Window title (`SFT` vs `KTO`)

#### Shared Dependencies (Also Duplicated)

| Module | SFT | KTO | Purpose |
|--------|-----|-----|---------|
| `model_loader.py` | ✓ | ✓ | Model loading with Unsloth |
| `inference.py` | ✓ | ✓ | Inference wrapper |
| `training_callbacks.py` | ✓ | ✓ | Training callbacks |

---

### 2. SOLID Principle Violations

#### Single Responsibility Principle (SRP)

`upload_to_hf.py` currently handles **10+ distinct responsibilities:**

1. **Windows compatibility patches** (lines 10-37)
2. **GPU memory management** (lines 53-166)
3. **Environment variable loading** (lines 184-188)
4. **Temporary directory cleanup** (lines 191-222)
5. **Local model saving** (lines 224-306)
6. **Standard model upload** (lines 309-416)
7. **GGUF creation** (lines 419-579)
8. **GGUF upload** (lines 582-620)
9. **Manifest generation** (lines 622-664)
10. **Model card generation** (lines 667-857)
11. **README creation** (lines 860-943)
12. **CLI argument parsing** (lines 946-1164)

**Violation:** Each of these should be a separate module/class with a single, well-defined purpose.

#### Open/Closed Principle (OCP)

**Problem:** Adding new save methods requires modifying core functions:

```python
# Current approach - violates OCP
def upload_standard_model(..., save_method: str = "merged_16bit"):
    if save_method == "lora":
        # LoRA-specific logic
    elif save_method == "merged_16bit":
        # 16-bit merge logic
    elif save_method == "merged_4bit":
        # 4-bit merge logic
```

**Issue:** Every new format (8-bit, GPTQ, AWQ, etc.) requires editing this function.

#### Liskov Substitution Principle (LSP)

**Not applicable** - Current design has no inheritance/abstraction to violate LSP.

**Opportunity:** Should introduce abstractions for model formats and uploaders.

#### Interface Segregation Principle (ISP)

**Not applicable** - No interfaces exist in current design.

**Opportunity:** Define focused interfaces for different upload operations.

#### Dependency Inversion Principle (DIP)

**Violation:** Direct dependencies on concrete implementations:

```python
from unsloth import FastLanguageModel  # Concrete dependency
from huggingface_hub import HfApi      # Concrete dependency
```

**Issue:**
- Cannot swap model loading implementations
- Tightly coupled to Unsloth
- Cannot mock for testing without complex patches
- Cannot support other frameworks (vanilla transformers, llama.cpp native, etc.)

---

### 3. Architecture Smells

#### God Function Anti-Pattern

`main()` function (lines 946-1164) orchestrates everything:
- CLI parsing
- Token validation
- Directory detection
- Upload orchestration
- GGUF creation
- Documentation generation

**Problem:** 218 lines doing too much, violating SRP.

#### Primitive Obsession

Configuration passed as primitive strings and booleans:
```python
def upload_standard_model(
    model_path: str,      # Should be Path or ModelPath object
    repo_id: str,         # Should be RepositoryId object
    hf_token: str,        # Should be Credential object
    save_method: str,     # Should be SaveMethod enum/strategy
    private: bool,        # OK
    output_dir: Path,     # Good
    save_local: bool      # OK
)
```

#### Feature Envy

GPU memory functions (lines 53-166) belong in a `GPUMemoryManager` class, not in upload script.

#### Shotgun Surgery

Adding GGUF support required changes in:
- `upload_to_hf.py` (3+ functions)
- `upload_model.sh` (2+ sections)
- `upload_model.ps1` (2+ sections)
- Duplication across both trainers

---

## Proposed Architecture

### 1. New Folder Structure

```
Trainers/
├── shared/                          # NEW: Shared utilities
│   ├── __init__.py
│   ├── upload/                      # Upload framework
│   │   ├── __init__.py
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── interfaces.py       # Abstract base classes
│   │   │   ├── config.py           # Configuration models
│   │   │   ├── exceptions.py       # Custom exceptions
│   │   │   └── types.py            # Type definitions
│   │   ├── strategies/              # Strategy pattern for save methods
│   │   │   ├── __init__.py
│   │   │   ├── base.py             # BaseSaveStrategy
│   │   │   ├── lora.py             # LoRASaveStrategy
│   │   │   ├── merged_16bit.py     # Merged16BitStrategy
│   │   │   ├── merged_4bit.py      # Merged4BitStrategy
│   │   │   └── registry.py         # Strategy registry
│   │   ├── converters/              # Format converters
│   │   │   ├── __init__.py
│   │   │   ├── base.py             # BaseConverter
│   │   │   ├── gguf.py             # GGUFConverter
│   │   │   └── registry.py         # Converter registry
│   │   ├── uploaders/               # Upload implementations
│   │   │   ├── __init__.py
│   │   │   ├── base.py             # BaseUploader
│   │   │   ├── huggingface.py      # HuggingFaceUploader
│   │   │   └── registry.py         # Uploader registry
│   │   ├── documentation/           # Documentation generation
│   │   │   ├── __init__.py
│   │   │   ├── manifest.py         # ManifestGenerator
│   │   │   ├── model_card.py       # ModelCardGenerator
│   │   │   └── readme.py           # ReadmeGenerator
│   │   ├── platform/                # Platform-specific utilities
│   │   │   ├── __init__.py
│   │   │   ├── gpu_memory.py       # GPUMemoryManager
│   │   │   ├── filesystem.py       # FilesystemHelper (WSL detection)
│   │   │   └── windows_patches.py  # Windows compatibility
│   │   ├── cli/                     # CLI interfaces
│   │   │   ├── __init__.py
│   │   │   ├── upload_cli.py       # Main CLI
│   │   │   └── interactive.py      # Interactive prompts
│   │   └── orchestrator.py          # UploadOrchestrator
│   ├── model_loading/               # NEW: Shared model loading
│   │   ├── __init__.py
│   │   ├── base.py                 # BaseModelLoader
│   │   ├── unsloth_loader.py       # UnslothModelLoader
│   │   └── registry.py             # Loader registry
│   └── utilities/                   # NEW: Shared utilities
│       ├── __init__.py
│       ├── env.py                  # Environment variable handling
│       ├── paths.py                # Path utilities
│       └── validation.py           # Input validation
├── rtx3090_sft/
│   ├── src/
│   │   ├── upload_to_hf.py         # SIMPLIFIED: Thin wrapper
│   │   ├── data_loader.py
│   │   ├── model_loader.py         # REFACTORED: Uses shared/model_loading
│   │   └── ...
│   ├── upload_model.sh             # SIMPLIFIED: Uses shared CLI
│   ├── upload_model.ps1            # SIMPLIFIED: Uses shared CLI
│   └── ...
├── rtx3090_kto/
│   ├── src/
│   │   ├── upload_to_hf.py         # SIMPLIFIED: Thin wrapper (or removed)
│   │   ├── data_loader.py
│   │   ├── model_loader.py         # REFACTORED: Uses shared/model_loading
│   │   └── ...
│   ├── upload_model.sh             # SIMPLIFIED: Uses shared CLI
│   ├── upload_model.ps1            # SIMPLIFIED: Uses shared CLI
│   └── ...
└── scripts/                         # NEW: Cross-trainer scripts
    ├── upload_model.py              # Universal upload CLI
    └── ...
```

### 2. Core Abstractions (Interfaces)

#### `shared/upload/core/interfaces.py`

```python
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any
from .types import ModelPath, RepositoryId, Credential

class ISaveStrategy(ABC):
    """Strategy for saving models in different formats."""

    @abstractmethod
    def save(
        self,
        model_path: ModelPath,
        output_dir: Path,
        **kwargs
    ) -> Path:
        """Save model using this strategy."""
        pass

    @abstractmethod
    def estimate_size_gb(self, model_size: str) -> float:
        """Estimate output size in GB."""
        pass

    @abstractmethod
    def requires_gpu(self) -> bool:
        """Whether this strategy requires GPU."""
        pass

class IConverter(ABC):
    """Converter for model formats (e.g., GGUF)."""

    @abstractmethod
    def convert(
        self,
        model_path: ModelPath,
        output_dir: Path,
        **options
    ) -> list[Path]:
        """Convert model to target format."""
        pass

    @abstractmethod
    def supported_quantizations(self) -> list[str]:
        """Get supported quantization methods."""
        pass

class IUploader(ABC):
    """Uploader for model repositories."""

    @abstractmethod
    def upload_model(
        self,
        local_path: Path,
        repo_id: RepositoryId,
        credential: Credential,
        **options
    ) -> str:
        """Upload model to repository."""
        pass

    @abstractmethod
    def upload_file(
        self,
        file_path: Path,
        repo_id: RepositoryId,
        credential: Credential,
        path_in_repo: str
    ) -> None:
        """Upload single file to repository."""
        pass

class IModelLoader(ABC):
    """Model loader abstraction."""

    @abstractmethod
    def load_model(
        self,
        model_path: ModelPath,
        **config
    ) -> tuple[Any, Any]:  # (model, tokenizer)
        """Load model and tokenizer."""
        pass

    @abstractmethod
    def save_merged(
        self,
        model: Any,
        tokenizer: Any,
        output_path: Path,
        save_method: str
    ) -> None:
        """Save merged model."""
        pass

class IDocumentationGenerator(ABC):
    """Documentation generator for models."""

    @abstractmethod
    def generate(self, **data) -> str:
        """Generate documentation content."""
        pass

    @abstractmethod
    def save(self, output_path: Path) -> Path:
        """Save documentation to file."""
        pass
```

#### `shared/upload/core/config.py`

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from .types import ModelPath, RepositoryId, Credential

@dataclass
class UploadConfig:
    """Configuration for model upload."""
    model_path: ModelPath
    repo_id: RepositoryId
    credential: Credential
    output_dir: Optional[Path] = None
    private: bool = False

@dataclass
class SaveConfig:
    """Configuration for model saving."""
    strategy_name: str  # "lora", "merged_16bit", "merged_4bit"
    save_local: bool = True

@dataclass
class ConversionConfig:
    """Configuration for format conversion."""
    converter_name: str  # "gguf"
    quantizations: list[str] = None
    cleanup_temp: bool = True

@dataclass
class DocumentationConfig:
    """Configuration for documentation generation."""
    training_lineage_path: Optional[Path] = None
    include_manifest: bool = True
    include_model_card: bool = True
    include_readme: bool = True
```

### 3. Strategy Pattern Implementation

#### `shared/upload/strategies/base.py`

```python
from abc import ABC, abstractmethod
from pathlib import Path
from ..core.interfaces import ISaveStrategy
from ..core.types import ModelPath

class BaseSaveStrategy(ISaveStrategy):
    """Base implementation of save strategy."""

    def __init__(self, model_loader):
        self.model_loader = model_loader

    def save(
        self,
        model_path: ModelPath,
        output_dir: Path,
        **kwargs
    ) -> Path:
        """Template method for saving."""
        self._pre_save_check(model_path, output_dir)
        result = self._execute_save(model_path, output_dir, **kwargs)
        self._post_save_cleanup(output_dir)
        return result

    @abstractmethod
    def _execute_save(
        self,
        model_path: ModelPath,
        output_dir: Path,
        **kwargs
    ) -> Path:
        """Actual save implementation."""
        pass

    def _pre_save_check(self, model_path: ModelPath, output_dir: Path):
        """Pre-save validation."""
        output_dir.mkdir(parents=True, exist_ok=True)

    def _post_save_cleanup(self, output_dir: Path):
        """Post-save cleanup."""
        pass
```

#### `shared/upload/strategies/lora.py`

```python
import shutil
from pathlib import Path
from .base import BaseSaveStrategy
from ..core.types import ModelPath

class LoRASaveStrategy(BaseSaveStrategy):
    """Strategy for saving LoRA adapters only."""

    def _execute_save(
        self,
        model_path: ModelPath,
        output_dir: Path,
        **kwargs
    ) -> Path:
        """Copy LoRA adapters."""
        save_dir = output_dir / "lora"
        shutil.copytree(str(model_path), save_dir, dirs_exist_ok=True)
        return save_dir

    def estimate_size_gb(self, model_size: str) -> float:
        """Estimate LoRA adapter size."""
        sizes = {
            "3b": 0.2,
            "7b": 0.32,
            "13b": 0.5,
            "20b": 0.8
        }
        return sizes.get(model_size, 0.32)

    def requires_gpu(self) -> bool:
        """LoRA copy doesn't need GPU."""
        return False
```

#### `shared/upload/strategies/merged_16bit.py`

```python
from pathlib import Path
from .base import BaseSaveStrategy
from ..core.types import ModelPath
from ..platform.gpu_memory import ensure_gpu_memory

class Merged16BitStrategy(BaseSaveStrategy):
    """Strategy for saving 16-bit merged models."""

    def _execute_save(
        self,
        model_path: ModelPath,
        output_dir: Path,
        **kwargs
    ) -> Path:
        """Load and save 16-bit merged model."""
        # Check GPU memory
        required_gb = self.estimate_size_gb(kwargs.get("model_size", "7b"))
        ensure_gpu_memory(required_gb, "16-bit model merge")

        # Load model
        model, tokenizer = self.model_loader.load_model(
            model_path,
            load_in_4bit=False
        )

        # Save merged
        save_dir = output_dir / "merged-16bit"
        self.model_loader.save_merged(
            model,
            tokenizer,
            save_dir,
            save_method="merged_16bit"
        )

        return save_dir

    def estimate_size_gb(self, model_size: str) -> float:
        """Estimate 16-bit merged size."""
        sizes = {
            "3b": 7.0,
            "7b": 14.0,
            "13b": 26.0,
            "20b": 40.0
        }
        return sizes.get(model_size, 14.0)

    def requires_gpu(self) -> bool:
        """16-bit merge requires GPU."""
        return True
```

#### `shared/upload/strategies/registry.py`

```python
from typing import Dict, Type
from .base import BaseSaveStrategy
from .lora import LoRASaveStrategy
from .merged_16bit import Merged16BitStrategy
from .merged_4bit import Merged4BitStrategy

class SaveStrategyRegistry:
    """Registry for save strategies."""

    _strategies: Dict[str, Type[BaseSaveStrategy]] = {
        "lora": LoRASaveStrategy,
        "merged_16bit": Merged16BitStrategy,
        "merged_4bit": Merged4BitStrategy,
    }

    @classmethod
    def register(cls, name: str, strategy_class: Type[BaseSaveStrategy]):
        """Register a new strategy."""
        cls._strategies[name] = strategy_class

    @classmethod
    def get(cls, name: str, model_loader) -> BaseSaveStrategy:
        """Get strategy instance."""
        strategy_class = cls._strategies.get(name)
        if not strategy_class:
            raise ValueError(f"Unknown save strategy: {name}")
        return strategy_class(model_loader)

    @classmethod
    def list_strategies(cls) -> list[str]:
        """List available strategies."""
        return list(cls._strategies.keys())
```

### 4. Orchestrator Pattern

#### `shared/upload/orchestrator.py`

```python
from pathlib import Path
from typing import Optional
from .core.config import (
    UploadConfig,
    SaveConfig,
    ConversionConfig,
    DocumentationConfig
)
from .strategies.registry import SaveStrategyRegistry
from .converters.registry import ConverterRegistry
from .uploaders.registry import UploaderRegistry
from .documentation.manifest import ManifestGenerator
from .documentation.model_card import ModelCardGenerator
from .documentation.readme import ReadmeGenerator

class UploadOrchestrator:
    """
    Orchestrates the complete model upload process.

    Responsibilities:
    - Coordinate saving, conversion, upload, and documentation
    - Manage dependencies between steps
    - Handle errors and cleanup
    """

    def __init__(
        self,
        upload_config: UploadConfig,
        save_config: SaveConfig,
        conversion_config: Optional[ConversionConfig] = None,
        documentation_config: Optional[DocumentationConfig] = None,
        model_loader=None
    ):
        self.upload_config = upload_config
        self.save_config = save_config
        self.conversion_config = conversion_config
        self.documentation_config = documentation_config or DocumentationConfig()
        self.model_loader = model_loader

        # Determine output directory
        self.output_dir = self._resolve_output_dir()

        # State tracking
        self.artifacts_created = []
        self.formats_created = []

    def execute(self) -> dict:
        """
        Execute the complete upload workflow.

        Returns:
            Dictionary with upload results and artifact paths
        """
        try:
            # Step 1: Save model locally
            saved_path = self._save_model_locally()

            # Step 2: Upload to repository
            upload_url = self._upload_to_repository(saved_path)

            # Step 3: Convert to additional formats (optional)
            converted_files = self._convert_formats()

            # Step 4: Upload converted files (optional)
            if converted_files:
                self._upload_converted_files(converted_files)

            # Step 5: Generate documentation
            docs = self._generate_documentation()

            # Step 6: Upload documentation
            self._upload_documentation(docs)

            return {
                "success": True,
                "upload_url": upload_url,
                "local_artifacts": self.output_dir,
                "formats": self.formats_created,
                "documentation": docs
            }

        except Exception as e:
            # Cleanup on failure
            self._cleanup_on_error()
            raise

    def _save_model_locally(self) -> Path:
        """Save model using configured strategy."""
        if not self.save_config.save_local:
            return self.upload_config.model_path

        strategy = SaveStrategyRegistry.get(
            self.save_config.strategy_name,
            self.model_loader
        )

        saved_path = strategy.save(
            self.upload_config.model_path,
            self.output_dir
        )

        self.formats_created.append(self.save_config.strategy_name)
        self.artifacts_created.append(saved_path)

        return saved_path

    def _upload_to_repository(self, local_path: Path) -> str:
        """Upload model to repository."""
        uploader = UploaderRegistry.get("huggingface")

        url = uploader.upload_model(
            local_path,
            self.upload_config.repo_id,
            self.upload_config.credential,
            private=self.upload_config.private
        )

        return url

    def _convert_formats(self) -> list[Path]:
        """Convert to additional formats."""
        if not self.conversion_config:
            return []

        converter = ConverterRegistry.get(
            self.conversion_config.converter_name
        )

        converted_files = converter.convert(
            self.upload_config.model_path,
            self.output_dir,
            quantizations=self.conversion_config.quantizations
        )

        self.formats_created.append(self.conversion_config.converter_name)
        self.artifacts_created.extend(converted_files)

        return converted_files

    def _upload_converted_files(self, files: list[Path]):
        """Upload converted files."""
        uploader = UploaderRegistry.get("huggingface")

        for file in files:
            uploader.upload_file(
                file,
                self.upload_config.repo_id,
                self.upload_config.credential,
                path_in_repo=file.name
            )

    def _generate_documentation(self) -> dict:
        """Generate all documentation."""
        docs = {}

        if self.documentation_config.include_manifest:
            manifest_gen = ManifestGenerator()
            docs["manifest"] = manifest_gen.generate(
                repo_id=self.upload_config.repo_id,
                formats=self.formats_created,
                # ... other params
            )
            manifest_gen.save(self.output_dir / "upload_manifest.json")

        if self.documentation_config.include_model_card:
            card_gen = ModelCardGenerator()
            if self.documentation_config.training_lineage_path:
                lineage = self._load_training_lineage()
                docs["model_card"] = card_gen.generate(lineage=lineage)
            card_gen.save(self.output_dir / "README.md")

        return docs

    def _upload_documentation(self, docs: dict):
        """Upload documentation files."""
        uploader = UploaderRegistry.get("huggingface")

        for doc_name, doc_path in [
            ("upload_manifest.json", self.output_dir / "upload_manifest.json"),
            ("README.md", self.output_dir / "README.md"),
        ]:
            if doc_path.exists():
                uploader.upload_file(
                    doc_path,
                    self.upload_config.repo_id,
                    self.upload_config.credential,
                    path_in_repo=doc_name
                )

    def _resolve_output_dir(self) -> Path:
        """Determine output directory."""
        if self.upload_config.output_dir:
            return self.upload_config.output_dir

        # Auto-detect from model path
        model_path = Path(self.upload_config.model_path)
        if model_path.name == "final_model":
            training_run_dir = model_path.parent
            model_name = self.upload_config.repo_id.split('/')[-1]
            return training_run_dir / model_name

        return model_path.parent / "upload_output"

    def _cleanup_on_error(self):
        """Cleanup artifacts on error."""
        # Could implement cleanup logic here
        pass

    def _load_training_lineage(self) -> dict:
        """Load training lineage if available."""
        import json
        path = self.documentation_config.training_lineage_path
        if path and path.exists():
            with open(path) as f:
                return json.load(f)
        return {}
```

### 5. Simplified Entry Points

#### Trainer-Specific Entry Point (Minimal)

`rtx3090_sft/src/upload_to_hf.py` becomes a thin wrapper:

```python
"""
SFT model upload entry point.
This is a thin wrapper around the shared upload framework.
"""

import sys
from pathlib import Path

# Add shared modules to path
SHARED_PATH = Path(__file__).parent.parent.parent / "shared"
sys.path.insert(0, str(SHARED_PATH))

from upload.cli.upload_cli import main

if __name__ == "__main__":
    # Override default output directory for SFT
    import os
    os.environ.setdefault("TRAINER_OUTPUT_DIR", "sft_output_rtx3090")
    main()
```

**Result:** 10 lines instead of 1,165 lines (99% reduction)

#### Universal CLI Entry Point

`Trainers/scripts/upload_model.py`:

```python
"""
Universal model upload CLI for all trainers.
"""

import argparse
from pathlib import Path
from shared.upload.orchestrator import UploadOrchestrator
from shared.upload.core.config import (
    UploadConfig,
    SaveConfig,
    ConversionConfig,
    DocumentationConfig
)
from shared.upload.core.types import ModelPath, RepositoryId, Credential
from shared.model_loading.registry import ModelLoaderRegistry
from shared.utilities.env import load_env_file

def main():
    parser = argparse.ArgumentParser(description="Upload model to HuggingFace Hub")

    # Required arguments
    parser.add_argument("model_path", type=str, help="Path to saved model")
    parser.add_argument("repo_id", type=str, help="HuggingFace repo ID")

    # Upload configuration
    parser.add_argument("--token", type=str, help="HuggingFace token")
    parser.add_argument("--private", action="store_true", help="Private repo")
    parser.add_argument("--output-dir", type=str, help="Output directory")

    # Save configuration
    parser.add_argument(
        "--save-method",
        type=str,
        default="merged_16bit",
        choices=["merged_16bit", "merged_4bit", "lora"],
        help="Save method"
    )
    parser.add_argument("--no-save-local", action="store_true")

    # Conversion configuration
    parser.add_argument("--create-gguf", action="store_true")
    parser.add_argument(
        "--gguf-quantizations",
        nargs="+",
        default=["Q4_K_M", "Q5_K_M", "Q8_0"]
    )

    # Documentation configuration
    parser.add_argument("--training-lineage", type=str)

    args = parser.parse_args()

    # Load environment
    load_env_file()

    # Get token
    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HuggingFace token required")

    # Create configurations
    upload_config = UploadConfig(
        model_path=ModelPath(args.model_path),
        repo_id=RepositoryId(args.repo_id),
        credential=Credential(token),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        private=args.private
    )

    save_config = SaveConfig(
        strategy_name=args.save_method,
        save_local=not args.no_save_local
    )

    conversion_config = None
    if args.create_gguf:
        conversion_config = ConversionConfig(
            converter_name="gguf",
            quantizations=args.gguf_quantizations
        )

    documentation_config = DocumentationConfig(
        training_lineage_path=Path(args.training_lineage) if args.training_lineage else None
    )

    # Get model loader
    model_loader = ModelLoaderRegistry.get("unsloth")

    # Create orchestrator
    orchestrator = UploadOrchestrator(
        upload_config=upload_config,
        save_config=save_config,
        conversion_config=conversion_config,
        documentation_config=documentation_config,
        model_loader=model_loader
    )

    # Execute upload
    result = orchestrator.execute()

    # Print results
    print(f"\n✓ Upload complete!")
    print(f"URL: {result['upload_url']}")
    print(f"Local artifacts: {result['local_artifacts']}")

if __name__ == "__main__":
    main()
```

### 6. Simplified Shell Scripts

#### Universal Bash Script

`Trainers/scripts/upload_model.sh`:

```bash
#!/bin/bash
# Universal upload script for all trainers

set -e

# Detect trainer type from current directory
TRAINER_DIR=$(basename "$(pwd)")
case "$TRAINER_DIR" in
    rtx3090_sft)
        OUTPUT_DIR="./sft_output_rtx3090"
        ;;
    rtx3090_kto)
        OUTPUT_DIR="./kto_output_rtx3090"
        ;;
    *)
        echo "Error: Unknown trainer directory: $TRAINER_DIR"
        exit 1
        ;;
esac

# Source the shared interactive CLI
source ../scripts/upload_interactive.sh "$OUTPUT_DIR"
```

**Result:** 20 lines instead of 204 lines (90% reduction)

---

## Implementation Roadmap

### Phase 1: Foundation (Week 1)

**Goal:** Create shared infrastructure without breaking existing code

**Tasks:**
1. Create `Trainers/shared/` directory structure
2. Implement core interfaces (`interfaces.py`, `types.py`, `config.py`)
3. Implement base classes (`BaseSaveStrategy`, `BaseConverter`, etc.)
4. Add unit tests for core abstractions

**Deliverables:**
- `shared/upload/core/` module fully implemented
- Unit tests with 80%+ coverage
- Documentation for interfaces

**Risk:** None - no existing code affected

### Phase 2: Extract Platform Utilities (Week 2)

**Goal:** Move platform-specific code to shared modules

**Tasks:**
1. Extract GPU memory management to `shared/upload/platform/gpu_memory.py`
2. Extract Windows patches to `shared/upload/platform/windows_patches.py`
3. Extract filesystem helpers to `shared/upload/platform/filesystem.py`
4. Update both trainers to import from shared

**Deliverables:**
- `shared/upload/platform/` module complete
- Both trainers using shared platform utilities
- Integration tests passing

**Risk:** Low - backwards compatible changes

### Phase 3: Implement Strategies (Week 3)

**Goal:** Implement save strategies and registry

**Tasks:**
1. Implement `LoRASaveStrategy`
2. Implement `Merged16BitStrategy`
3. Implement `Merged4BitStrategy`
4. Implement `SaveStrategyRegistry`
5. Add strategy unit tests

**Deliverables:**
- All save strategies implemented
- Strategy registry working
- Strategy tests with 90%+ coverage

**Risk:** Low - new code, doesn't affect existing

### Phase 4: Implement Converters & Uploaders (Week 4)

**Goal:** Implement format converters and upload backends

**Tasks:**
1. Implement `GGUFConverter`
2. Implement `ConverterRegistry`
3. Implement `HuggingFaceUploader`
4. Implement `UploaderRegistry`
5. Add integration tests

**Deliverables:**
- GGUF conversion working via converter pattern
- HuggingFace upload working via uploader pattern
- Integration tests passing

**Risk:** Medium - GGUF conversion is complex

### Phase 5: Documentation Generators (Week 5)

**Goal:** Extract documentation generation

**Tasks:**
1. Implement `ManifestGenerator`
2. Implement `ModelCardGenerator`
3. Implement `ReadmeGenerator`
4. Add tests for generators

**Deliverables:**
- Documentation generators working
- Generated docs match current format
- Tests covering edge cases

**Risk:** Low - mostly string formatting

### Phase 6: Orchestrator (Week 6)

**Goal:** Implement upload orchestrator

**Tasks:**
1. Implement `UploadOrchestrator`
2. Add error handling and cleanup
3. Add progress reporting
4. Integration tests

**Deliverables:**
- Orchestrator coordinates full workflow
- Error handling robust
- Progress reporting clear

**Risk:** Medium - orchestration logic complex

### Phase 7: CLI Migration (Week 7)

**Goal:** Create universal CLI and migrate trainers

**Tasks:**
1. Implement `shared/upload/cli/upload_cli.py`
2. Create `Trainers/scripts/upload_model.py`
3. Update `rtx3090_sft/src/upload_to_hf.py` to wrapper
4. Update `rtx3090_kto/src/upload_to_hf.py` to wrapper
5. Update shell scripts
6. Update PowerShell scripts

**Deliverables:**
- Universal CLI working
- Both trainers using shared CLI
- Shell scripts simplified
- PowerShell scripts simplified

**Risk:** Medium - must maintain backwards compatibility

### Phase 8: Testing & Validation (Week 8)

**Goal:** Comprehensive testing and validation

**Tasks:**
1. End-to-end tests for SFT upload
2. End-to-end tests for KTO upload
3. Test all save strategies
4. Test GGUF conversion
5. Test documentation generation
6. Performance testing
7. Cross-platform testing (WSL, Windows, Linux)

**Deliverables:**
- All tests passing
- Coverage > 85%
- Performance comparable or better
- Works on all platforms

**Risk:** High - integration testing can reveal issues

### Phase 9: Documentation & Cleanup (Week 9)

**Goal:** Document new system and remove old code

**Tasks:**
1. Write migration guide
2. Update CLAUDE.md
3. Create architecture diagrams
4. Add inline code documentation
5. Remove old duplicate code
6. Update README files

**Deliverables:**
- Complete documentation
- Old code removed
- Clean git history

**Risk:** Low - documentation and cleanup

---

## Benefits Analysis

### Code Reduction

| Component | Before | After | Reduction |
|-----------|--------|-------|-----------|
| Core upload logic | 2,330 lines (2x duplicated) | 400 lines | 83% |
| Shell scripts | 408 lines (2x duplicated) | 40 lines | 90% |
| PowerShell scripts | 480 lines (2x duplicated) | 50 lines | 90% |
| **Total** | **3,218 lines** | **490 lines** | **85%** |

### Maintainability Improvements

1. **Single Source of Truth:** Bug fixes apply to all trainers automatically
2. **Easier Testing:** Small, focused modules are easier to test
3. **Clear Responsibilities:** Each module has one job
4. **Extension Points:** New formats/trainers added without modifying core
5. **Better Documentation:** Interfaces document contracts explicitly

### Extensibility Examples

**Adding a new save method (GPTQ):**

Before (violates OCP):
```python
# Must edit upload_to_hf.py in both trainers
def upload_standard_model(..., save_method):
    if save_method == "gptq":  # NEW CODE IN CORE
        # GPTQ logic
```

After (follows OCP):
```python
# Create new file: shared/upload/strategies/gptq.py
class GPTQSaveStrategy(BaseSaveStrategy):
    def _execute_save(self, ...):
        # GPTQ logic

# Register it
SaveStrategyRegistry.register("gptq", GPTQSaveStrategy)
```

**Adding a new trainer:**

Before:
- Copy all upload files
- Update paths
- Maintain 3 more files

After:
- Create 10-line wrapper
- Done

### Performance Considerations

**Potential concerns:**
- Extra abstraction layers might add overhead
- Registry lookups add small cost

**Mitigations:**
- Registry lookups are one-time (negligible)
- Strategy pattern adds zero runtime overhead
- GPU/network operations dominate timing (abstraction is <1%)

**Expected performance:** Identical or slightly better (better memory management)

---

## Testing Strategy

### Unit Tests

**Coverage targets:**
- Core interfaces: 100%
- Strategies: 95%
- Converters: 90%
- Uploaders: 85%
- Documentation generators: 90%

**Test structure:**
```
Trainers/shared/tests/
├── unit/
│   ├── test_strategies.py
│   ├── test_converters.py
│   ├── test_uploaders.py
│   ├── test_documentation.py
│   └── test_platform.py
├── integration/
│   ├── test_orchestrator.py
│   ├── test_sft_upload.py
│   ├── test_kto_upload.py
│   └── test_gguf_workflow.py
└── fixtures/
    ├── mock_models/
    └── test_configs/
```

### Integration Tests

**Key scenarios:**
1. SFT upload with merged_16bit
2. KTO upload with merged_4bit
3. Upload with GGUF conversion
4. Upload with training lineage
5. Error handling (insufficient GPU, network failure, etc.)

### Mocking Strategy

**Mock external dependencies:**
- `FastLanguageModel` (Unsloth)
- `HfApi` (HuggingFace)
- GPU operations
- Filesystem operations (for temp file testing)

**Tools:**
- `pytest` for test framework
- `pytest-mock` for mocking
- `pytest-cov` for coverage
- `hypothesis` for property-based testing

---

## Migration Guide

### For Developers

**Before (current):**
```bash
cd Trainers/rtx3090_sft
python src/upload_to_hf.py ./sft_output/20251129/final_model username/model
```

**After (new):**
```bash
cd Trainers/rtx3090_sft
python src/upload_to_hf.py ./sft_output/20251129/final_model username/model
# Wrapper automatically uses shared CLI
```

**No change required** - wrappers maintain backwards compatibility

### For CI/CD

**Update scripts to use universal CLI:**
```bash
# Old approach (still works via wrappers)
python Trainers/rtx3090_sft/src/upload_to_hf.py ...

# New approach (recommended)
python Trainers/scripts/upload_model.py ...
```

### For New Trainers

**Creating a new trainer:**

1. Create trainer directory: `Trainers/my_new_trainer/`
2. Create thin upload wrapper (10 lines)
3. Update shell/PowerShell scripts (20 lines each)
4. Done - all upload logic inherited

**Example wrapper:**
```python
# Trainers/my_new_trainer/src/upload_to_hf.py
import sys, os
from pathlib import Path
SHARED_PATH = Path(__file__).parent.parent.parent / "shared"
sys.path.insert(0, str(SHARED_PATH))
os.environ.setdefault("TRAINER_OUTPUT_DIR", "my_trainer_output")
from upload.cli.upload_cli import main
if __name__ == "__main__":
    main()
```

---

## Risk Assessment

### High Risk Areas

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Breaking existing workflows | High | Medium | Maintain wrappers, extensive testing |
| GGUF conversion errors | High | Low | Comprehensive integration tests |
| Platform-specific issues | Medium | Medium | Test on Windows, WSL, Linux |
| Performance regression | Low | Low | Benchmark before/after |

### Rollback Plan

**If critical issues arise:**

1. **Phase 1-6:** No rollback needed (new code, existing unchanged)
2. **Phase 7:** Revert wrapper changes, use old upload scripts
3. **Phase 8-9:** Revert git commits, restore old files

**Rollback time:** < 1 hour (git revert)

---

## Success Metrics

### Quantitative Metrics

- **Code reduction:** 85% fewer lines
- **Duplication:** 0% (down from 100%)
- **Test coverage:** >85% (up from ~0%)
- **Bugs per release:** <2 (baseline TBD)
- **Time to add new format:** <1 day (down from 1 week)

### Qualitative Metrics

- **Developer satisfaction:** Survey after 1 month
- **Ease of extension:** Time to add new trainer
- **Code clarity:** Readability review
- **Documentation quality:** Completeness check

### Review Points

- **After Phase 6:** Review orchestrator design
- **After Phase 7:** Review CLI usability
- **After Phase 8:** Review test coverage
- **After Phase 9:** Final architecture review

---

## Conclusion

The proposed refactoring will:

1. **Eliminate 85% of duplicate code** (~2,700 lines)
2. **Apply SOLID principles** throughout upload system
3. **Enable easy extension** via strategy/plugin patterns
4. **Improve testability** with dependency injection
5. **Maintain backwards compatibility** via thin wrappers
6. **Reduce maintenance burden** with single source of truth

**Recommendation:** Proceed with phased implementation starting with Phase 1.

**Timeline:** 9 weeks (part-time) or 5 weeks (full-time)

**Next steps:**
1. Review and approve this architecture
2. Create detailed task breakdown for Phase 1
3. Set up test infrastructure
4. Begin implementation

---

## Appendix A: File-by-File Comparison

### Exact Duplication

**`upload_to_hf.py` (SFT vs KTO):**
- Lines 1-1165: **100% identical**
- No differences in logic
- No trainer-specific code
- Complete waste of duplication

**`upload_model.sh` (SFT vs KTO):**
- Line 20: `OUTPUT_DIR` different
- Line 48: `OUTPUT_DIR` different
- Lines 1-204: 99% identical
- Difference: 2 lines out of 204

**`upload_model.ps1` (SFT vs KTO):**
- Line 5: Window title different
- Line 48: `OUTPUT_DIR` different
- Lines 1-240: 99% identical
- Difference: 2 lines out of 240

### Conclusion from Comparison

**The files are functionally identical.** The only differences are:
1. Output directory names
2. Cosmetic labels

This is a **textbook case** for refactoring to shared code.

---

## Appendix B: SOLID Principles Applied

### Single Responsibility Principle

**Before:** `upload_to_hf.py` has 12 responsibilities
**After:** Each module has 1 responsibility

| Module | Single Responsibility |
|--------|----------------------|
| `GPUMemoryManager` | GPU memory checking and cleanup |
| `WindowsPatches` | Windows compatibility patches |
| `LoRASaveStrategy` | Saving LoRA adapters |
| `GGUFConverter` | Converting to GGUF format |
| `HuggingFaceUploader` | Uploading to HuggingFace |
| `ManifestGenerator` | Generating upload manifest |
| `ModelCardGenerator` | Generating model card |
| `UploadOrchestrator` | Coordinating upload workflow |

### Open/Closed Principle

**Before:** Modify core functions to add formats
**After:** Extend via new strategy classes

```python
# Adding 8-bit quantization - no core modification needed
class Merged8BitStrategy(BaseSaveStrategy):
    # Implementation
    pass

SaveStrategyRegistry.register("merged_8bit", Merged8BitStrategy)
```

### Liskov Substitution Principle

**Before:** No abstractions to substitute
**After:** All strategies substitutable

```python
# Any ISaveStrategy can be used interchangeably
def save_model(strategy: ISaveStrategy, model_path, output_dir):
    return strategy.save(model_path, output_dir)

# Works with any strategy
save_model(LoRASaveStrategy(...), ...)
save_model(Merged16BitStrategy(...), ...)
save_model(Merged8BitStrategy(...), ...)  # New strategy
```

### Interface Segregation Principle

**Before:** No interfaces
**After:** Focused interfaces

```python
# Clients only depend on what they use
class ISaveStrategy(ABC):
    # Only save-related methods
    pass

class IConverter(ABC):
    # Only conversion-related methods
    pass

class IUploader(ABC):
    # Only upload-related methods
    pass
```

### Dependency Inversion Principle

**Before:** Concrete dependencies
**After:** Depend on abstractions

```python
# Before - depends on Unsloth directly
from unsloth import FastLanguageModel
model, tok = FastLanguageModel.from_pretrained(...)

# After - depends on abstraction
class Merged16BitStrategy(BaseSaveStrategy):
    def __init__(self, model_loader: IModelLoader):  # Abstraction
        self.model_loader = model_loader

    def _execute_save(self, ...):
        model, tok = self.model_loader.load_model(...)  # Uses abstraction
```

---

## Appendix C: Extension Examples

### Example 1: Adding AWQ Quantization

**Step 1:** Create strategy
```python
# shared/upload/strategies/awq.py
class AWQSaveStrategy(BaseSaveStrategy):
    def _execute_save(self, model_path, output_dir, **kwargs):
        # AWQ quantization logic
        pass

    def estimate_size_gb(self, model_size: str) -> float:
        return {"3b": 2.0, "7b": 3.5, "13b": 7.0}[model_size]

    def requires_gpu(self) -> bool:
        return True
```

**Step 2:** Register strategy
```python
# shared/upload/strategies/__init__.py
from .awq import AWQSaveStrategy
SaveStrategyRegistry.register("awq", AWQSaveStrategy)
```

**Step 3:** Use it
```bash
python upload_model.py ... --save-method awq
```

**Total changes:** 1 new file, 2 lines added to registry. **No core code modified.**

### Example 2: Adding New Trainer

**Create wrapper:**
```python
# Trainers/rtx3090_dpo/src/upload_to_hf.py
import sys, os
from pathlib import Path
SHARED_PATH = Path(__file__).parent.parent.parent / "shared"
sys.path.insert(0, str(SHARED_PATH))
os.environ.setdefault("TRAINER_OUTPUT_DIR", "dpo_output_rtx3090")
from upload.cli.upload_cli import main
if __name__ == "__main__":
    main()
```

**Create shell script:**
```bash
# Trainers/rtx3090_dpo/upload_model.sh
#!/bin/bash
export TRAINER_OUTPUT_DIR="./dpo_output_rtx3090"
source ../scripts/upload_interactive.sh "$TRAINER_OUTPUT_DIR"
```

**Total effort:** 30 lines of code, 5 minutes of work.

### Example 3: Adding Ollama Direct Upload

**Step 1:** Create uploader
```python
# shared/upload/uploaders/ollama.py
class OllamaUploader(IUploader):
    def upload_model(self, local_path, repo_id, credential, **options):
        # Call Ollama API
        pass
```

**Step 2:** Register
```python
UploaderRegistry.register("ollama", OllamaUploader)
```

**Step 3:** Use
```bash
python upload_model.py ... --uploader ollama
```

---

*End of Architecture Document*
