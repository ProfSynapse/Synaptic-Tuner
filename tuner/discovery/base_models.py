"""
Base model discovery service.

Location: /mnt/f/Code/Toolset-Training/tuner/discovery/base_models.py
Purpose: Discover base models from trainer configs and local fine-tuned models
Used by: List handler to display available models (base and fine-tuned)

This module implements the BaseModelDiscovery service which reads trainer
configuration files to find configured base models and scans output directories
for fine-tuned models.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import yaml

from shared.utilities.paths import TRAINING_METHODS, get_trainer_root, iter_training_output_dirs
from tuner.discovery.training_runs import TrainingRunDiscovery
from tuner.project import ProjectContext


@dataclass
class ModelInfo:
    """Information about a discovered model."""
    name: str
    model_type: str  # 'base' or 'finetuned'
    source: str  # 'HuggingFace', 'SFT', 'KTO', 'GRPO'
    path: Optional[str]  # Local path for fine-tuned models
    declaring_root: Optional[Path] = None


class BaseModelDiscovery:
    """
    Discover available base and fine-tuned models.

    This service:
    1. Reads trainer configs to find configured base models (HuggingFace)
    2. Scans output directories to find local fine-tuned models

    Example:
        from tuner.discovery import BaseModelDiscovery

        discovery = BaseModelDiscovery()
        models = discovery.discover_all()

        for model in models:
            print(f"{model.name} ({model.model_type}) from {model.source}")
    """

    # Known base models (commented out in config but commonly used)
    KNOWN_BASE_MODELS = [
        "Qwen/Qwen3.5-0.8B",
        "Qwen/Qwen3.5-2B",
        "Qwen/Qwen3.5-4B",
        "Qwen/Qwen3.5-9B",
        "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1",
        "unsloth/Qwen3-1.7B-unsloth-bnb-4bit",
        "unsloth/Qwen3-4B-unsloth-bnb-4bit",
        "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
        "unsloth/llama-3.1-8b-instruct-bnb-4bit",
        "unsloth/mistral-7b-v0.3-bnb-4bit",
    ]

    def __init__(
        self,
        repo_root: Path = None,
        *,
        context: ProjectContext | None = None,
    ):
        """
        Initialize the base model discovery service.

        Args:
            repo_root: Repository root path. If None, uses module location to find repo root.
        """
        self.context = context
        self.repo_root = context.engine_root if context else (repo_root or Path(__file__).parent.parent.parent)

    def discover_all(self) -> Tuple[List[ModelInfo], List[ModelInfo]]:
        """
        Discover all available models.

        Returns:
            Tuple of (base_models, finetuned_models):
            - base_models: List of ModelInfo for HuggingFace base models
            - finetuned_models: List of ModelInfo for local fine-tuned models
        """
        base_models = self._discover_base_models()
        finetuned_models = self._discover_finetuned_models()

        return base_models, finetuned_models

    def _discover_base_models(self) -> List[ModelInfo]:
        """
        Discover base models from trainer configs.

        Returns:
            List of ModelInfo for base models.
        """
        results: List[ModelInfo] = []
        seen_models = set()

        # Check each trainer config
        config_paths: list[Path] = []
        if self.context is not None and self.context.mode == "host" and self.context.config_root.exists():
            config_paths.extend(sorted(self.context.config_root.rglob("*.yaml")))
        config_paths.extend(
            [
                get_trainer_root("sft", self.repo_root) / "configs" / "config.yaml",
                get_trainer_root("kto", self.repo_root) / "configs" / "config.yaml",
            ]
        )

        for config_path in config_paths:
            if config_path.exists():
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f)

                    model_name = config.get('model', {}).get('model_name')
                    if model_name and model_name not in seen_models:
                        results.append(ModelInfo(
                            name=model_name,
                            model_type='base',
                            source='HuggingFace',
                            path=None,
                            declaring_root=config_path.parent,
                        ))
                        seen_models.add(model_name)
                except Exception:
                    continue

        # Add known base models that aren't in configs
        for model_name in self.KNOWN_BASE_MODELS:
            if model_name not in seen_models:
                results.append(ModelInfo(
                    name=model_name,
                    model_type='base',
                    source='HuggingFace',
                    path=None,
                    declaring_root=self.repo_root,
                ))
                seen_models.add(model_name)

        return results

    def _discover_finetuned_models(self) -> List[ModelInfo]:
        """
        Discover fine-tuned models from output directories.

        Returns:
            List of ModelInfo for fine-tuned models.
        """
        results: List[ModelInfo] = []

        # Trainer output directories. Derived from the TRAINING_METHODS SSOT
        # (shared/utilities/paths.py) so a trained run of ANY registered method —
        # including embedding and ace_step — is discoverable here and never silently
        # invisible. iter_training_output_dirs() resolves every method's dir (verified),
        # and the inner final_model/ check naturally skips methods without one.
        discovery = TrainingRunDiscovery(repo_root=self.repo_root, context=self.context)
        for trainer_type in TRAINING_METHODS:
            for run_dir in discovery.discover(trainer_type, limit=None):
                final_model = run_dir / "final_model"
                if not final_model.exists():
                    continue
                try:
                    relative_path = str(final_model.relative_to(self.context.project_root if self.context else self.repo_root))
                except ValueError:
                    relative_path = str(final_model)
                results.append(ModelInfo(
                    name=run_dir.name,
                    model_type='finetuned',
                    source=trainer_type.upper(),
                    path=relative_path,
                    declaring_root=next(
                        (root for root in discovery.output_roots(trainer_type) if final_model.is_relative_to(root)),
                        final_model.parent,
                    ),
                ))

        return results
