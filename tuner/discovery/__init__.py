"""
Discovery services for the tuner package.

Location: /mnt/f/Code/Toolset-Training/tuner/discovery/__init__.py
Purpose: Export all discovery service classes for resource enumeration
Used by: Handlers to discover training runs, checkpoints, models, and prompt sets

This module provides discovery services that scan the filesystem and external
systems to enumerate available resources:
- TrainingRunDiscovery: Find training run directories
- CheckpointDiscovery: Find and analyze checkpoints with metrics
- ModelDiscovery: List available models from evaluation backends
- PromptSetDiscovery: Find and parse prompt sets
- DatasetDiscovery: Find and analyze JSONL datasets
- RubricDiscovery: Find rubric YAML files
- BaseModelDiscovery: Find base and fine-tuned models
"""

from tuner.discovery.training_runs import TrainingRunDiscovery
from tuner.discovery.checkpoints import CheckpointDiscovery
from tuner.discovery.models import ModelDiscovery
from tuner.discovery.prompt_sets import PromptSetDiscovery, PromptSetInfo
from tuner.discovery.datasets import DatasetDiscovery, DatasetInfo
from tuner.discovery.rubrics import RubricDiscovery, RubricInfo
from tuner.discovery.base_models import BaseModelDiscovery, ModelInfo

__all__ = [
    "TrainingRunDiscovery",
    "CheckpointDiscovery",
    "ModelDiscovery",
    "PromptSetDiscovery",
    "PromptSetInfo",
    "DatasetDiscovery",
    "DatasetInfo",
    "RubricDiscovery",
    "RubricInfo",
    "BaseModelDiscovery",
    "ModelInfo",
]
