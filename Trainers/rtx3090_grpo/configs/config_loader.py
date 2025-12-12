"""
YAML configuration loader for GRPO / GSPO training.
Loads configs/config.yaml and converts it into Python dataclass objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def load_yaml_config(config_path: str | None = None) -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Args:
        config_path: Path to config.yaml (defaults to configs/config.yaml)

    Returns:
        Dictionary with configuration values
    """
    if config_path is None:
        config_path = str(Path(__file__).parent / "config.yaml")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@dataclass
class ModelConfig:
    model_name: str
    max_seq_length: int
    dtype: Optional[str]
    load_in_4bit: bool
    chat_template: Optional[str] = None


@dataclass
class LoRAConfig:
    r: int
    lora_alpha: int
    lora_dropout: float
    bias: str
    target_modules: List[str]
    use_gradient_checkpointing: str
    random_state: int
    use_rslora: bool = False
    use_dora: bool = False


@dataclass
class GRPOTrainingConfig:
    output_dir: str
    per_device_train_batch_size: int
    gradient_accumulation_steps: int

    num_generations: int
    max_prompt_length: int
    max_completion_length: int
    temperature: float

    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    lr_scheduler_type: str
    num_train_epochs: int
    max_steps: int

    fp16: bool
    bf16: bool
    optim: str

    logging_steps: int
    save_steps: int
    save_total_limit: int

    report_to: str
    use_gspo: bool = False
    extra_args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetConfig:
    dataset_name: str
    dataset_file: str
    local_file: Optional[str]
    num_proc: int
    prompt_column: str = "prompt"


@dataclass
class WandbConfig:
    enabled: bool
    project: Optional[str]
    run_name: Optional[str]
    entity: Optional[str]


@dataclass
class Config:
    model: ModelConfig
    lora: LoRAConfig
    training: GRPOTrainingConfig
    dataset: DatasetConfig
    wandb: WandbConfig
    rewards: Dict[str, Any] = field(default_factory=dict)
    seed: int = 42

    @property
    def use_wandb(self) -> bool:
        return bool(self.wandb.enabled)

    @use_wandb.setter
    def use_wandb(self, value: bool):
        self.wandb.enabled = bool(value)

    @property
    def wandb_project(self) -> Optional[str]:
        return self.wandb.project

    @wandb_project.setter
    def wandb_project(self, value: Optional[str]):
        self.wandb.project = value

    @property
    def wandb_run_name(self) -> Optional[str]:
        return self.wandb.run_name

    @wandb_run_name.setter
    def wandb_run_name(self, value: Optional[str]):
        self.wandb.run_name = value


def dict_to_dataclass(cls, data: Dict[str, Any]):
    """
    Convert dictionary to dataclass instance.
    Ignores unknown keys, and converts numeric strings to numbers.
    """
    import typing

    fieldtypes = {f.name: f.type for f in cls.__dataclass_fields__.values()}
    converted_data: Dict[str, Any] = {}

    for key, value in (data or {}).items():
        if key not in fieldtypes:
            continue

        field_type = fieldtypes[key]

        # Handle Optional[T]
        if hasattr(field_type, "__origin__") and field_type.__origin__ is typing.Union:
            types = [t for t in field_type.__args__ if t is not type(None)]
            if types:
                field_type = types[0]

        if field_type == float and isinstance(value, str):
            converted_data[key] = float(value)
        elif field_type == int and isinstance(value, str):
            converted_data[key] = int(value)
        else:
            converted_data[key] = value

    return cls(**converted_data)


def load_config(config_path: str | None = None) -> Config:
    yaml_config = load_yaml_config(config_path)

    model_config = dict_to_dataclass(ModelConfig, yaml_config.get("model", {}))
    lora_config = dict_to_dataclass(LoRAConfig, yaml_config.get("lora", {}))
    training_config = dict_to_dataclass(GRPOTrainingConfig, yaml_config.get("training", {}))
    dataset_config = dict_to_dataclass(DatasetConfig, yaml_config.get("dataset", {}))
    wandb_config = dict_to_dataclass(WandbConfig, yaml_config.get("wandb", {}))
    rewards_config = yaml_config.get("rewards", {}) or {}

    return Config(
        model=model_config,
        lora=lora_config,
        training=training_config,
        dataset=dataset_config,
        wandb=wandb_config,
        rewards=rewards_config,
        seed=int(yaml_config.get("seed", 42)),
    )
